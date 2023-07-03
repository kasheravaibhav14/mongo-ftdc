import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot as plt
from pprint import pprint
from PyPDF4 import PdfFileReader, PdfFileWriter
from FTDC_plot import FTDC_plot
import multiprocessing as mp
from scipy.stats import zscore


class FTDC_an:
    def __init__(self, metricObj, qTstamp, outPDFpath,duration):
        self.metricObj = metricObj
        self.queryTimeStamp = qTstamp
        self.ticketlim = 50  # when available tickets are more than 15, we assume that the system is in normal state or is heading toward crash
        self.tdelta = duration
        self.threshold = 0.25
        self.outPDF = outPDFpath
        self.nbuckets = 5

    def __plot(self, df, to_monitor, vert_x, main_title="metric-A", outfilename="fig.pdf"):
        to_monitor.sort()
        df1=df[to_monitor]
        end_index = self.outPDF.rindex("/")
        outlierPath = self.outPDF[:end_index]+'/outliers.csv'
        df.to_csv(outlierPath,index=True)
        # print("Monitoring: ", str(to_monitor))
        print(len(to_monitor))
        plot_ob = FTDC_plot(df,to_monitor,vert_x,self.outPDF)
        plot_ob.plot()
        return

    def __partitionPlot(self, df, to_monitor, vert_x, outfilename="fig.pdf"):
        def merge_pdfs(file_list, output):
            # Create a PDF writer object
            pdf_writer = PdfFileWriter()

            # Loop through list of files
            for file in file_list:
                pdf_reader = PdfFileReader(file)
                pdf_writer.addPage(pdf_reader.getPage(0))

            # Write out the merged PDF
            with open(output, 'wb') as out:
                pdf_writer.write(out)
        to_monitor.sort()
        mypart = {
            'ss_wt': [],
            'ss_metric': [],
            'ss': [],
            'sm': [],
            'etc': []
        }
        for met in to_monitor:
            if "sm" in met:
                mypart["sm"].append(met)
                continue
            if "ss wt" in met:
                mypart["ss_wt"].append(met)
            elif "ss metric" in met:
                mypart["ss_metric"].append(met)
            elif "ss" in met:
                mypart["ss"].append(met)
            else:
                mypart["etc"].append(met)
        ofnames = []

        for key, val in mypart.items():
            print(key, len(val))
            if len(val) == 0:
                continue
            ofilename = "fig_"+key+".pdf"
            ofnames.append(ofilename)
            self.__plot(df, val, vert_x, key, ofilename)
        merge_pdfs(ofnames, outfilename)

    def __renameCols(self, df):
        rename_cols = {}
        for col_name in df.columns:
            ccol = col_name.replace("serverStatus.", "ss ")
            ccol = ccol.replace("wiredTiger.", "wt ")
            ccol = ccol.replace("systemMetrics.", "sm ")
            ccol = ccol.replace("tcmalloc.tcmalloc", "tcmalloc")
            # ccol = ccol.replace("transaction.transaction", "txn")
            ccol = ccol.replace("transaction.", "txn ")
            ccol = ccol.replace("local.oplog.rs.stats.", "locOplogRsStats ")
            ccol = ccol.replace("aggStageCounters.", "aggCnt ")
            # print(col_name, ccol)
            rename_cols[col_name] = ccol
        df.rename(columns=rename_cols, inplace=True)

    def __meanCalc(self, df, tbounds):
        curr_mean = {}
        prev_mean = {}
        for col in df.columns:
            curr_mean[col] = df.iloc[tbounds['c_lb']
                :tbounds['c_ub']+1][col].mean()
            prev_mean[col] = df.iloc[tbounds['p_lb']
                :tbounds['c_lb']+1][col].mean()
        return curr_mean, prev_mean

    def __findOutliersWithZscore(self, df, field):
        colname = field
        df['z_score'] = (df[colname] - df[colname].mean()) / df[colname].std()
        threshold = 7
        df['is_outlier'] = np.abs(df['z_score']) > threshold
        plt.figure(figsize=(20, 10))
        plt.plot(df.index, df[field], label=field)
        outliers = df[df['is_outlier'] == True]
        plt.scatter(outliers.index,
                    outliers[field], color='red', label='Outliers')
        plt.title('Z-Score Outlier detection')
        plt.xlabel('Time')
        plt.ylabel(field)
        plt.legend()
        plt.show()

    def __findOutliersWithIsoForest(self, df, field):
        columns = [field]
        X = df[columns]
        clf = IsolationForest(contamination=0.01)
        clf.fit(X)
        outliers = clf.predict(X) == -1
        X['IsOutlier'] = outliers
        plt.figure(figsize=(20, 10))
        plt.plot(X.index, X[field], label=field)
        outliers = X[X['IsOutlier'] == True]
        plt.scatter(outliers.index,
                    outliers[field], color='red', label='Outliers')
        plt.title('Isolation Forest Outlier detection')
        plt.xlabel('Time')
        plt.ylabel(field)
        plt.legend()
        plt.show()

    def getDTObj(self, date_string):
        format_string = "%Y-%m-%d_%H-%M-%S"
        parsed_datetime = datetime.strptime(date_string, format_string)
        return parsed_datetime

    def getDTFromSecs(self, ms):
        return datetime.fromtimestamp(ms/1000)

    def __getDirtyFillRatio(self, metricObj):
        total_cache = metricObj["serverStatus.wiredTiger.cache.bytes currently in the cache"]
        dirty_cache = metricObj["serverStatus.wiredTiger.cache.tracked dirty bytes in the cache"]
        metricObj["ss wt cache dirty fill ratio"] = []
        for idx in range(len(total_cache)):
            if total_cache[idx] != 0:
                ratio = (dirty_cache[idx] / total_cache[idx])
            else:
                ratio = 0
            metricObj["ss wt cache dirty fill ratio"].append(100*ratio)

    def __getCacheFillRatio(self, metricObj):
        total_cache = metricObj["serverStatus.wiredTiger.cache.maximum bytes configured"]
        curr_cache = metricObj["serverStatus.wiredTiger.cache.bytes currently in the cache"]
        metricObj["ss wt cache fill ratio"] = []
        
        for idx in range(len(total_cache)):
            if total_cache[idx] != 0:
                ratio = (curr_cache[idx] / total_cache[idx])
            else:
                ratio = 0
            metricObj["ss wt cache fill ratio"].append(100*ratio)

    def __getMemoryFragRatio(self, metricObj):

        tCache = "serverStatus.tcmalloc.generic.current_allocated_bytes"
        trCache = "serverStatus.tcmalloc.generic.heap_size"
        nkey = "serverStatus.wiredTiger.memory fragmentation ratio"
        if trCache not in metricObj or tCache not in metricObj:
            return
        metricObj[nkey] = []
        for idx in range(len(metricObj[trCache])):
            if metricObj[trCache][idx] != 0:
                metricObj[nkey].append(
                    100*((metricObj[trCache][idx]-metricObj[tCache][idx])/metricObj[trCache][idx]))
            else:
                metricObj[nkey].append(-1)

    def __getAverageLatencies(self, metricObj):
        base = "serverStatus.opLatencies."
        for command in ["reads.", "writes.", "commands.", "transactions."]:
            opkey = base+command+"ops"
            ltkey = base+command+"latency"
            if opkey in metricObj:
                for idx in range(len(metricObj[opkey])):
                    if metricObj[opkey][idx] != 0:
                        metricObj[ltkey][idx] = metricObj[ltkey][idx] / \
                            metricObj[opkey][idx]
    def __diskUtilization(self,metricObj):
        disks=[]
        for key in metricObj:
            if key.startswith("systemMetrics.disks"):
                print(key)
                mystr=key
                disk=mystr.split("systemMetrics.disks.")[1].split('.')[0]
                if disk not in disks:
                    disks.append(disk)
        
        for disk in disks:
            io="systemMetrics.disks."+disk+".io_time_ms"
            queue="systemMetrics.disks."+disk+".io_queued_ms"
            newkey="systemMetrics.disks."+disk+" utilization"
            if io not in metricObj or queue not in metricObj:
                continue
            metricObj[newkey]=[]
            for idx in range(len(metricObj[io])):
                if metricObj[io][idx]==0:
                    metricObj[newkey].append(0)
                else:
                    metricObj[newkey].append((100*metricObj[io][idx])/(metricObj[io][idx]+metricObj[queue][idx]))

    def __tcmallocminuswt(self, metricObj):
        wtcache = "serverStatus.wiredTiger.cache.bytes currently in the cache"
        tcmalloc = "serverStatus.tcmalloc.generic.current_allocated_bytes"
        newkey = "serverStatus.wiredTiger.tcmalloc derived: allocated minus wt cache MiB"
        if wtcache not in metricObj or tcmalloc not in metricObj:
            return
        itr = 0
        mib_conv = 2**20
        itr += 1
        metricObj[newkey] = []
        for idx in range(len(metricObj[wtcache])):
            metricObj[newkey].append(
                (metricObj[tcmalloc][idx]-metricObj[wtcache][idx])/mib_conv)

    def calcBounds(self, df, pos, delt): #if our bucket is 5 mins each, delt is 2.5 mins
        tbounds=[]
        t0=-1
        # for idx in range(pos, -1, -1):
        #     if df.iloc[idx]['ss wt concurrentTransactions.write.available'] > self.ticketlim \
        #             and df.iloc[idx]['ss wt concurrentTransactions.read.available'] > self.ticketlim:# work on it
        #         t0 = idx
        #         break
        pos1=pos-600
        for idx in range(pos1,pos1+900):
            print(df.index[idx])
            if df.iloc[idx]['ss wt concurrentTransactions.write.available'] < self.ticketlim \
                    or df.iloc[idx]['ss wt concurrentTransactions.read.available'] < self.ticketlim:
                t0=idx
                print("found ticket drop at:",df.index[t0])
                break
        print(t0)
        idx=t0+delt #
        while(not df.index[idx] and idx <len(df)):
            idx+=1
        tbounds.insert(0,idx)
        for i in range(0,self.nbuckets):
            idx-=2*delt
            while(not df.index[idx] and idx>0):
                idx-=1
            tbounds.insert(0,idx)
        print(pos)
        print(tbounds)
        return tbounds
    
    def has_outliers(self,data, multiplier=1.1):
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        dev=multiplier*IQR
        # Compute lower and upper bounds for outliers
        lower_bound = Q1 -dev
        upper_bound = Q3 +dev

        # Check if any value is an outlier
        ctr=0
        curr_ctr=0
        curr_inc=None
        for value in data:
            # nmean=np.mean(data)
            if value < lower_bound or value > upper_bound :
                nmean=(np.sum(data)-value)/(len(data)-1)
                if nmean==0:
                    continue
                inc=(value-nmean)/nmean
                if abs(inc)>=0.05:
                    if curr_inc==None:
                        curr_inc=inc
                        curr_ctr=ctr
                    elif ctr>curr_ctr:
                        curr_inc=inc
                        curr_ctr=ctr
                    print(value)
                    # return True,ctr,inc
            ctr+=1
        
        if curr_inc!=None and curr_ctr>=len(data)-3:
            return True,curr_ctr,curr_inc

        return False,ctr,0
    def percentileChange(self,data,mean):
        print(data)
        rate_of_increase = [((data[i+1] - data[i])/data[i]) if data[i]!=0 else 0 for i in range(len(data)-1)]
        pc=0;nc=0
        for r in rate_of_increase:
            if r<0 and r<-0.5:
                nc+=1
            elif r>0 and r>0.5:
                pc+=1
        if all(m<=1.25*np.mean(mean) for m in mean):
            return False
        if pc >= len(rate_of_increase)//2 or nc >=len(rate_of_increase)//2:
            return True
        return False
    def checkMetric(self,df, met, timebounds):
        df1=df.iloc[timebounds[0]:timebounds[-1]]
        maxval=df1[met].max()
        maxpos= np.argmax(df1[met].values)
        bucket=maxpos//self.tdelta
        if "ss wt cache fill ratio"==met and maxval>=80:
            nmean = df1[met].mean()
            print(met,bucket,nmean)
            ch= (maxval-nmean)/nmean
            return True,bucket,(nmean,ch)
        if "ss wt cache dirty fill ratio"==met and maxval>=5:
            nmean = df1[met].mean()
            print(met,bucket,nmean)
            ch= (maxval-nmean)/nmean
            return True,bucket,(nmean,ch)
    # print(met)
        p99=[]
        mean=[]
        for t in range(0,len(timebounds)-1):
            temp=df.iloc[timebounds[t]:timebounds[t+1]][met].values
            p99.append(np.percentile(temp,99))
            mean.append(np.mean(temp))
        print(met)
        print(mean,p99)
        tr,idx,inc=self.has_outliers(mean)
        tr1 = self.percentileChange(p99,mean)
        if tr or tr1:
            return True,idx,(np.mean(mean),inc)
        return False,idx,(0,0,0)

    def analytics(self, metricObj, queryTimestamp):
        self.__getAverageLatencies(metricObj)
        self.__tcmallocminuswt(metricObj)
        self.__getMemoryFragRatio(metricObj)
        self.__getDirtyFillRatio(metricObj)
        self.__getCacheFillRatio(metricObj)
        # self.__diskUtilization(metricObj)
        un_len = {}  # debug
        for key in metricObj:
            if len(metricObj[key]) not in un_len:
                # print(key, len(metricObj[key]))
                un_len[len(metricObj[key])] = [key]
            else:
                un_len[(len(metricObj[key]))].append(key)
        pprint(un_len)
        df = pd.DataFrame(metricObj)
        wstr = ''
        for col in df.columns:
            wstr = wstr+col+'\n'
        with open('metricList','w') as filew:
            filew.write(wstr)
        print(df)
        df['serverStatus.start'] = df['serverStatus.start'].apply(
            self.getDTFromSecs)
        df['serverStatus.start'] = pd.to_datetime(df['serverStatus.start'])
        df.set_index('serverStatus.start', inplace=True)
        df.columns.name = 'metrics'
        df.to_csv('./cases/1.csv')
        print(df)
        pos = np.where(df.index == queryTimestamp)[0][0]
        # self.__findOutliersWithZscore(
        #     df, 'serverStatus.wiredTiger.concurrentTransactions.write.out')
        # self.__findOutliersWithZscore(df, 'serverStatus.wiredTiger.concurrentTransactions.read.out')
        self.__renameCols(df)
        # self.hourlyAnalytics(df)
        to_monitor = []
        
        # return
        tbounds = self.calcBounds(df, pos, self.tdelta//2)
        # print(df.index[pos])
        for ky in tbounds:
            print(ky, df.index[ky])
        # gpt_str = gpt_str_base
        # curr_mean, prev_mean = self.__meanCalc(df, tbounds)
        anomaly_map={}

        for metric in df.columns:
            try:
                tr,val,inc=self.checkMetric(df,metric, tbounds)
            except Exception as e:
                print(self.checkMetric(df,metric,tbounds))
                # exit(1)
            if tr:
                to_monitor.append(metric)
                if val not in anomaly_map:
                    anomaly_map[val]=[]
                anomaly_map[val].append([metric,inc])
            # break
        print(to_monitor)
        print(len(to_monitor))
        gpt_str_base = f'''My mongodb server has seen a sheer drop in write Tickets at Timestamp {queryTimestamp}. The task is to perform root cause analysis of the ticket drop, so I want you to locate what was the cause of the ticket drop. FYI, a ticket drop means that concurrentTransactions.write.out has increased and keeps pushing towards 128(its maximum configured limit). Use your own understanding of mongodb and wiredtiger metrics to analyse the data. In the given data, each timestamp has a list of anomalous metrics followed by the percentage change from the mean and the mean, all space separated. A negative value means the value has decreased compared to the mean value of the metric. A negative value at an older timestamp would mean that the metric was lower in value previously. Using the data, try to locate the root cause of the ticket drop.  First give a summary of the analysis and then move to the explanation. Structure the output properly. Do not leave out any important information/relevant metric. 

When analyzing, please consider:

Cache dirty ratio > 5% and/or Cache fill ratio > 80%, which indicate that cache eviction has started by worker threads.
Cache dirty ratio > 20% and/or Cache fill ratio > 95%, which indicate that cache eviction has started by application threads.
'cursor.cached cursor count', the number of cursors currently cached by the WiredTiger storage engine.
'history store score', an indicator of the pressure the history store is placing on the cache.
However, do not limit the analysis to these pointers. Each timestamp represents a 10-minute interval following it, for example, 13:07:00 represents the period from 13:07:00 to 13:16:59. Hence, do not use exact timestamps in your response.

Provide a brief summary of the analysis first, then delve into detailed explanations. Structure your response effectively and ensure all relevant information/metrics are included.

Please note the abbreviations:

'sm' stands for system metrics
'ss' stands for server status mongo db
'wt' stands for wiredtiger.'''
        sorted_keys = sorted(anomaly_map.keys())
        for ts in sorted_keys:
            tsss=df.index[tbounds[ts]]
            gpt_str_base+=f"{tsss}:\n"
            for val in anomaly_map[ts]:
                gpt_str_base+=f"{val[0]} {val[1][1]*100:.2f}% {val[1][0]}\n"
        with open("gpt-input.txt",'w') as gpt:
            gpt.write(gpt_str_base)
        # return
            
        #     try:
        #         tr, val = checkMetric(
        #             df.iloc[tbounds['p_lb']:tbounds['c_ub']+1], metric)
        vertical = (df.index[(tbounds[-1]+tbounds[-2])//2])
        self.__plot(df.iloc[tbounds[0]:tbounds[-1]],
                             to_monitor,main_title="metric-A", vert_x=vertical)

    def parseAll(self):
        def delta(metrList):
            mylst = [metrList[i] for i in range(len(metrList))]
            for i in range(1, len(metrList)):
                mylst[i] -= metrList[i-1]
            return mylst

        def checkCriteria(met):
            # if met.startswith("serverStatus.metrics.aggStageCounters") or met.startswith("serverStatus.metrics.commands"):
            #     return True
            if met.startswith("systemMetrics.disks"):
                return True
            if met.startswith("replSetGetStatus.members") and (met.endswith("state") or met.endswith("health")):
                return True
            return False

        data = {}
        iter_keys = iter(self.metricObj)
        # extract the first level of data
        date_string = next(iter_keys)
        data = {}
        metObj = self.metricObj[date_string]
        selected_keys = json.load(open('FTDC_metrics.json', 'r'))
        sel_metr_c = selected_keys["to_monitor_c"]
        sel_metr_p = selected_keys["to_monitor_p"]
        locks = selected_keys["locks"]
        for lk in locks["type"]:
            for ops in locks["ops"]:
                for mode in locks["mode"]:
                    new_c_met = "serverStatus.locks."+lk+"."+ops+"."+mode
                    sel_metr_c.append(new_c_met)

        deltactr = len(metObj["serverStatus.start"])
        delta1 = 0
        for met in metObj:
            if met in sel_metr_p:
                data[met] = metObj[met]

        for met in metObj:
            # checkCriteria implements string matching for certain cumulative metrics
            if met in sel_metr_c or checkCriteria(met):
                data[met] = delta(metObj[met])

        for key in iter_keys:
            metObj = self.metricObj[key]
            sel_metr_c_new = [s for s in metObj.keys() if (
                s in sel_metr_c or checkCriteria(s))]
            sel_metr_p_new = [s for s in metObj.keys() if s in sel_metr_p]

            '''
            handle edge case that a certain metric gets acquired halfway which was not present in the initial list
            eg. serverStatus.locks.Global.acquireWaitTime
            '''
            new_c = [
                item for item in sel_metr_c_new if item not in data]  # metric not in data ever before
            new_p = [item for item in sel_metr_p_new if item not in data]

            for met in new_c:
                # add zeros for those metric who have never been there in the data before
                data[met] = [0 for i in range(deltactr)]
                print("occurence of new accumulate metric", met)

            for met in new_p:
                data[met] = [0 for i in range(deltactr)]
                print("occurence of new point metric", met)

            for met in sel_metr_p_new:
                # now fill all the values obtained
                data[met].extend(metObj[met])
            for met in sel_metr_c_new:
                data[met].extend(delta(metObj[met]))
            delta1 = len(metObj[sel_metr_p_new[0]])

            # handle the case where unusual number of nmetrics or ndeltas occur,
            # i.e. less metrics are reported compared to previous iteration, so fill with zeros for point in time data and previous value for accumulative data
            for met in data:
                if met not in sel_metr_p_new and not checkCriteria(met) and met not in sel_metr_c:
                    data[met].extend([0] * delta1)
                elif met not in sel_metr_c_new and met not in sel_metr_p:
                    prev_val = data[met][-1]
                    data[met].extend([prev_val] * delta1)

            deltactr += len(metObj["serverStatus.start"])
        print(len(data.keys()))
        print(deltactr)
        # 2023-06-09 12:03:28
        self.analytics(data, self.queryTimeStamp)
