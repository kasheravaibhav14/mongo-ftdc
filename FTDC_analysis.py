import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot as plt
from pprint import pprint
from FTDC_plot import FTDC_plot


class FTDC_an:
    def __init__(self, metricObj, qTstamp, outPDFpath, duration):
        self.metricObj = metricObj
        self.queryTimeStamp = qTstamp
        self.ticketlim = 50
        self.tdelta = duration
        self.outPDF = outPDFpath
        self.nbuckets = 12
        self.meanThreshold =1.1

    def __plot(self, df, to_monitor, vert_x, main_title="metric-A", outfilename="fig.pdf"):
        to_monitor.sort()
        df1 = df[to_monitor]
        end_index = self.outPDF.rindex("/")
        outlierPath = self.outPDF[:end_index]+'/outliers.csv'
        df.to_csv(outlierPath, index=True)
        # print("Monitoring: ", str(to_monitor))
        print(len(to_monitor))
        plot_ob = FTDC_plot(df, to_monitor, vert_x, self.outPDF)
        plot_ob.plot()
        return

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

    def __diskUtilization(self, metricObj):
        disks = []
        for key in metricObj:
            if key.startswith("systemMetrics.disks"):
                print(key)
                mystr = key
                disk = mystr.split("systemMetrics.disks.")[1].split('.')[0]
                if disk not in disks:
                    disks.append(disk)

        for disk in disks:
            io = "systemMetrics.disks."+disk+".io_time_ms"
            queue = "systemMetrics.disks."+disk+".io_queued_ms"
            newkey = "systemMetrics.disks."+disk+" utilization"
            if io not in metricObj or queue not in metricObj:
                continue
            metricObj[newkey] = []
            for idx in range(len(metricObj[io])):
                if metricObj[io][idx] == 0:
                    metricObj[newkey].append(0)
                else:
                    metricObj[newkey].append(
                        (100*metricObj[io][idx])/(metricObj[io][idx]+metricObj[queue][idx]))

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

    def calcBounds(self, df, pos, delt):  # if our bucket is 5 mins each, delt is 2.5 mins
        tbounds = []
        t0 = -1
        pos1 = pos-self.tdelta*2
        for idx in range(pos1, pos1+3*self.tdelta):
            print(df.index[idx])
            if df.iloc[idx]['ss wt concurrentTransactions.write.available'] < self.ticketlim \
                    or df.iloc[idx]['ss wt concurrentTransactions.read.available'] < self.ticketlim:
                t0 = idx
                print("found ticket drop at:", df.index[t0])
                break
        print(t0)
        idx = t0+delt
        for i in range(0,self.nbuckets//6):
            while (not df.index[idx] and idx < len(df)):
                idx += 1
            tbounds.append(idx)
            idx+=2*delt
        idx=t0+delt
        for i in range(0, self.nbuckets):
            idx -= 2*delt
            while (not df.index[idx] and idx > 0):
                idx -= 1
            tbounds.insert(0, idx)
        print(pos)
        print(tbounds)
        return tbounds,t0

    def has_outliers(self, data, multiplier=1.1):
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        dev = multiplier*IQR
        # Compute lower and upper bounds for outliers
        lower_bound = Q1 - dev
        upper_bound = Q3 + dev

        # Check if any value is an outlier
        ctr = 0
        curr_ctr = 0
        curr_val = None
        for value in data:
            # nmean=np.mean(data)
            if value < lower_bound or value > upper_bound:
                if curr_val == None:
                    curr_val = value
                    curr_ctr = ctr
                elif ctr > curr_ctr:
                    curr_val = value
                    curr_ctr = ctr
                print(value, curr_ctr)
                # return True,ctr,inc
            ctr += 1

        if curr_val != None and curr_ctr >= len(data)-3:
            return True, curr_ctr
        return False, ctr

    def percentileChange(self, data):
        data = np.array(data).reshape(-1, 1)
        iso_forest = IsolationForest(contamination=0.05)
        iso_forest.fit(data)
        pred = iso_forest.predict(data)
        outliers = data[pred == -1]
        indices = np.where(outliers)
        if len(outliers) > 0 and indices[-1][0]>=len(data)-5:
            print("outliers:", outliers,indices[-1][0])
            return True, indices[-1][0]
        return False, 0

    def checkMetric(self, df, met, timebounds):
        df1 = df.iloc[timebounds[0]:timebounds[-1]]
        maxval = df1[met].max()
        maxpos = np.argmax(df1[met].values)
        bucket = maxpos//self.tdelta
        if ("ss wt cache fill ratio" == met and maxval >= 80) or ("ss wt cache dirty fill ratio" == met and maxval >= 5) or ("ss wt cache.history store score" == met and maxval >= 50):
            nmean = df1[met].mean()
            # print(met, bucket, nmean)
            ch = (maxval-nmean)/nmean
            return True, bucket, (0, nmean, maxval)
        p99 = []
        mean = []
        for t in range(0, len(timebounds)-1):
            temp = df.iloc[timebounds[t]:timebounds[t+1]][met].values
            # print(temp)
            p99.append(np.percentile(temp, 99))
            mean.append(np.mean(temp))
        # print(met)
        # print(mean, p99)
        if np.mean(mean) == 0 or (np.mean(mean) != 0 and all(abs(m/np.mean(mean)) <=  self.meanThreshold for m in mean)):
            return False, 0, ()
        tr, idx = self.has_outliers(mean)
        tr1, idx1 = self.percentileChange(p99)
        if tr:
            return True, idx, (0, np.mean(mean), mean[idx])
        elif tr1:
            return True, idx1, (1, np.mean(p99), p99[idx1])
        return False, 0, ()

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
        tbounds,t0 = self.calcBounds(df, pos, self.tdelta//2)
        # print(df.index[pos])
        for ky in tbounds:
            print(ky, df.index[ky])
        # gpt_str = gpt_str_base
        # curr_mean, prev_mean = self.__meanCalc(df, tbounds)
        anomaly_map = {}

        for metric in df.columns:
            try:
                tr, idx, val = self.checkMetric(df, metric, tbounds)

            except Exception as e:
                print(self.checkMetric(df, metric, tbounds))
                # exit(1)
            if tr:
                to_monitor.append(metric)
                if idx not in anomaly_map:
                    anomaly_map[idx]={0:[],1:[]}
                anomaly_map[idx][val[0]].append([metric,val[2],val[1]])
            # break
        if len(to_monitor)>60:
            self.meanThreshold=1.25
            anomaly_map={}
            to_monitor_new=[]
            for i in to_monitor:
                try:
                    tr, idx, val = self.checkMetric(df, i, tbounds)
                except Exception as e:
                    print(self.checkMetric(df, i, tbounds))
                if tr:
                    to_monitor_new.append(i)
                    if idx not in anomaly_map:
                        anomaly_map[idx]={0:[],1:[]}
                    anomaly_map[idx][val[0]].append([i,val[2],val[1]])
            to_monitor=to_monitor_new

        print(to_monitor)
        print(len(to_monitor))
        gpt_str_base = f'''You are a mongodb diagnostic engine specialising in determining the root cause of anomalous metrics provided to you. The given mongodb server has seen a sheer drop in write Tickets at Timestamp {df.index[t0]}.During this time period, there were no significant changes in the server's hardware or software configuration. 

        TASK: Perform root cause analysis of the ticket drop. 

        Background: In this context, a "ticket drop" means that concurrentTransactions.write.out or concurrentTransactions.read.out  has increased and keeps pushing towards 128(its maximum configured limit). This can happen when operations take longer to release the ticket, or there are a large number of concurrent requests to the server.Any operation or command first acquires a ticket based on its requirement(read/write), performs the operation and releases the ticket. When the number of avaialble tickets drop, it can be either due to some operations taking longer time to release the ticket, or a large number of concurrent requests to the server. 
        
        Therefore, I want you to use your own understanding of mongodb and wiredtiger engine metrics to understand and analyse the provided data. Using the data, try to locate the root cause of the ticket drop. First give a summary of the analysis and then move to the explanation. Structure the output properly. Do not leave out any important information/relevant metric. 

Specific Information:

Values(not percentage change) of cache dirty ratio > 5% and/or cache fill ratio > 80%, which indicate that cache eviction has started by worker threads.
Similarly, values(not percentage change) of cache dirty ratio > 20% and/or cache fill ratio > 95%, which indicate that cache eviction has started by application threads.
'cursor.cached cursor count' indicates the number of cursors currently cached by the WiredTiger storage engine.
'history store score' is an indicator of the pressure the history store is placing on the cache.

However, do not limit the analysis to these pointers. As a mongodb diagnostic specialist you have the responsibility to analyse the meaning of each metric and the consequence of the change in the values.

Please note the abbreviations:

'sm' stands for system metrics
'ss' stands for server status
'wt' stands for wiredtiger.

Data Format & Interval:

Each timestamp marks the beginning of a 10-minute interval, leading up to just before the next timestamp. The anomalous metrics data provided for each interval corresponds to either the mean or the 99th percentile value within the interval, which is then contrasted against the average value of that metric over all recorded intervals. Each timestamp has a list of anomalous metrics followed by information on the anomaly.

DATA:

'''
        sorted_keys = sorted(anomaly_map.keys())
        anomalyObj={}
        for ts in sorted_keys:
            tsss=str(df.index[tbounds[ts]])
            if tsss not in anomalyObj:
                anomalyObj[tsss]=[]
            # gpt_str_base+=f"{tsss}:\n"
            for val in anomaly_map[ts][0]:
                anomalyObj[tsss].append({
                    "metric":val[0],
                    "anomaly interval mean": val[1],
                    "overall mean": val[2],
                    "change percentage": 100*(val[1]-val[2])/val[2]
                })
                # gpt_str_base+=f"{val[0]}: current interval mean:{val[1]}, overall mean: {val[2]}, change: {100*(val[1]-val[2])/val[2]}%\n"
            for val in anomaly_map[ts][1]:
                anomalyObj[tsss].append({
                    "metric":val[0],
                    "99th percentile of anomaly interval": val[1],
                    "mean of 99th percentile of all intervals": val[2]
                })
                # gpt_str_base+=f"{val[0]}: 99 percentile of current interval:{val[1]}, mean of percentile of all intervals: {val[2]}\n"
            gpt_str_base+=json.dumps(anomalyObj,default=str)
        with open("gpt-input.txt",'w') as gpt:
            gpt.write(gpt_str_base)
        vertical = (df.index[t0])
        self.__plot(df.iloc[tbounds[0]:tbounds[-1]],
                    to_monitor, main_title="metric-A", vert_x=vertical)

    def parseAll(self):
        def delta(metrList):
            mylst = [metrList[i] for i in range(len(metrList))]
            for i in range(1, len(metrList)):
                mylst[i] -= metrList[i-1]
            return mylst

        def checkCriteria(met):
            if met.startswith("serverStatus.metrics.aggStageCounters") or met.startswith("serverStatus.metrics.commands"):
                return True
            if met.startswith("systemMetrics.disks"):
                return True
            if met.startswith("replSetGetStatus.members") and (met.endswith("state") or met.endswith("health") or met.endswith("lag")):
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
