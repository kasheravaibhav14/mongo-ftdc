import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from math import ceil
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot as plt
from pprint import pprint
import sesd
from PyPDF4 import PdfFileReader, PdfFileWriter
from FTDC_plot import FTDC_plot
import multiprocessing as mp
def checkMetric(df, met):
    ts = df[met]
    anom_count=30
    if  ts.max()>=1.25*ts.mean() and ts.mean()!=0:
        try:
            outliers_indices = sesd.seasonal_esd(
                ts, alpha=0.05, max_anomalies=anom_count, hybrid=True)
            if len(outliers_indices) > 0:
                return True, len(outliers_indices)
        except Exception as e:
            print(met,"couldn't determine for outliers")
    meanval=df[met].mean()
    maxval=df[met].max()
    if "ss wt cache dirty fill ratio" == met:
        print(met,maxval)
        if maxval >= 5:
            return True, 1
    if "ss wt cache fill ratio" == met:
        print(met,maxval)
        if maxval >= 75:
            return True, 1
    if "ss wt txn transaction checkpoint currently running" == met and meanval>0.20:
        return True, 1
    # if "ss wt concurrentTransactions.write.available" == met or "ss wt concurrentTransactions.read.available" == met:
    #     return True,1
    return False, 0
def checkMetric_wrapper(args):
    df, met = args
    tr, val = checkMetric(df, met)
    if tr:
        print(met, "done")
        return val, met
    else:
        return val, None
class FTDC_an:
    def __init__(self, metricObj, qTstamp, outPDFpath,duration):
        self.metricObj = metricObj
        self.queryTimeStamp = qTstamp
        self.ticketlim = 15  # when available tickets are more than 15, we assume that the system is in normal state or is heading toward crash
        self.tdelta = duration
        self.threshold = 0.25
        self.outPDF = outPDFpath

    def __plot(self, df, to_monitor, vert_x, main_title="metric-A", outfilename="fig.pdf"):
        to_monitor.sort()
        df1=df[to_monitor]
        end_index = self.outPDF.rindex("/")
        outlierPath = self.outPDF[:end_index]+'/outliers.csv'
        df1.to_csv(outlierPath,index=False)
        # print("Monitoring: ", str(to_monitor))
        print(len(to_monitor))
        plot_ob = FTDC_plot(df,to_monitor,self.outPDF)
        plot_ob.plot()
        return
        n_cols = 1
        n_rows = int(ceil(len(to_monitor)/n_cols))
        fig = make_subplots(n_rows, n_cols, subplot_titles=to_monitor)
        for i in range(len(to_monitor)):
            fig.add_trace(go.Scatter(
                x=df.index, y=df[to_monitor[i]]), 1+i//n_cols, 1+i % n_cols)
            max_val = df[to_monitor[i]].max()
            min_val = df[to_monitor[i]].min()
            mean_val = df[to_monitor[i]].mean()
            # print(max_val,min_val,mean_val)
            if type(vert_x) == list:
                for vx in vert_x:
                    fig.add_shape(type='line', x0=vx, y0=min_val, x1=vx, y1=1.15*max_val,
                                  line=dict(width=1, dash='dot'), row=1+i//n_cols, col=1+i % n_cols)
            else:
                fig.add_shape(type='line', x0=vert_x, y0=min_val, x1=vert_x, y1=1.15*max_val,
                              line=dict(width=1, dash='dot'), row=1+i//n_cols, col=1+i % n_cols)
            fig.add_annotation(x=df.index[0], y=max_val, text=f'Max: {max_val:.2f}<br>Mean: {mean_val:.2f}<br>Min:{min_val:.2f}',
                               showarrow=False,
                               font=dict(color='black', size=8),
                               row=1 + i // n_cols, col=1 + i % n_cols)
        fig.update_layout(title={
            'text':main_title,
            # 'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=10,
                color="black",
                family="Courier New, monospace",
            )
        }, showlegend=False, margin=dict(
            l=100,
            r=20,  # Increase right margin to fit text
            b=20,
            t=100,
            pad=5))
        fig.write_image(outfilename, 'pdf', width=n_cols *
                        800, height=n_rows*125, scale=1)

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

    def checkMetricHourly(self, curr_mean, prev_mean, met):
        if prev_mean[met] != 0 and (abs(curr_mean[met]-prev_mean[met])/prev_mean[met]) > self.threshold:
            return True
        # if met == "ss wt cache.bytes dirty in the cache cumulative" or (met.startswith("ss wt concurrentTransactions") and not met.endswith("totalTickets")):
        #     return True
        # if met.startswith("ss wt txn"):
        #     return True
        return False

    

    def calcBounds(self, df, pos, delt):
        tbounds = {'t0': -1, 'c_ub': -1, 'c_lb': -
                   1, 'p_lb': -1}  # p_ub is equal to c_lb
        tdelta = timedelta(seconds=delt)
        for idx in range(pos, -1, -1):
            if df.iloc[idx]['ss wt concurrentTransactions.write.available'] > self.ticketlim \
                    and df.iloc[idx]['ss wt concurrentTransactions.read.available'] > self.ticketlim:
                tbounds['t0'] = idx
                break
        t0_val = df.index[tbounds['t0']]
        for idx in range(tbounds['t0'], len(df), 1):
            if df.index[idx]-t0_val >= tdelta:
                tbounds['c_ub'] = idx
                break
        for idx in range(tbounds["t0"], -1, -1):
            if t0_val-df.index[idx] >= tdelta and tbounds['c_lb'] == -1:
                tbounds['c_lb'] = idx
            if t0_val-df.index[idx] >= 3*tdelta and tbounds['p_lb'] == -1:
                tbounds['p_lb'] = idx
                break
        return tbounds

    def hourlyAnalytics(self, df):
        to_monitor = []
        start = df.index[0]
        end = min(df.index[-1], start+timedelta(hours=6))
        while (start <= end):
            nxt = start+timedelta(minutes=60)
            for met in df.columns:
                if met not in to_monitor:
                    tr, v = checkMetric(df.loc[start:nxt], met)
                    if tr:
                        to_monitor.append(met)
            start = nxt
        vert_x = []
        start = df.index[0]
        while start < end:
            start = start+timedelta(hours=1)
            vert_x.append(start)
        print(vert_x)
        self.__plot(df, to_monitor, vert_x, "hourly data", "fig_hourly.pdf")

    def analytics(self, metricObj, queryTimestamp):
        self.__getAverageLatencies(metricObj)
        self.__tcmallocminuswt(metricObj)
        self.__getMemoryFragRatio(metricObj)
        self.__getDirtyFillRatio(metricObj)
        self.__getCacheFillRatio(metricObj)
        self.__diskUtilization(metricObj)
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
        gpt_str_base = '''
        Following are the anomalous metrics obtained for a server running mongodb when around the time it started facing ticket drops. Each line has metric and its percentage change separated by space. What do you infer from these metrics, instead of looking individually at each metric, select those metrics which you think are more impactful than others.
        sm stands for system metrics
        ss stands for server status
        wt stands for wiredtiger
        If the majority of these changes are negative, consider that now conditions are normal and was severe before, use it to predict what could be a root cause to these conditions in a mongo server. Analyse each metric, but report metrics of the highest impact and those that could be interrelated with each other as well. Perform any other analysis as you deem fit. Try to conclude to a list of specific possible reasons causing the issue.
        '''
        # return
        tbounds = self.calcBounds(df, pos, self.tdelta)
        print(df.index[pos])
        for ky in tbounds:
            print(ky, df.index[tbounds[ky]])
        # gpt_str = gpt_str_base
        # curr_mean, prev_mean = self.__meanCalc(df, tbounds)
        pool = mp.Pool(6)  # create a multiprocessing Pool
        # prepare the arguments for the worker function
        args = [(df.iloc[tbounds['p_lb']:tbounds['c_ub']+1], metric) for metric in df.columns]
        results = pool.map(checkMetric_wrapper, args)  # process the list using pool of workers
        # get the metrics for which checkMetric returned True
        to_monitor = [result[1] for result in results if result[1] is not None]
        pool.close()
        print(to_monitor)
        results.sort(reverse=True)
        for x in results:
            if x[1] is not None:
                print(x)

        # for metric in df.columns:
            
        #     try:
        #         tr, val = checkMetric(
        #             df.iloc[tbounds['p_lb']:tbounds['c_ub']+1], metric)
        
        self.__plot(df[tbounds['p_lb']:tbounds['c_ub']+1],
                             to_monitor,main_title="metric-A", vert_x=queryTimestamp)

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
