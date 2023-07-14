import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import openai
import os
import time
from FTDC_plot import FTDC_plot

class FTDC_an:
    def __init__(self, metricObj, qTstamp, outPDFpath, duration, exact):
        self.metricObj = metricObj
        self.queryTimeStamp = qTstamp
        self.ticketlim = 50
        self.tdelta = duration
        self.outPDF = outPDFpath
        self.nbuckets = 12
        self.anomalyBuckets = 3
        self.meanThreshold = 1.25
        self.totalTickets = 128
        self.exact = exact

    def __plot(self, df, to_monitor, vert_x, gpt_out=""):
        to_monitor.sort()
        print(len(to_monitor), "metrics")
        plot_ob = FTDC_plot(df, to_monitor, vert_x, gpt_out, self.outPDF)
        plot_ob.plot()
        return

    def __renameCols(self, df):
        rename_cols = {}
        for col_name in df.columns:
            ccol = col_name.replace("serverStatus.", "ss ")
            ccol = ccol.replace("wiredTiger.", "wt ")
            ccol = ccol.replace("systemMetrics.", "sm ")
            ccol = ccol.replace("tcmalloc.tcmalloc", "tcmalloc")
            ccol = ccol.replace("transaction.", "txn ")
            ccol = ccol.replace("local.oplog.rs.stats.", "locOplogRsStats ")
            ccol = ccol.replace("aggStageCounters.", "aggCnt ")
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
                mystr = key
                disk = mystr.split("systemMetrics.disks.")[1].split('.')[0]
                if disk not in disks:
                    disks.append(disk)

        for disk in disks:
            io = "systemMetrics.disks."+disk+".io_time_ms"
            newkey = "systemMetrics.disks."+disk+" utilization%"
            if io not in metricObj:
                continue
            metricObj[newkey] = []
            for idx in range(len(metricObj[io])):
                if metricObj[io][idx] == 0:
                    metricObj[newkey].append(0)
                else:
                    metricObj[newkey].append(
                        (metricObj[io][idx])/(10))

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
        typ = -1
        pos1 = max(0,pos-self.tdelta*2)
        pos2 = min(pos+self.tdelta*6,len(df))
        read_ticket = 'ss wt concurrentTransactions.read.available'
        write_ticket = 'ss wt concurrentTransactions.write.available'
        for idx in range(pos1, pos2):
            if df.iloc[idx][write_ticket] < self.ticketlim and df.iloc[idx][read_ticket] < self.ticketlim:
                t0 = idx
                typ = 0
                print("found both read and write ticket drop at: ",df.index[t0])
                break
            if df.iloc[idx][write_ticket] < self.ticketlim:
                t0 = idx
                typ = 1
                print("found write ticket drop at:", df.index[t0])
                break
            if df.iloc[idx][read_ticket] < self.ticketlim:
                t0 = idx
                typ = 2
                print("found read ticket drop at:", df.index[t0])
                break
        # print(t0)
        if typ == -1 or self.exact == 1:
            t0=pos
            print("Setting the ticket drop to:",df.index[pos],"as requested")
        idx = t0+delt
        # for i in range(0, 2): # one extra bucket ahead if available
        while (not df.index[idx] and idx < len(df)):
            idx += 1
        tbounds.append(idx)
        for i in range(0, self.nbuckets):
            if idx <=0:
                break
            idx -= 2*delt
            while (not df.index[idx] and idx > 0):
                idx -= 1
            tbounds.insert(0, idx)
        return tbounds, t0, typ

    def has_outliers(self, data):
        multiplier = self.meanThreshold
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
                # print(value,ctr)
            ctr += 1

        if curr_val != None and curr_ctr >= len(data)-self.anomalyBuckets:
            # print("Mean outlier: ",curr_val,curr_ctr)
            return True, curr_ctr
        return False, ctr

    def percentileChange(self, data, cont=0.2):
        # Ensure data is a numpy array and reshape it.
        data = np.array(data).reshape(-1, 1)

        # Initialize and fit the IsolationForest model.
        iso_forest = IsolationForest(contamination=cont)
        iso_forest.fit(data)
        pred = iso_forest.predict(data)
        outliers = data[pred == -1]

        # Check if there are no outliers.
        if len(outliers) == 0:
            return False, 0

        # Find the index of the last outlier.
        last_outlier_index = np.where(data == outliers[-1])[0][0]

        # Check if the last outlier is within the last three elements.
        if last_outlier_index >= len(data)-self.anomalyBuckets:
            return True, last_outlier_index

        return False, 0

    def calculate_statistics(self, data):
        return np.mean(data), np.percentile(data, 99)

    def is_mean_stable(self, mean_values, perc_values):
        if np.mean(mean_values) == 0 or np.mean(perc_values) == 0:
            return True
        if all(abs(m / np.mean(mean_values)) <= self.meanThreshold for m in mean_values) and all(abs(m / np.mean(perc_values)) <= self.meanThreshold for m in perc_values):
            return True
        return False

    def check_metric(self, df, met, timebounds, containment=0.2):
        """
        Analyzes a specific metric in a pandas DataFrame within a specific time window.

        Args:
            df (DataFrame): Input pandas DataFrame.
            met (str): The metric to be analyzed.
            timebounds (list): Time boundaries.

        Returns:
            tuple: Tuple containing a boolean indicating whether a condition has been met,
                a bucket or index, and a tuple of statistics.
        """

        p99 = []
        means = []
        maxes = []
        for t in range(0, len(timebounds) - 1):
            temp_values = df.iloc[timebounds[t]:timebounds[t + 1]][met].values
            mean, percentile_99 = self.calculate_statistics(temp_values)
            p99.append(percentile_99)
            means.append(mean)
            maxes.append(np.max(temp_values))
            
        special_metrics = {"ss wt cache fill ratio": 80,
                           "ss wt cache dirty fill ratio": 5}
        special_metrics_perc = {"ss wt concurrentTransactions.read.out": self.totalTickets-self.ticketlim, "ss wt concurrentTransactions.write.out": self.totalTickets-self.ticketlim}
        if self.is_mean_stable(means, p99) and not (met in special_metrics_perc) and (met not in special_metrics):
            return False, 0, ()
        if met in special_metrics:
            indices = [index for index, element in enumerate(p99) if (element >= special_metrics[met]*0.95 or maxes[index] >= special_metrics[met]) and element >= np.mean(p99)]
            if indices:
                _idx = max(indices)
                return True, _idx, (means[_idx], np.mean(means), p99[_idx], np.mean(p99))
            return False, 0, ()
        if met in special_metrics_perc: # if 99 percentile of out tickets is more than 78(50 tickets remaining)
            indices = [index for index, element in enumerate(
                p99) if element > special_metrics_perc[met]]
            if indices:
                return True, max(indices), (means[max(indices)], np.mean(means), p99[max(indices)], np.mean(p99))
            else:
                return False, 0, ()

        tr, idx = self.has_outliers(means)
        tr1, idx1 = self.percentileChange(p99, cont=containment)

        if tr and tr1:
            return True, idx, (means[idx], np.mean(means), p99[idx], np.mean(p99))
        elif tr:
            return True, idx, (means[idx], np.mean(means), p99[idx], np.mean(p99))
        elif tr1:
            return True, idx1, (means[idx1], np.mean(means), p99[idx1], np.mean(p99))
        return False, 0, ()

    def _init_analytics(self, metricObj):
        # self.__getAverageLatencies(metricObj)
        self.__tcmallocminuswt(metricObj)
        self.__getMemoryFragRatio(metricObj)
        self.__getDirtyFillRatio(metricObj)
        self.__getCacheFillRatio(metricObj)
        self.__diskUtilization(metricObj)

    def _prepare_dataframe(self, metricObj):
        df = pd.DataFrame(metricObj)
        df['serverStatus.start'] = df['serverStatus.start'].apply(
            self.getDTFromSecs)
        df['serverStatus.start'] = pd.to_datetime(df['serverStatus.start'])
        df.drop(index=0)
        df.set_index('serverStatus.start', inplace=True)
        df.columns.name = 'metrics'
        self.__renameCols(df)
        # print(df)
        return df

    def _calculate_anomalies(self, df, tbounds, to_monitor):
        def compare_strings(s): # sorting for getting prompt better
            if s.startswith("ss wt concurrentTransactions."):
                return (0, s)
            elif s.startswith("ss wt cache"):
                return (1, s)
            elif s.startswith("ss wt"):
                return (2, s)
            elif s.startswith("ss metrics"):
                return (3, s)
            elif s.startswith("ss opcounters"):
                return (5, s)
            else:
                return (6, s)
        anomaly_map = {}
        myList = df.columns.tolist()
        myList.sort(key=compare_strings)
        for metric in df.columns:
            try:
                tr, idx, val = self.check_metric(df, metric, tbounds)
                if tr and not (metric.startswith("sm disks") and metric.endswith("io_time_ms")):
                    to_monitor.append(metric)
                    anomaly_map = self._update_anomaly_map(
                        metric, idx, val, anomaly_map)
            except Exception as e:
                print(e)
                print("unable to insert metric:", metric)
        return anomaly_map, to_monitor

    def _update_anomaly_map(self, metric, idx, val, anomaly_map):
        if idx not in anomaly_map:
            anomaly_map[idx] = []
        anomaly_map[idx].append([metric, val[0], val[1], val[2], val[3]])
        return anomaly_map

    def _create_anomaly_obj(self, sorted_keys, anomaly_map, tbounds, df):
        anomalyObj = {}
        for ts in sorted_keys:
            tsss = str(df.index[tbounds[ts]])
            if tsss not in anomalyObj:
                anomalyObj[tsss] = []
            for val in anomaly_map[ts]:
                anomalyObj[tsss].append({
                    "metric": val[0],
                    "anomaly interval mean": val[1],
                    "overall mean": val[2],
                    "anomaly interval 99th percentile": val[3],
                    "overall mean 99th percentile": val[4]
                })
        return anomalyObj

    def _create_gpt_str_base(self, df, t0, typ):
        ticket_type = "write" if typ == 1 else "read"
        if typ == 0:
            ticket_type = "both read and write"
        gpt_str_base = f'''You are a mongodb diagnostic engine specialising in determining the root cause of anomalous metrics provided to you. The given mongodb server has seen a drop in available {ticket_type} tickets at Timestamp {df.index[t0]}.During this time period, there were no significant changes in the server's hardware or software configuration. A "ticket drop" in this scenario signifies a rise in either concurrentTransactions.write.out or concurrentTransactions.read.out, typically due to lengthy operations not releasing the ticket promptly or an influx of concurrent server requests. Each operation acquires a ticket and releases it after the task is done.

TASK: Your task, as a MongoDB diagnostic specialist, is to analyze the given data with reference to MongoDB and WiredTiger engine metrics to determine the ticket drop's root cause. Please analyze each and every metric listed in the list provided.

Important thresholds and information include:
1. Analyze ss metrics commands, operation, queryExecutor, etc. and opCounters (updates, deletes etc.). Any surge in opCounters or any ss metrics(commands, operation, queryExecutor) is indicative of increase in workload,a potential reason for a ticket drop which *must* be included in analysis. 
2. Examine cache dirty/fill ratios. When cache dirty ratio surpasses 5%, eviction is initiated by worker threads and on crossing 20%, by application threads. A cache fill ratio over 80% initiates worker thread eviction and above 95% starts application thread eviction.
3. Reviewing eviction statistics due to their impact on worker threads and cache. Remember that evicting a modified page demands more resources.
4. Check 'cursor.cached cursor count', a measure of currently cached cursors by the WiredTiger engine.
5. Monitor logicalSessionRecordCache, used by MongoDB to track client sessions status.
6. Review disk utilization values. High values can indicate disk bottleneck. Anything below 50% can be safely ignored.

These pointers should supplement your analysis, not limit it. As a specialist, interpret each metric and its implications.

Note: Always examine percentile values for cache dirty and fill ratios, and be alert for any anomalies, especially in opCounters and ss metrics (commands, operation, queryExecutor). Since we are dealing with intervals, a looking at both mean and 99th percentile would give you a better insight.

Abbreviations to note:
'sm' - system metrics, 'ss' - server status, 'wt' - wiredtiger.

Data Format:
Each timestamp denotes the interval from itself to {self.tdelta//60} minutes ahead of it. For example, anomaly interval mean at timestamp t, means the mean of the given metric in [t,t+{self.tdelta//60} minutes]. 

The data contains timestamps and a list of anomalous metrics that were anomalous in the interval denoted by the timestamp. The meaning of each heading is as follows:
`anomaly interval mean`: mean of the metric in the timestamp interval where it was anomalous 
`overall mean`: mean of the metric over the monitored duration
`anomaly interval 99th percentile`: 99th percentile value of the metric in the timestamp interval where it was anomalous 
`overall mean 99th percentile`: mean of 99th percentile value of all intervals in the monitored duration 

Output Format: Provide a well-structured and comprehensive summary first and then a deeper detailed explanation of your analysis. Make sure no crucial details or metrics are overlooked. Every place you mention a timestamp, use "In the interval between <Timestamp> and <Timestamp+{self.tdelta//60}> ...."

NOTE: The focus is on in-depth analysis, so please refer to definitions and detailed implications of each metric as needed from your model.
'''
        return gpt_str_base

    def _save_gpt_str_base(self, gpt_str_base):
        with open("gpt-input.txt", 'w') as gpt:
            gpt.write(gpt_str_base)

    def _openAI_req(self, message):
        KEY_NAME = 'OPENAI_API_KEY'
        req = {"model": "gpt-4",
               "messages": [{"role": "user", "content": message}]}
        # return ""
        key = os.environ.get(KEY_NAME)
        if key is not None:
            openai.api_key = key
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "user", "content": message}
                    ]
                )
                outp = completion.choices[0].message.content
                print(outp)
            except Exception as e:
                print(e)
                print("Generating report without summary as openAI failed to respond.")
                outp = ""
            return outp
        else:
            print("No openAI API key found in the environment variable")
            return ""

    def analytics(self, metricObj, queryTimestamp):
        self._init_analytics(metricObj)
        df = self._prepare_dataframe(metricObj)
        df.to_csv('test.csv')
        # print(df)
        pos = np.where(df.index == queryTimestamp)[0][0]
        tbounds, t0, typ = self.calcBounds(df, pos, self.tdelta//2)
        if typ == -1:
            pos1 = df.index[max(0,pos-self.tdelta*2)]
            pos2 = df.index[min(pos+self.tdelta*6,len(df))]
            print(
                f"No ticket drop found in the interval {pos1} and {pos2}. Please try with another timestamp or a higher interval size. Currently generating graphs corresponding to query")
        to_monitor = []
        # typ=-1 # uncomment if want to print all graphs regardless of ticket drop
        if typ != -1:
            anomaly_map, to_monitor = self._calculate_anomalies(
                df, tbounds, to_monitor)
            gpt_str_base = self._create_gpt_str_base(df, t0, typ)
            anomalyObj = self._create_anomaly_obj(
                sorted(anomaly_map.keys(), reverse=True), anomaly_map, tbounds, df)
            gpt_str = ''''''
            headers = ["metric", "anomaly interval mean", "overall mean",
                    "anomaly interval 99th percentile", "overall mean 99th percentile"]
            for idx, head in enumerate(headers):
                if idx == len(headers)-1:
                    gpt_str += head+"\n"
                else:
                    gpt_str += head+","
            for timestamp, objects in anomalyObj.items():
                gpt_str += str(timestamp)+":\n"
                for obj in objects:
                    tmpstr = ""
                    for idx, head in enumerate(headers):
                        if idx == len(headers)-1:
                            tmpstr += (str(obj[head])+"\n")
                        else:
                            tmpstr += (str(obj[head])+",")
                    gpt_str += tmpstr
            gpt_str_base += gpt_str
            # self._save_gpt_str_base(gpt_str_base)
            st=time.time()
            gpt_res = self._openAI_req(gpt_str_base)
            st=time.time()-st
        else:
            gpt_res = ""
            to_monitor = df.columns.tolist()
        vertical = (df.index[t0])
        tickets = ['ss wt concurrentTransactions.write.out',
                   'ss wt concurrentTransactions.read.out']
        for tick in tickets:
            if tick not in to_monitor:
                to_monitor.append(tick)
        st = time.time()
        self.__plot(df.iloc[tbounds[0]:tbounds[-1]],
                    to_monitor, vert_x=vertical, gpt_out=gpt_res)
        st = time.time()-st
        # print("Time taken to render:",st)

    def parseAll(self):
        def delta(metrList, prevVal=0):
            mylst = [metrList[i] for i in range(len(metrList))]
            for i in range(1, len(metrList)):
                mylst[i] -= metrList[i-1]
            mylst[0] -= prevVal
            return mylst

        def checkCriteria(met):
            if met.startswith("systemMetrics.disks") and met.endswith("io_time_ms"):
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
        prevVal = {}
        prevUptime = metObj["serverStatus.uptime"][-1]

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
                prevVal[met] = metObj[met][-1]
                data[met] = delta(metObj[met])

        for key in iter_keys:
            metObj = self.metricObj[key]
            try:
                if "serverStatus.uptime" not in metObj or prevUptime > metObj["serverStatus.uptime"][0]:
                    for key in prevVal:
                        prevVal[key]=0
            except:
                print(key)
                exit(0)
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
                # print("occurence of new accumulate metric", met)

            for met in new_p:
                data[met] = [0 for i in range(deltactr)]
                # print("occurence of new point metric", met)

            for met in sel_metr_p_new:
                # now fill all the values obtained
                data[met].extend(metObj[met])
            for met in sel_metr_c_new:
                if met in prevVal:
                    previous = prevVal[met]
                else:
                    previous = 0
                prevVal[met] = metObj[met][-1]
                data[met].extend(delta(metObj[met], previous))
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
            if "serverStatus.uptime" in metObj:
                prevUptime = metObj["serverStatus.uptime"][-1]
            else:
                prevUptime = 0
        self.analytics(data, self.queryTimeStamp)
