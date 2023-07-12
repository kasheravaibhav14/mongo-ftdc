import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import openai
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import markdown
from bs4 import BeautifulSoup


class FTDC_plot:
    def __init__(self, df, to_monitor, vert_x, gpt_out="", outfilepath="report.pdf"):
        self.df = df
        self.to_monitor = to_monitor
        self.outPath = outfilepath
        self.vert_x = vert_x
        self.subheadings = ["ss wt", "ss", "ss metrics", "sm", "ss tcmalloc",]
        self.expandedSub = {"ss wt": "wiredTiger", "ss": "Server status",
                            "ss metrics": "Query metrics", "sm": "System metrics", "ss tcmalloc": "tcmalloc"}
        self.markdown_string = gpt_out

    def custom_sort(self, arr):
        def compare_strings(s):
            if s.startswith("ss wt concurrentTransactions."):
                return (0, s)
            elif s.startswith("ss wt cache"):
                return (1, s)
            elif s.startswith("ss wt"):
                return (2, s)
            elif s.startswith("ss metrics"):
                return (3, s)
            elif s.startswith("ss") and not s.startswith("ss locks"):
                return (4, s)
            elif s.startswith("ss locks") and "acquireCount" in s:
                return (5, s)
            elif s.startswith("sm"):
                return (10, s)
            else:
                return (7, s)
        arr.sort(key=compare_strings)

    def dumpGPT(self, canvas, y=A3[1]):
        """
        Convert a markdown string to a PDF file.

        :param markdown_string: The markdown content.
        :param canvas: The reportlab.pdfgen.canvas.Canvas instance.
        """
        STYLES = getSampleStyleSheet()
        STYLE_P = ParagraphStyle(
            'paragraphItem', parent=STYLES['Normal'], fontSize=12, leading=15, firstLineIndent=10)

        html = markdown.markdown(self.markdown_string)
        soup = BeautifulSoup(html, 'html.parser')

        width, height = A3
        width -= 50
        height -= 50

        for elem in soup:
            if y < 50:  # Arbitrarily chosen value; you may need to adjust.
                canvas.showPage()
                y = height
            # print(elem.name)
            if elem.name == 'h1':
                para = Paragraph(elem.text, STYLES['Heading1'])
                w, h = para.wrap(width, height)
                y -= h
                para.drawOn(canvas, 20, y)
                y -= 15
            elif elem.name == 'h2':
                para = Paragraph(elem.text, STYLES['Heading2'])
                w, h = para.wrap(width, height)
                y -= h
                para.drawOn(canvas, 20, y)
                y -= 15
            elif elem.name == 'p':
                para = Paragraph(elem.text, STYLE_P)
                w, h = para.wrap(width, height)
                y -= h
                para.drawOn(canvas, 20, y)
                y -= 15
            elif elem.name in ['ol', 'ul']:
                for it in elem:
                    if it.name == 'li':
                        para = Paragraph(it.text, STYLE_P)
                        w, h = para.wrap(width, height)
                        y -= h
                        para.drawOn(canvas, 20, y)
                        y -= 10
        canvas.showPage()

    def plot(self):
        self.custom_sort(self.to_monitor)
        seen_subheadings = set()
        locks_acq = []
        color_names = ['red', 'orange', 'green', 'cyan', 'gray', 'magenta']

        # Prepare the canvas
        c = canvas.Canvas(self.outPath, pagesize=A3)
        page_width, page_height = A3
        plt.style.use('ggplot')
        # Define column widths
        total_parts = 2 + 2 + 7 + 3
        column_widths = [page_width / total_parts * i for i in [2, 2, 7, 3]]

        # Define image dimensions
        image_width = column_widths[2]
        image_height = page_height / 35  # Modified image height

        # Variables to track the current y position and space needed for each plot+text pair
        header_height = 30  # Define header height
        padding = 10  # Define padding
        current_y = page_height - image_height - header_height - padding
        space_per_pair = image_height

        # Get sample style for paragraphs
        styles = getSampleStyleSheet()
        styleN = styles['Normal']
        styleN.alignment = 1
        # Function to draw column headers

        def draw_headers():
            headers = ["Mean", "Max", "Plot", "Metric"]
            for idx, header in enumerate(headers):
                P = Paragraph(f"<font size=12><b>{header}</b></font>", styleN)
                P.wrapOn(c, column_widths[idx], 500)
                P.drawOn(c, sum(column_widths[:idx]),
                         page_height - header_height)

        def check_locks(met):
            if met.startswith("ss locks") and "acquireCount" in met:
                return True
            return False

        def save_to_pdf(current_y, text_before1, text_before2, text_after, height_factor = 1):
            drawing = svg2rlg(f"plot_temp_{i+1}.svg")
            scaleX = image_width / drawing.width
            scaleY = height_factor*image_height / drawing.height
            drawing.width = image_width
            drawing.height = height_factor*image_height
            drawing.scale(scaleX, scaleY)
            renderPDF.draw(drawing, c, sum(column_widths[:2]), current_y)

            P = Paragraph(text_before1, styleN)
            P.wrapOn(c, column_widths[0], 500)
            P.drawOn(c, sum(column_widths[:1]) -
                     column_widths[0], current_y+padding)

            P = Paragraph(text_before2, styleN)
            P.wrapOn(c, column_widths[1], 500)
            P.drawOn(c, sum(column_widths[:2]) -
                     column_widths[1], current_y+padding)

            P = Paragraph(text_after, styleN)
            P.wrapOn(c, column_widths[3], 500)
            P.drawOn(c, sum(column_widths[:3]), current_y+padding)

            # Remove the SVG file
            os.remove(f"plot_temp_{i+1}.svg")


        draw_headers()
        subheadings_sorted = sorted(self.subheadings, key=len, reverse=True)
        for i in range(len(self.to_monitor)):
            if check_locks(self.to_monitor[i]):
                locks_acq.append(self.to_monitor[i])
                if i < (len(self.to_monitor)-1) and check_locks(self.to_monitor[i+1]):
                    continue
                else:
                    lock_dic = {"r": [], "R": [], "w": [], "W": []}
                    for lk in locks_acq:
                        lock_dic[lk[-1]].append(lk)
                    for ltype in lock_dic:
                        if len(lock_dic[ltype]) != 0:
                            current_y -= space_per_pair
                            if current_y - 2*space_per_pair < 0:
                                c.showPage()
                                current_y = page_height - image_height - header_height - padding
                                draw_headers()
                            minval = 1e9
                            meanval = 0
                            maxval = -1
                            x = self.df.index
                            fig, ax = plt.subplots(
                                figsize=(image_width / 100, 2*image_height / 100))
                            text_after = f"<font size = 10>ss locks acquireCount {ltype}: </font>"
                            for idx, col in enumerate(lock_dic[ltype]):
                                y = self.df[col].tolist()
                                # print(col)
                                maxval = max(np.max(y), maxval)
                                minval = min(np.min(y), minval)
                                meanval+=np.mean(y)
                                ax.plot(x, y, linewidth=0.15,
                                        color=color_names[idx])
                                text_after += f"<font size=10 color={color_names[idx]}>{col.split('.')[1]}, </font>"
                            ax.axvline(x=self.vert_x, color='b',
                                       linestyle='--', linewidth=0.25)
                            if current_y - 2*space_per_pair < 0 or i == len(self.to_monitor)-1:
                                # print the y label only when it is last plot of page
                                ax.set_xlabel('Timestamp', fontsize=8)
                            plt.yticks(fontsize=5, fontname='Helvetica')
                            plt.xticks(fontsize=5)
                            fmt = mtick.StrMethodFormatter('{x:0.2e}')
                            ax.yaxis.set_major_formatter(fmt)
                            inc = (maxval-minval)*0.1
                            ax.set_ylim(minval-inc, maxval+inc)
                            meanval/=len(lock_dic[ltype])

                            # Add a horizontal line at y=0.5
                            ax.axhline(y=meanval, color='black',
                                    linestyle='--', linewidth=0.2)

                            # Save the plot as an SVG image file
                            plt.savefig(f"plot_temp_{i+1}.svg", format="svg")
                            plt.close()
                            # Add the corresponding text to the PDF
                            text_before1 = f"<font size=8>{meanval:,.3f}</font>"
                            text_before2 = f"<font size=8>{maxval:,.3f}</font>"
                            # Add the image to the PDF
                            save_to_pdf(current_y, text_before1,
                                        text_before2, text_after, height_factor=2)

                            # Update the current y position
                            current_y -= space_per_pair
                              # Draw headers at the start of each new page
                    continue

            # If we don't have enough space for the next pair, create a new page
            if current_y - space_per_pair < 0:
                c.showPage()
                current_y = page_height - image_height - header_height - padding
                draw_headers()  # Draw headers at the start of each new page

            for subheading in subheadings_sorted:
                if self.to_monitor[i].startswith(subheading):
                    if subheading not in seen_subheadings:
                        # Remember that we've seen this subheading
                        seen_subheadings.add(subheading)
                        # Add the subheading to the PDF
                        text_subheading = f"<font size=10><i>{self.expandedSub[subheading]}</i></font>"
                        P = Paragraph(text_subheading)
                        P.wrapOn(c, page_width, 500)
                        current_y -= 1 * space_per_pair  # Leave some space after the subheading
                        # Adjust the position as needed
                        P.drawOn(c, 25, current_y + 1.5 * space_per_pair)
                    break  # Stop checking other subheadings

            x = self.df.index
            y = self.df[self.to_monitor[i]]
            minval = self.df[self.to_monitor[i]].min()
            meanval = self.df[self.to_monitor[i]].mean()
            maxval = self.df[self.to_monitor[i]].max()

            # Create a plot
            fig, ax = plt.subplots(
                figsize=(image_width / 100, image_height / 100))
            ax.plot(x, y, linewidth=0.3)

            ax.axvline(x=self.vert_x, color='b',
                       linestyle='--', linewidth=0.25)
            if current_y - 2*space_per_pair < 0 or i == len(self.to_monitor)-1:
                # print the y label only when it is last plot of page
                ax.set_xlabel('Timestamp', fontsize=8)
            plt.yticks(fontsize=5, fontname='Helvetica')
            plt.xticks(fontsize=5)
            fmt = mtick.StrMethodFormatter('{x:0.2e}')
            ax.yaxis.set_major_formatter(fmt)
            inc = (maxval-minval)*0.1
            ax.set_ylim(minval-inc, maxval+inc)

            # Add a horizontal line at y=0.5
            ax.axhline(y=meanval, color='black',
                       linestyle='--', linewidth=0.25)

            # Save the plot as an SVG image file
            plt.savefig(f"plot_temp_{i+1}.svg", format="svg")
            plt.close()

            # Add the corresponding text to the PDF
            text_before1 = f"<font size=8>{meanval:,.3f}</font>"
            text_before2 = f"<font size=8>{maxval:,.3f}</font>"
            text_after = f"<font size=10>{self.to_monitor[i]}</font>"

            save_to_pdf(current_y, text_before1, text_before2, text_after)
            # Update the current y position
            current_y -= space_per_pair

        # Save the PDF
        # c.showPage()
        self.dumpGPT(c, current_y)
        c.save()


class FTDC_an:
    def __init__(self, metricObj, qTstamp, outPDFpath, duration):
        self.metricObj = metricObj
        self.queryTimeStamp = qTstamp
        self.ticketlim = 50
        self.tdelta = duration
        self.outPDF = outPDFpath
        self.nbuckets = 12
        self.meanThreshold = 1.25
        self.totalTickets = 128

    def __plot(self, df, to_monitor, vert_x, gpt_out=""):
        to_monitor.sort()
        # end_index = self.outPDF.rindex("/")
        # outlierPath = self.outPDF[:end_index]+'/outliers.csv'
        # df.to_csv(outlierPath, index=True)
        # print("Monitoring: ", str(to_monitor))
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
        pos1 = pos-self.tdelta*2
        typ = -1
        for idx in range(pos1, pos1+6*self.tdelta):
            
            if df.iloc[idx]['ss wt concurrentTransactions.write.available'] < self.ticketlim and df.iloc[idx]['ss wt concurrentTransactions.read.available'] < self.ticketlim:
                t0 = idx
                typ = 0
                print("found both read and write ticket drop at: ",df.index[t0])
                break
            if df.iloc[idx]['ss wt concurrentTransactions.write.available'] < self.ticketlim:
                t0 = idx
                typ = 1
                print("found write ticket drop at:", df.index[t0])
                break
            if df.iloc[idx]['ss wt concurrentTransactions.read.available'] < self.ticketlim:
                t0 = idx
                typ = 2
                print("found read ticket drop at:", df.index[t0])
                break
        # print(t0)
        idx = t0+delt
        # for i in range(0, 2): # one extra bucket ahead if available
        while (not df.index[idx] and idx < len(df)):
            idx += 1
        tbounds.append(idx)
        # idx += 2*delt
        # idx = t0+delt
        for i in range(0, self.nbuckets):
            idx -= 2*delt
            while (not df.index[idx] and idx > 0):
                idx -= 1
            tbounds.insert(0, idx)
        # print(pos)
        # print(tbounds)
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

        if curr_val != None and curr_ctr >= len(data)-3:
            # print("Mean outlier: ",curr_val,curr_ctr)
            return True, curr_ctr
        return False, ctr

    def percentileChange(self, data, cont=0.2):
        # Ensure data is a numpy array and reshape it.
        data = np.array(data).reshape(-1, 1)

        # Initialize and fit the IsolationForest model.
        iso_forest = IsolationForest(contamination=cont)
        iso_forest.fit(data)

        # Predict outliers in the data.
        pred = iso_forest.predict(data)

        # Identify and print the outliers.
        outliers = data[pred == -1]
        # print('Outliers:', outliers)

        # Check if there are no outliers.
        if len(outliers) == 0:
            return False, 0

        # Find the index of the last outlier.
        last_outlier_index = np.where(data == outliers[-1])[0][0]

        # Check if the last outlier is within the last three elements.
        if last_outlier_index >= len(data)-3:
            # print(f"Outlier within last 3 elements: {outliers[-1]}, Index: {last_outlier_index}")
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
        # print(met)
        df_slice = df.iloc[timebounds[0]:timebounds[-1]]
        maxval = df_slice[met].max()
        maxpos = np.argmax(df_slice[met].values)
        bucket = maxpos // self.tdelta

        p99 = []
        means = []
        maxes = []
        for t in range(0, len(timebounds) - 1):
            temp_values = df.iloc[timebounds[t]:timebounds[t + 1]][met].values
            mean, percentile_99 = self.calculate_statistics(temp_values)
            p99.append(percentile_99)
            means.append(mean)
            maxes.append(np.max(temp_values))
        # print(means)
        # print(p99)
        special_metrics = {"ss wt cache fill ratio": 80,
                           "ss wt cache dirty fill ratio": 5}
        special_metrics_perc = {"ss wt concurrentTransactions.read.out": self.totalTickets -
                                self.ticketlim, "ss wt concurrentTransactions.write.out": self.totalTickets-self.ticketlim}
        if self.is_mean_stable(means, p99) and not (met in special_metrics_perc) and (met not in special_metrics):
            return False, 0, ()
        if met in special_metrics:
            indices = [index for index, element in enumerate(p99) if (
                element >= special_metrics[met]*0.95 or maxes[index] >= special_metrics[met]) and element >= np.mean(p99)]
            if indices:
                _idx = max(indices)
                return True, _idx, (means[_idx], np.mean(means), p99[_idx], np.mean(p99))
            return False, 0, ()
        if met in special_metrics_perc:
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
        df.set_index('serverStatus.start', inplace=True)
        df.columns.name = 'metrics'
        self.__renameCols(df)
        # print(df)
        return df

    def _calculate_anomalies(self, df, tbounds, to_monitor, mean_threshold=1.5):
        def compare_strings(s): # sorting for AI inference
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
            except Exception as e:
                print("unable to insert metric:", metric)
            if tr and not (metric.startswith("sm disks") and metric.endswith("io_time_ms")):
                to_monitor.append(metric)
                anomaly_map = self._update_anomaly_map(
                    metric, idx, val, anomaly_map)
        if len(to_monitor) > 60:
            anomaly_map, to_monitor = self._recalculate_anomalies(
                df, tbounds, to_monitor)
        return anomaly_map, to_monitor

    def _update_anomaly_map(self, metric, idx, val, anomaly_map):
        if idx not in anomaly_map:
            anomaly_map[idx] = []
        anomaly_map[idx].append([metric, val[0], val[1], val[2], val[3]])
        return anomaly_map

    def _recalculate_anomalies(self, df, tbounds, to_monitor):
        anomaly_map = {}
        to_monitor_new = []
        for i in to_monitor:
            try:
                tr, idx, val = self.check_metric(
                    df, i, tbounds, containment=0.08)
            except Exception as e:
                print("unable to insert metric: ", i)
            if tr:
                to_monitor_new.append(i)
                anomaly_map = self._update_anomaly_map(
                    i, idx, val, anomaly_map)
        to_monitor = to_monitor_new
        return anomaly_map, to_monitor

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
                    # "change percentage mean": 100*(val[1]-val[2])/val[2],
                    "anomaly interval 99th percentile": val[3],
                    "overall mean 99th percentile": val[4]
                })
        return anomalyObj

    def _create_gpt_str_base(self, df, t0, typ):
        ticket_type = "write" if typ == 1 else "read"
        if typ == 0:
            ticket_type = "both read and write"
        gpt_str_base = f'''You are a mongodb diagnostic engine specialising in determining the root cause of anomalous metrics provided to you. The given mongodb server has seen a drop in available {ticket_type} tickets at Timestamp {df.index[t0]}.During this time period, there were no significant changes in the server's hardware or software configuration. A "ticket drop" in this scenario signifies a continuous rise towards the 128(max) of either concurrentTransactions.write.out or concurrentTransactions.read.out, typically due to lengthy operations not releasing the ticket promptly or an influx of concurrent server requests. Each operation acquires a ticket and releases it after the task is done.

TASK: Your task, as a MongoDB diagnostic specialist, is to analyze the given data with reference to MongoDB and WiredTiger engine metrics to determine the ticket drop's root cause. Please analyze each and every metric listed in the table provided. 


Important thresholds and information include:
1. Analyze ss metrics commands, operation, queryExecutor, etc. and opCounters (updates, deletes etc.). Any surge in opCounters or any metrics(commands, operation, queryExecutor) is indicative of increase in workload, which can be a potential reason for a ticket drop and must be included in analysis. 
2. Examine cache dirty/fill ratios. When cache dirty ratio surpasses 5%, eviction is initiated by worker threads and on crossing 20%, by application threads. A cache fill ratio over 80% initiates worker thread eviction and above 95% starts application thread eviction.
3. Prioritize reviewing eviction statistics due to their impact on worker threads and cache. Remember that evicting a modified page demands more resources.
4. Check 'cursor.cached cursor count', a measure of currently cached cursors by the WiredTiger engine.
5. Note 'history store score', which indicates cache pressure from the history store.
6. Monitor logicalSessionRecordCache, used by MongoDB to track client sessions status.
7. Review disk utilization values. High values can indicate disk bottleneck. Anything below 50% can be safely ignored.

Note: Always examine percentile values for cache dirty and fill ratios, and be alert for any anomalies, especially in opCounters and metrics commands. Since we are dealing with intervals, a looking at both mean and 99th percentile could give you a better insight.

These pointers should supplement your analysis, not limit it. As a specialist, interpret each metric and its implications.
Abbreviations to note:
'sm' - system metrics, 'ss' - server status, 'wt' - wiredtiger.

Data Format:
Each timestamp denotes the interval from itself to {self.tdelta//60} minutes ahead of it. For example, anomaly interval mean at timestamp t, means the mean of the given metric in [t,t+{self.tdelta//60} minutes]. 

The data contains timestamps and a list of anomalous metrics that were anomalous in that timestamp interval. The meaning of each heading is as follows:
`anomaly interval mean`: mean of the metric in the timestamp interval where it was anomalous 
`overall mean`: mean of the metric over the monitored duration
`anomaly interval 99th percentile`: 99th percentile value of the metric in the timestamp interval where it was anomalous 
`overall mean 99th percentile`: mean of 99th percentile value of all intervals in the monitored duration 

Output Format: Provide a well-structured summary first and then a deeper detailed explanation of your analysis. Make sure no crucial details or metrics are overlooked. Every place you mention a timestamp, use "In the interval between <Timestamp> and <Timestamp+{self.tdelta//60}> ...."

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
            print("No key found in the environment variable")
            return ""

    def analytics(self, metricObj, queryTimestamp):
        self._init_analytics(metricObj)
        df = self._prepare_dataframe(metricObj)
        # print(df)
        pos = np.where(df.index == queryTimestamp)[0][0]

        tbounds, t0, typ = self.calcBounds(df, pos, self.tdelta//2)
        if typ == -1:
            raise ValueError(
                "No ticket drop found in the nearby interval. Please try with another timestamp or a higher bucket")
        to_monitor = []
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
        self._save_gpt_str_base(gpt_str_base)
        gpt_res = self._openAI_req(gpt_str_base)
        vertical = (df.index[t0])
        tickets = ['ss wt concurrentTransactions.write.out',
                   'ss wt concurrentTransactions.read.out']
        for tick in tickets:
            if tick not in to_monitor:
                to_monitor.append(tick)

        self.__plot(df.iloc[tbounds[0]:tbounds[-1]],
                    to_monitor, vert_x=vertical, gpt_out=gpt_res)

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
        self.analytics(data, self.queryTimeStamp)
