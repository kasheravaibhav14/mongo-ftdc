import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import numpy as np
import matplotlib.ticker as mtick
import pandas as pd
import os
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
                        text_subheading = f"<font size=12><i>{self.expandedSub[subheading]}</i></font>"
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

        self.dumpGPT(c, current_y)
        c.save()
