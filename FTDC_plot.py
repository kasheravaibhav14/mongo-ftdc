import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import numpy as np
import matplotlib.ticker as mtick
import pandas as pd
import os
import plotly.graph_objects as go

class FTDC_plot:
    def __init__(self, df, to_monitor,outfilepath="report.pdf"):
        self.df = df
        self.to_monitor = to_monitor
        self.outPath = outfilepath
    
    def plot(self):
        # Prepare the canvas
        c = canvas.Canvas(self.outPath, pagesize=A3)
        page_width, page_height = A3
        # plt.cm.get_cmap('magma')
        plt.style.use('ggplot')
        # Define column widths
        total_parts = 2 + 2 + 7 + 4
        column_widths = [page_width / total_parts * i for i in [2, 2, 7, 4]]

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
                P.drawOn(c, sum(column_widths[:idx]), page_height - header_height)
        
        draw_headers()
        for i in range(len(self.to_monitor)):
            # If we don't have enough space for the next pair, create a new page
            if current_y - space_per_pair < 0:
                c.showPage()
                current_y = page_height - image_height - header_height - padding
                draw_headers()  # Draw headers at the start of each new page
            # Generate random data
            x = self.df.index
            y = self.df[self.to_monitor[i]]
            minval=self.df[self.to_monitor[i]].min()
            meanval=self.df[self.to_monitor[i]].mean()
            maxval=self.df[self.to_monitor[i]].max()
            if "concurrent" in self.to_monitor[i]:
                print(self.to_monitor[i],minval,meanval,maxval)

            # Create a plot
            fig, ax = plt.subplots(figsize=(image_width / 100, image_height / 100))
            ax.plot(x, y, linewidth=0.5)
            if current_y - 2*space_per_pair <0 or i==len(self.to_monitor)-1:
                ax.set_xlabel('Timestamp',fontsize=8) # print the y label only when it is last plot of page
            # ax.set_xticks([])
            # ax.set_yticks([])
            plt.yticks(fontsize=5,fontname='Helvetica')
            plt.xticks(fontsize=5)
            fmt = mtick.StrMethodFormatter('{x:0.2e}')
            ax.yaxis.set_major_formatter(fmt)
            inc = (maxval-minval)*0.1
            ax.set_ylim(minval-inc,maxval+inc)
            # # ax.set_ylabel('y')

            # Add a horizontal line at y=0.5
            ax.axhline(y=meanval, color='black', linestyle='--',linewidth=0.25)

            # Save the plot as an SVG image file
            plt.savefig(f"plot_temp_{i+1}.svg", format="svg")
            plt.close()
            

            # Add the image to the PDF
            drawing = svg2rlg(f"plot_temp_{i+1}.svg")
            scaleX = image_width / drawing.width
            scaleY = image_height / drawing.height
            drawing.width = image_width
            drawing.height = image_height
            drawing.scale(scaleX, scaleY)
            renderPDF.draw(drawing, c, sum(column_widths[:2]), current_y)

            # Add the corresponding text to the PDF
            text_before1 = f"<font size=8>{meanval:,.3f}</font>"
            text_before2 = f"<font size=8>{maxval:,.3f}</font>"
            # text_before2 = f"<font size=12>Before 2 plot {i+1}.</font>"
            text_after = f"<font size=10>{self.to_monitor[i]}.</font>"

            P = Paragraph(text_before1, styleN)
            P.wrapOn(c, column_widths[0], 500)
            P.drawOn(c, sum(column_widths[:1]) - column_widths[0], current_y+padding)

            P = Paragraph(text_before2, styleN)
            P.wrapOn(c, column_widths[1], 500)
            P.drawOn(c, sum(column_widths[:2]) - column_widths[1], current_y+padding)

            P = Paragraph(text_after, styleN)
            P.wrapOn(c, column_widths[3], 500)
            P.drawOn(c, sum(column_widths[:3]), current_y+padding)

            # Remove the SVG file
            os.remove(f"plot_temp_{i+1}.svg")

            # Update the current y position
            current_y -= space_per_pair

        # Save the PDF
        c.showPage()
        c.save()
