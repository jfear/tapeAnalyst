#!/usr/bin/env python

# Built-in
import datetime
TODAY = datetime.date.today().strftime("%m/%d/%Y")
import pickle

# 3rd Party
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as signal

from jinja2 import Environment, PackageLoader
import base64
from io import BytesIO

class HtmlReport():
    """ Generate an HTML report. 

    :Args:
        :param gel: A processed gel image with gray scale intensities.
        :type gel: TapeStationGel 

    """
    def __init__(self, gel, df):
        # Get the image of the ladder if present
        if gel.ladder is not None:
            ladders = list()
            for ladder in gel.ladder:
                ladders.append(self.pullGrayLane(gel, ladder))

            self.ladderGel = np.column_stack(ladders)
            self.get_molecular_weights()
        else:
            self.ladderGel = None
            self.peaks = None

        # make image of the original gels
        # Generate Figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.imshow(gel.colorGel)
        ax.get_xaxis().set_visible(False)
        self.addMW_Y(ax)
        self.fullGelPlot = self.convertImage(fig)
        plt.close(fig)

        # Iterate over lanes and make report, skip lanes that are ladders
        # because I am adding the ladders seperately
        output = df[df['description'].str.lower() != 'ladder'].apply(lambda row: self.build_output(row, gel), axis=1)
        self.htmlOut = output.values.tolist()

        # build HTML report
        self.rendered = self.build_report()

    def pullGrayLane(self, gel, col, border=5):
        """ Pull out the given lane. """
        lane = gel.laneLocations[col]
        start = lane[0] - border
        end = lane[1] + border
        image = gel.grayGel[:, start:end]
        
        return image

    def build_output(self, row, gel):
        """ """
        # Get Sample information
        index, wellID, conc, description, alert, notes = row

        # Get Gel image
        currGel = self.pullGrayLane(gel, index)
        spacer = np.zeros((currGel.shape[0], 10))

        # Try to combine Gel with ladder
        try:
            outGel = np.column_stack([spacer, currGel, spacer, self.ladderGel, spacer])
        except:
            outGel = currGel

        # Generate Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.imshow(outGel, cmap='Greys')
        ax1.get_xaxis().set_visible(False)
        self.addMW_Y(ax1)
        ax2.plot(currGel.mean(axis=1))
        self.addMW_X(ax2)
        plt.tight_layout()

        # Convert figure to bytestring for embeding
        figdata_png = self.convertImage(fig)
        plt.close(fig)

        return {'lane': index + 1, 'wellID': wellID, 'description': description, 'figdata_png': figdata_png}

    def convertImage(self, fig):
        """ Converts an image into its base64 byte representation """
        figfile = BytesIO()
        fig.savefig(figfile, format='png')

        return base64.b64encode(figfile.getvalue()).decode('utf8')

    def build_report(self):
        """ """
        # Set up jinja2 template environment
        env = Environment(loader=PackageLoader('tapeAnalyst', 'templates'))
        template = env.get_template("report.html")

        return template.render(date=TODAY, fullGelPlot=self.fullGelPlot, rows=self.htmlOut)

    def write(self, fname):
        """ Write generate output.

        :Args:
            :param fname: Output file name ending in .html.
            :type fname: str
        """
        with open(fname, 'w') as OUT:
            OUT.write(self.rendered)

    def get_molecular_weights(self):
        """ """
        # Take mean across ladder
        his = self.ladderGel.mean(axis=1)

        # Set hard filter help make peaks stand out
        his[his < 0.4] = 0

        # Find Peaks
        self.peaks = signal.find_peaks_cwt(his, widths=np.arange(3, 5))
        if len(self.peaks) < 10:
            self.peaks = None
        
        # Molecular Weights
        self.weights = ['1500', '1000', '700', '500', '400', '300', '200', '100', '50', '25']
    
    def addMW_Y(self, ax):
        if not self.peaks is None:
            ax.yaxis.tick_right()
            ax.set_yticks(self.peaks)
            ax.set_yticklabels(self.weights)

    def addMW_X(self, ax):
        if not self.peaks is None:
            ax.set_xticks(self.peaks)
            ax.set_xticklabels(self.weights, rotation=90)


if __name__ == '__main__':
    pass
