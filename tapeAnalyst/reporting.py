#!/usr/bin/env python

# Built-in
import datetime
TODAY = datetime.date.today().strftime("%m/%d/%Y")

# 3rd Party
from jinja2 import Environment, PackageLoader

class HtmlReport():
    """ Generate an HTML report. 

    :Args:
        :param gel: A processed gel image with gray scale intensities.
        :type gel: TapeStationGel 

    """
    def __init__(self, gelImage, laneSummary, targetRange=None):
        # Set up jinja2 template environment
        env = Environment(loader=PackageLoader('tapeAnalyst', 'templates'))
        template = env.get_template("report.html")

        self.rendered = template.render(date=TODAY, fullGelPlot=gelImage, rows=laneSummary, targetRange=targetRange)

    def write(self, fname):
        """ Write generate output.

        :Args:
            :param fname: Output file name ending in .html.
            :type fname: str
        """
        with open(fname, 'w') as OUT:
            OUT.write(self.rendered)


if __name__ == '__main__':
    pass
