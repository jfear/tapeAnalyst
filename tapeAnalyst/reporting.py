#!/usr/bin/env python

from jinja2 import Environment, FileSystemLoader
import base64
import matplotlib.pyplot as plt
from io import BytesIO


def build_report(fig):
    """ """
    env = Environment(loader=FileSystemLoader('./templates'))
    template = env.get_template("report.html")

    figfile = BytesIO()

    fig.savefig(figfile, format='png')
    figdata_png = base64.b64encode(figfile.getvalue()).decode('utf8')

    return template.render(title='bob', mygraph=figdata_png)




if __name__ == '__main__':
    pass
