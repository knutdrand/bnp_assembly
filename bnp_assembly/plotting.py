import logging
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from pathlib import PurePath


import re

import numpy as np
import pandas as pd
import plotly.express as _px


class DummyPlot:
    def show(self):
        pass


class Dummy:
    def __getattr__(self, name):
        return lambda *args, **kwargs: DummyPlot()


_registered_names = dict()

level_dict = {'debug': logging.DEBUG,
              'info': logging.INFO}


class ResultFolder:
    def __init__(self, path):
        self.path = path
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def mock_from_value(value):
    if isinstance(value, ResultFolder):
        return FolderSaver(value.path)
    assert False


def register(**kwargs):
    for key, value in kwargs.items():
        if key in _registered_names:
            raise ValueError(f'{key} is already registered')
        _registered_names[key] = mock_from_value(value)


def px(level=logging.INFO, name=None):
    if name is not None and name in _registered_names:
        return _registered_names[name]
    if isinstance(level, str):
        level = level_dict[level]
    if level >= logging.getLogger().level:
        return _px
    return Dummy()


@dataclass
class ResultsFolder:
    path: PurePath


def urlify(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)

    # Replace all runs of whitespace with a single dash
    s = re.sub(r"\s+", '-', s)

    return s


class FolderSaver:
    def __init__(self, folder_name: str):
        self._folder_name = folder_name
        self._file_names =[]

    def array(self, arr, title):
        filename = f'{self._folder_name}/{title}.npy'
        np.save(filename, arr)
        return arr

    def table(self, df, title):
        filename = f'{self._folder_name}/{title}.csv'
        pd.DataFrame(df).to_csv(filename)
        return df

    def decorator(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if 'title' in kwargs:
                title = '-' + urlify(kwargs['title'])
            else:
                title = ''
            fig = func(*args, **kwargs)
            filename=f'{self._folder_name}/{func.__name__}{title}.html'
            fig.write_html(filename)
            print(filename)
            self._file_names.append(f'./{func.__name__}{title}.html')
            self.write_report()
            return fig

        return wrapper

    def __getattr__(self, name):
        return self.decorator(getattr(_px, name))

    def write_report(self):
        with open(f'{self._folder_name}/report.html', 'w') as f:
            f.write(self.generate_html())

    def generate_html(self):
        html = html_string
        print(self._file_names)
        images = [image_template.replace('{{plot_url}}', plot_url) for plot_url in self._file_names]
        body = body_template.replace('{{name}}', 'Plots').replace('{{images}}', '\n'.join(images))
        html = html.replace('{{body}}', body)
        return html


class Report:
    def __init__(self, foldername, names=[]):
        self._filepath = filepath
        self._plots = defaultdict(list)

    def generate_html(self):
        for name, plot_urls in self._plots.items():
            images = '\n'.join([image_template.format(plot_url=plot_url) for plot_url in plot_urls])


html_string = '''
<html>
    <head>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
        <style>body{ margin:0 100; background:whitesmoke; }</style>
    </head>
    <body>
        {{body}}
        <iframe width="1000" height="550" name="plot" frameborder="0" seamless="seamless" scrolling="no" \
src="{{plot_url}}"></iframe>

    </body>
</html>'''

#image_template = '''<iframe width="1000" height="550" name="plot" frameborder="0" seamless="seamless" scrolling="no" \
#src="{{plot_url}}"></iframe>'''
image_template = '''<a href="{{plot_url}}" target="plot">{{plot_url}}</a>'''
# .embed?width=800&height=550
body_template = '''
        <h1>{{name}}</h1>
        {{images}}
'''
