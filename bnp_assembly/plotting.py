import logging
import plotly.express as _px

class DummyPlot:
    def show(self):
        pass
        

class Dummy:
    def __getattr__(self, name):
        return lambda *args, **kwargs: DummyPlot()
        

level_dict  = {'debug': logging.DEBUG,
               'info': logging.INFO}

def px(level=logging.INFO):
    if isinstance(level, str):
        level = level_dict[level.lower()]
    if logging.root.level<=level:
        return _px
    return Dummy()
