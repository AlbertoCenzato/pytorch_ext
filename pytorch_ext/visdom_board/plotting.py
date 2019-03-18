from typing import Optional

import numpy as np
import plotly.graph_objs as go

import torch

import visdom

from .core import VisObject


colorscale = ['rgba(229,   0,  53, 255)', 
              'rgba(227,   0,  29, 255)', 
              'rgba(226,   0,   5, 255)', 
              'rgba(225,  17,   0, 255)', 
              'rgba(223,  40,   0, 255)', 
              'rgba(222,  63,   0, 255)', 
              'rgba(221,  85,   0, 255)',
              'rgba(219, 108,   0, 255)',
              'rgba(218, 130,   0, 255)',
              'rgba(217, 151,   0, 255)',
              'rgba(215, 173,   0, 255)',
              'rgba(214, 194,   0, 255)',
              'rgba(211, 213,   0, 255)',
              'rgba(188, 211,   0, 255)',
              'rgba(165, 210,   0, 255)',
              'rgba(142, 209,   0, 255)',
              'rgba(120, 208,   0, 255)',
              'rgba( 98, 206,   0, 255)',
              'rgba( 76, 205,   0, 255)',
              'rgba( 54, 204,   0, 255)',
              'rgba( 33, 202,   0, 255)',
              'rgba( 12, 201,   0, 255)',
              'rgba(  0, 200,   8, 255)',
              'rgba(  0, 198,  28, 255)',
              'rgba(  0, 197,  48, 255)',
              'rgba(  0, 196,  68, 255)',
              'rgba(  0, 194,  88, 255)',
              'rgba(  0, 193, 107, 255)',
              'rgba(  0, 192, 127, 255)',
              'rgba(  0, 191, 146, 255)']

colorscale_alpha = ['rgba(229,   0,  53, 0.2)', 
                    'rgba(227,   0,  29, 0.2)', 
                    'rgba(226,   0,   5, 0.2)', 
                    'rgba(225,  17,   0, 0.2)', 
                    'rgba(223,  40,   0, 0.2)', 
                    'rgba(222,  63,   0, 0.2)', 
                    'rgba(221,  85,   0, 0.2)',
                    'rgba(219, 108,   0, 0.2)',
                    'rgba(218, 130,   0, 0.2)',
                    'rgba(217, 151,   0, 0.2)',
                    'rgba(215, 173,   0, 0.2)',
                    'rgba(214, 194,   0, 0.2)',
                    'rgba(211, 213,   0, 0.2)',
                    'rgba(188, 211,   0, 0.2)',
                    'rgba(165, 210,   0, 0.2)',
                    'rgba(142, 209,   0, 0.2)',
                    'rgba(120, 208,   0, 0.2)',
                    'rgba( 98, 206,   0, 0.2)',
                    'rgba( 76, 205,   0, 0.2)',
                    'rgba( 54, 204,   0, 0.2)',
                    'rgba( 33, 202,   0, 0.2)',
                    'rgba( 12, 201,   0, 0.2)',
                    'rgba(  0, 200,   8, 0.2)',
                    'rgba(  0, 198,  28, 0.2)',
                    'rgba(  0, 197,  48, 0.2)',
                    'rgba(  0, 196,  68, 0.2)',
                    'rgba(  0, 194,  88, 0.2)',
                    'rgba(  0, 193, 107, 0.2)',
                    'rgba(  0, 192, 127, 0.2)',
                    'rgba(  0, 191, 146, 0.2)']


class Plot(VisObject):

    def __init__(self, vis: visdom.Visdom, env: Optional[str]=None, 
                 title: str='', xaxis: str='', yaxis: str=''):
        super(Plot, self).__init__(vis, env)
        self.title = title
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.opts = dict(title=title, xlabel=xaxis, ylabel=yaxis)


class LinePlot(Plot):

    def __init__(self, vis: visdom.Visdom, env: Optional[str]=None, 
                 title='', xaxis='', yaxis=''):
        super(LinePlot, self).__init__(vis, env, title, xaxis, yaxis)

    def plot(self, x: np.array, y: np.array) -> None:
        if not self.check_connection():
            return

        if self._win is None:
            self._win = self._vis.line(X=x, Y=y, opts=self.opts, env=self._env)
        else:
            self._vis.line(X=x, Y=y, opts=self.opts, win= self._win, env=self._env)

    def append(self, x: np.array, y: np.array) -> None:
        if not self.check_connection():
            return

        if self._win is None:
            self._win = self._vis.line(X=x, Y=y, opts=self.opts, env=self._env)
        else:
            self._vis.line(X=x, Y=y, win=self._win, opts=self.opts, env=self._env, update='append')


class LineStdPlot(Plot):

    def __init__(self, vis: visdom.Visdom, env: Optional[str]=None, title: str='', 
                 xaxis: str='', yaxis: str='', total_traces: int=len(colorscale)):
        super(LineStdPlot, self).__init__(vis, env, title, xaxis, yaxis)
        self.color_index = 0
        if total_traces < len(colorscale):
            self.color_increment = len(colorscale) // total_traces
        else:
            self.color_increment = 1
        self.data = []

        self.layout = go.Layout(
                        title=self.title,
                        paper_bgcolor='rgb(255,255,255)',
                        plot_bgcolor='rgb(229,229,229)',
                        xaxis=dict(
                            title=self.xaxis,
                            gridcolor='rgb(255,255,255)',
                            showgrid=True,
                            showline=True,
                            showticklabels=True,
                            tickcolor='rgb(127,127,127)',
                            ticks='outside',
                            zeroline=False
                        ),
                        yaxis=dict(
                            title=self.yaxis,
                            gridcolor='rgb(255,255,255)',
                            showgrid=True,
                            showline=True,
                            showticklabels=True,
                            tickcolor='rgb(127,127,127)',
                            ticks='outside',
                            zeroline=False
                        )
                    )

    def plot(self, mean: np.array, std: np.array, trace_id: str) -> None:
        if not self.check_connection():
            return 

        upper = mean + std
        lower = mean - std
        x = [i for i in range(len(mean))]
        x_rev = x[::-1]
        upper_bound = go.Scatter(
                        x=x+x_rev,
                        y=np.concatenate((upper, lower[::-1])),
                        fill='tozerox',
                        fillcolor=colorscale_alpha[self.color_index],
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=False,
                        legendgroup=trace_id,
                        hoverinfo='none'
                      )
        error_line = go.Scatter(
                        y=mean,
                        mode='lines',
                        line=dict(color=colorscale[self.color_index]),
                        showlegend=True,
                        legendgroup=trace_id,
                        name=trace_id
                      )
        self.data = self.data + [upper_bound, error_line]
        fig = go.Figure(data=self.data, layout=self.layout)
        if self._win is None:
            self._win = self._vis.plotlyplot(fig, env=self._env)
        else:
            self._vis.plotlyplot(fig, win=self._win, env=self._env)
        self.color_index += self.color_increment


class HistogramPlot(Plot):

    def __init__(self, vis: visdom.Visdom, env: Optional[str]=None, 
                 title: str='', xaxis: str='', yaxis: str='', bins: int=50):
        super(HistogramPlot, self).__init__(vis, env, title, xaxis, yaxis)
        self.opts['numbins'] = bins

    def plot(self, data: torch.Tensor) -> None:
        hist_data = data.contiguous().view(-1)

        if self._win:
            self._vis.histogram(X=hist_data, opts=self.opts, win=self._win, env=self._env)
        else:
            self._win = self._vis.histogram(X=hist_data, opts=self.opts, env=self._env)


class RibbonPlot(Plot):

    def __init__(self, vis: visdom.Visdom, env: Optional[str]=None,
                 title: str='', xaxis: str='', yaxis: str='', bins: int=50):
        super(RibbonPlot, self).__init__(vis, env, title, xaxis, yaxis)
        self._bins    = bins
        self._traces  = []
        self._counter = 1

        self.layout = {'title': self.title, 
                       'xaxis': {'title': self.xaxis, 'automargin': True}, 
                       'yaxis': {'title': self.yaxis, 'automargin': True}}

    def plot(self, data: np.array) -> None:
        hist_data, bin_edges = np.histogram(data, bins=self._bins)
        bin_centers = (bin_edges + (bin_edges[1] - bin_edges[0])/2)[:-1]
        z = [[i, i] for i in hist_data]
        y = [[j, j] for j in bin_centers]
        x = [[self._counter-1, self._counter-0.5] for _ in range(len(hist_data))]
        ci = 12*(self._counter-1)  # ci = "color index"
        trace = dict(z=z, x=x, y=y,
                     colorscale=[[i, 'rgb(%d,%d,255)' % (ci, ci)] for i in np.arange(0, 1.1, 0.1)],
                     showscale=False,
                     type='surface',
                    )
        self._traces.append(trace)

        fig = {'data': self._traces, 'layout': self.layout}  # go.Figure(data=self._traces, layout=self.layout)

        if self._win is None:
            self._win = self._vis.plotlyplot(fig, env=self._env)
        else:
            self._vis.plotlyplot(fig, win=self._win, env=self._env)

        self._counter += 1
