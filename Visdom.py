import torch
import visdom

class Visdom:
    def __init__(self, name=None):
        self.vis = visdom.Visdom(env=name)
    
    def plot(self, x, y, win, legend=None, title=None, name=None, xlabel=None, ylabel=None):
        self.vis.line(
            X=torch.tensor([x]),
            Y=torch.tensor([y]),
            win=win,
            update='append',
            name=name,
            opts={
                'legend' : [legend],
                'title'  : title,
                'xlabel' : xlabel,
                'ylabel' : ylabel
            }
        )
