# -*- coding: utf-8 -*-
"""
Created on Wed May 29 23:17:21 2024

@author: Starr
"""

# from ipynb.fs.full.plotfunc import draw, create_widgets

from plotfunc import draw_novideo, draw_video, timedata, draw_video_timedata
from IPython.display import display
import ipywidgets as widgets
# Example usage of the draw function with parameters
targets = 10000
beta = 2
agents = 10
mu = 1.1
runs = 25
#%%
# tickspeed = 5
file = "simulation20.h5"
tick_speed = 1
alpha = 1e-5
Rv = 0.01
rho = 0.2
# draw_video(targets, beta, agents, mu, alpha, rho, Rv, tick_speed, filename, runs, save)
#%%
save = None
rho_series = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7]
for rho in rho_series:
    print(rho)
    filename = 'Rv' + str(Rv) + 'rho=' + str(rho) + '_' + file
    videofilename = 'rho='+ str(rho) + '_motion.mp4'
    timedata(targets, beta, agents, mu, alpha, rho, Rv, tick_speed, runs, filename, save)
