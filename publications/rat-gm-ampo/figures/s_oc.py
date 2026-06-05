#| label: suppfig-s-oc
#| fig-cap: Predicted CE length over time for maximally attainable AMPO, shown for two 
#|   imposed FTS values (A) and two imposed MTC length excursions (B), at a cycle frequency 
#|   of 3.5 Hz for rat 1**. In panel A, only cycle frequency and FTS were imposed, while in 
#|   panel B only cycle frequency and MTC length excursions were imposed. Apart from these 
#|   two constraints, CE and MTC length over time were completely unconstrained and thus the 
#|   shape of CE length over time could be different between every combination of imposed 
#|   cycle frequency and FTS (A) or imposed cycle frequency and MTC length excursion (B). 
#|   CE stimulation was maximal during the period indicated by the coloured bars, and 'off' 
#|   elsewhere.

#%%
import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import cust_fig, stimulation

plt.close('all')

#%% Select rat
mus = 'GMe1'
parFile = os.path.join(dataDir,mus,mus+'_IM.pkl')
muspar = pickle.load(open(parFile, 'rb'))[0]
dataDirSim = os.path.join(dataDir,mus,'simsOC','')

#%%
cust_fig.style(plt, fontname='Minion Pro',fontsize=11,grid=False)
fig = plt.figure(figsize=(15.92/2.54+50/600, 5.37/2.54), constrained_layout=True) # 3:2 ratio
gs = fig.add_gridspec(2, 2, height_ratios=[1/13.6,1], wspace=0.3/3)
axs = [fig.add_subplot(gs[i]) for i in range(0,gs.ncols*gs.nrows)]

colorSet = plt.rcParams['axes.prop_cycle'].by_key()['color']
colorSet[0] = '#000000'

#%% Impsed FTS (&CF)
cf = 3.5
for iFts,fts in enumerate([0.5, 0.95]):
    fileName = mus+f'_cf{cf:0.1f}Hz_fts{fts:0.2f}_mleOpt'
    
    # Experimental data
    df = pd.read_csv(dataDirSim+fileName+'.csv')
    data = df.to_numpy()
    time,lmtc,stim,fsee,gamma,lcerel = data.T[0:6]
    time = np.concatenate((time-time[-1],time,time+time[-1]))
    lcerel = np.concatenate((lcerel,lcerel,lcerel))
    lce = lcerel*muspar['lce_opt']*1e3
    stim = np.concatenate((stim,stim,stim))
    
    # Plot STIM(t)
    tStimOn, tStimOff = stimulation.get_stim_timing(time,stim)
    cust_fig.plot_stim(axs[0],tStimOn[0],tStimOff[0],y=iFts, lw=1/3, color=colorSet[iFts])
    cust_fig.plot_stim(axs[0],tStimOn[1],tStimOff[1],y=iFts, lw=1/3, color=colorSet[iFts])
    cust_fig.plot_stim(axs[0],tStimOn[2],tStimOff[2],y=iFts, lw=1/3, color=colorSet[iFts])

    # Plot Lce(t)
    axs[2].plot(time,lce, color=colorSet[iFts], label=f'{fts:.2f}')

axs[2].set_ylim(8, 20)
ax2T = axs[2].twinx()
ax2T.spines['right'].set_visible(True)
scale_factor = muspar['lce_opt']*1000
y1_min, y1_max = axs[2].get_ylim()
ax2T.set_ylim(y1_min / scale_factor, y1_max / scale_factor)
# ax2T.plot(time,lcerel)

#%% Impsed AMP (&CF)
cf = 3.5
for iFts,mle in enumerate([2, 10]):
    fileName = mus+f'_cf{cf:0.1f}Hz_ftsOpt_mle{mle:04.1f}mm'
    
    # Experimental data
    df = pd.read_csv(dataDirSim+fileName+'.csv')
    data = df.to_numpy()
    time,lmtc,stim,fsee,gamma,lcerel = data.T[0:6]
    time = np.concatenate((time-time[-1],time,time+time[-1]))
    lcerel = np.concatenate((lcerel,lcerel,lcerel))
    lce = lcerel*muspar['lce_opt']*1e3
    stim = np.concatenate((stim,stim,stim))
    
    # Plot STIM(t)
    tStimOn, tStimOff = stimulation.get_stim_timing(time,stim)
    cust_fig.plot_stim(axs[1],tStimOn[0],tStimOff[0],y=iFts, lw=1/3, color=colorSet[iFts])
    cust_fig.plot_stim(axs[1],tStimOn[1],tStimOff[1],y=iFts, lw=1/3, color=colorSet[iFts])
    cust_fig.plot_stim(axs[1],tStimOn[2],tStimOff[2],y=iFts, lw=1/3, color=colorSet[iFts])
    
    # Plot Lce(t)
    axs[3].plot(time,lce, color=colorSet[iFts], label=f'{mle:.2f}')

axs[3].set_ylim(8,20)
ax3T = axs[3].twinx()
ax3T.spines['right'].set_visible(True)
scale_factor = muspar['lce_opt']*1000
y1_min, y1_max = axs[3].get_ylim()
ax3T.set_ylim(y1_min / scale_factor, y1_max / scale_factor)
# ax3T.plot(time,lcerel)

#%% Legends
axs[2].legend(['0.50', '0.95'],
    title='FTS',
    title_fontproperties={'weight': 'bold'},
    loc='upper right',
    bbox_to_anchor=(0.98, 1.2),  # shift left (into space between axes) and center vertically
    frameon=False,
    handlelength=1,
    handletextpad=0.5,
    labelspacing=0.2,
    alignment='right'
)


axs[3].legend(['2 mm', '10 mm'],
    title='MTC length excursion',
    title_fontproperties={'weight': 'bold'},
    loc='upper right',
    bbox_to_anchor=(1.03, 1.2),  # shift left (into space between axes) and center vertically
    frameon=False,
    handlelength=1,
    handletextpad=0.5,
    labelspacing=0.2,
    alignment='right',
)

#%% Labels etc.
# STIM(t)
for ax in axs[0:2]:
    ax.set_xlim(-0.25/3.5,1.75/3.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.autoscale(enable=True, axis='y', tight=True)
    # ax.set_ylim(-1/3,4/3)
  
# Lmtc(t)
for ax in axs[2:]:
    ax.set_xlim(-0.25/3.5,1.75/3.5)
    ax.set_xticks([0,0.2,0.4])
    ax.set_xticks([0.1,0.3,0.5],minor=True)
    # ax.set_ylim(8.5,19.5)
    ax.set_yticks([10,14,18])
    ax.set_yticks([12,16],minor=True)
    ax.set_ylabel('$L_{CE}$ [mm]')
    
for ax in [ax2T, ax3T]:
    ax.set_yticks([0.6,1.0,1.4])
    ax.set_yticks([0.8,1.2],minor=True)
    ax.set_ylabel('$L_{CE}^{rel}$ [ ]')

axs[2].set_xlabel('Time [s]')
axs[3].set_xlabel('Time [s]')

fig.align_labels()
cust_fig.add_labels(fig, axs, ['','','A','B'])

# %% Show and save
plt.show()
# fig.savefig("s_oc.png", bbox_inches="tight", pad_inches=0, dpi=600)
fig.savefig("s_oc.pdf", bbox_inches="tight", pad_inches=0)
# fig.savefig("s_oc.svg", bbox_inches="tight", pad_inches=0)

# %% Checks
if len(sys.argv) > 1:
    check_size = sys.argv[1]
else:
    check_size = True  

if check_size == True or check_size == 'True':
    cust_fig.report_axes_size(fig,axs)
    # cust_fig.report_fig_size("s_oc.png") 
    cust_fig.report_fig_size("s_oc.pdf") 
    # cust_fig.report_fig_size("s_oc.svg")