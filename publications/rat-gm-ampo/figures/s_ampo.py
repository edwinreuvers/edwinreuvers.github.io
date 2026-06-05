#| label: suppfig-s-ampo
#| fig-cap: Predicted influence of cycle frequency (A), FTS (B) and MTC length excursion 
#|   (C) on the maximum attainable AMPO**. The maximally attainable AMPO — averaged across 
#|   the three rats —  slightly increased in SSCs without a constraint on MTC length over 
#|   time (orange lines), compared to SSCs with a constant MTC shortening and lengthening 
#|   velocity (black lines).

#%%
import os,sys
import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import cust_fig

plt.close('all')

#%%
cust_fig.style(plt, fontname='Minion Pro',fontsize=11,grid=False)
fig = plt.figure(figsize=(15.92/2.54, 4.77/2.54), constrained_layout=True) # width & height  
gs = fig.add_gridspec(1,3)
axs = [fig.add_subplot(gs[i]) for i in range(0,gs.ncols*gs.nrows)]

colorSet = plt.rcParams['axes.prop_cycle'].by_key()['color']
colorSet[0] = '#000000'

sf = 1
    
#%% Imposed CF
cfSet = np.arange(1.0,6.1,0.5)
for i,traj in enumerate(['simsCV', 'simsOC']):
    AMPOall = []
    for mus in ['GMe1', 'GMe2', 'GMe3']:
        dataFolder = os.path.join(dataDir,mus,traj,'')
        AMPOmus = []
        for cf in cfSet:
            try:
                filepath = os.path.join(dataFolder, f'{mus}_cf{cf:0.1f}Hz_ftsOpt_mleOpt.csv')
                df = pd.read_csv(filepath)
                data = df.to_numpy()
                time,lmtc,_,fsee = data.T[0:4]
                Wmech = -integrate.trapezoid(fsee,lmtc)*1e3 # [mJ]
                AMPOmus.append(Wmech/time[-1]) # [mW]
            except:
                AMPOmus.append(np.nan)
            
        AMPOmus = np.array(AMPOmus)/sf      
        AMPOall.append(AMPOmus)
    axs[0].plot(cfSet,np.mean(AMPOall,0),'.-',color=colorSet[i],clip_on=False)
    
#%% Imposed FTS
ftsSet = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
ftsSet = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
for i,traj in enumerate(['simsCV', 'simsOC']):
    AMPOall = []
    for mus in ['GMe1', 'GMe2', 'GMe3']:
        dataFolder = os.path.join(dataDir,mus,traj,'')
        AMPOmus = []
        for fts in ftsSet:
            try:
                filepath = os.path.join(dataFolder, f'{mus}_cfOpt_fts{fts:0.2f}_mleOpt.csv')
                df = pd.read_csv(filepath)
                data = df.to_numpy()
                time,lmtc,_,fsee = data.T[0:4]
                Wmech = -integrate.trapezoid(fsee,lmtc)*1e3 # [mJ]
                AMPOmus.append(Wmech/time[-1]) # [mW]
            except:
                AMPOmus.append(np.nan)
            
        AMPOmus = np.array(AMPOmus)/sf      
        AMPOall.append(AMPOmus)
    axs[1].plot(ftsSet,np.mean(AMPOall,0),'.-',color=colorSet[i],clip_on=False)
    
#%% Imposed MLE
mleSet = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
for i,traj in enumerate(['simsCV', 'simsOC']):
    AMPOall = []
    for mus in ['GMe1', 'GMe2', 'GMe3']:
        dataFolder = os.path.join(dataDir,mus,traj,'')
        AMPOmus = []
        for mle in mleSet:
            try:
                filepath = os.path.join(dataFolder, f'{mus}_cfOpt_ftsOpt_mle{mle:04.1f}mm.csv')
                df = pd.read_csv(filepath)
                data = df.to_numpy()
                time,lmtc,_,fsee = data.T[0:4]
                Wmech = -integrate.trapezoid(fsee,lmtc)*1e3 # [mJ]
                AMPOmus.append(Wmech/time[-1]) # [mW]
            except:
                AMPOmus.append(np.nan)
            
        AMPOmus = np.array(AMPOmus)/sf      
        AMPOall.append(AMPOmus)
    axs[2].plot(mleSet,np.mean(AMPOall,0),'.-',color=colorSet[i],clip_on=False)
    
#%%
leg = axs[2].legend(['Constant', 'Optimal'],
    title='MTC velocities',
    title_fontproperties={'weight': 'bold'},
    loc='lower right',
    bbox_to_anchor=(1.08, 0),  # shift left (into space between axes) and center vertically
    frameon=False,
    handlelength=1,
    handletextpad=0.5,
    labelspacing=0.2,
    alignment='right'
)
    
#%% Labels etc.
axs[0].set_xlabel('Cycle frequency [Hz]')
axs[1].set_xlabel('FTS [ ]')
axs[2].set_xlabel('MTC length excursion [mm]',x=0.395)
axs[0].set_ylabel('AMPO [mW]')
axs[1].set_ylabel('AMPO [mW]')
axs[2].set_ylabel('AMPO [mW]')

axs[0].set_xlim(1,6)
axs[0].set_xticks([2,4,6])
axs[0].set_xticks([1,3,5],minor=True)

axs[1].set_xlim(0.25,1)
axs[1].set_xticks([0.25,0.50,0.75,1.00])
axs[1].set_xticks([0.375,0.625,0.875], minor=True)

ax = axs[2]
ax.set_xlim(2,11)
ax.set_xticks([2,6,10])
ax.set_xticks([4,8], minor=True)

for ax in axs:
    ax.set_ylim(50,175)
    ax.set_yticks([50,100,150])
    ax.set_yticks([75,125,175],minor=True)

for ax in axs:
    ax.spines['left'].set_position(('outward', 12)) 

fig.align_labels()
cust_fig.add_labels(fig, axs, ['A','B','C'], -15/72)

# %% Show and save
plt.show()
# fig.savefig("s_ampo.png", bbox_inches="tight", pad_inches=0, dpi=600)
fig.savefig("s_ampo.pdf", bbox_inches="tight", pad_inches=0)
# fig.savefig("s_ampo.svg", bbox_inches="tight", pad_inches=0)

# %% Checks
if len(sys.argv) > 1:
    check_size = sys.argv[1]
else:
    check_size = True  

if check_size == True or check_size == 'True':
    cust_fig.report_axes_size(fig,axs)
    # cust_fig.report_fig_size("s_ampo.png") 
    cust_fig.report_fig_size("s_ampo.pdf") 
    # cust_fig.report_fig_size("s_ampo.svg")