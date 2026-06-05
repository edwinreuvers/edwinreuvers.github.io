#| label: fig-r-interrelationships
#| fig-cap: Predicted interrelationships between cycle frequency (CF), FTS, MTC length
#|   excursion (MLE) and the maximally attainable AMPO. These plots collectively illustrate how cycle
#|   frequency, FTS, and MTC length excursion interact to maximise AMPO under different conditions. This
#|   figure presents a 3x3 grid of plots created by imposing one SSC parameter at a time while optimising the two
#|   other SSC parameters to maximise AMPO. In the first column, cycle frequency was imposed, and AMPO
#|   (A1) was maximised by identifying the optimal FTS (A2) and MTC length excursion (A3) at each cycle
#|   frequency. In the second column, FTS was imposed, and AMPO (A1) was maximised by identifying the
#|   optimal MTC length excursion (B2) and cycle frequency (B3) at FTS. In the third column, MTC length
#|   excursion was imposed, and AMPO (C1) was maximised by identifying the optimal cycle frequency (C2) and
#|   FTS (C3) at each MTC length excursion. For all columns, muscle stimulation onset time and duration were
#|   optimised to maximise AMPO.

#%% Load packages & set directories
import os, sys, pickle
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
customlay = cust_fig.style(plt, fontname='Minion Pro',fontsize=11,grid='false')

fig = plt.figure(figsize=(15.92/2.54+1/600, 11.9/2.54), constrained_layout=True) # 3:2 ratio  
gs = fig.add_gridspec(3,3)
# fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)
fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0.1)
axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(gs.ncols)] for i in range(gs.nrows)])

#%%
nInterpol = 100
colorSet = plt.rcParams['axes.prop_cycle'].by_key()['color']
colorSet[0] = '#000000'
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', colorSet)

symbols = ['o','^', 's']
markersize = 3
for iMus,mus in enumerate(['GMe1', 'GMe2',  'GMe3']):
    dataDirSim = os.path.join(dataDir,mus,'simsCV','')
    parFile = os.path.join(dataDir,mus,mus+'_IM.pkl')
    muspar = pickle.load(open(parFile, 'rb'))[0]
    
    sf = muspar['fmax']*muspar['lce_opt']*1e3
    sf = 1
    
    #%% Imposed CF
    cf_set = np.arange(1.0,6.1,0.5)
    AMPO, cf_imposed, fts_opt, mle_opt = [], [], [], []
    for cf in cf_set:
        try:
            fileName = mus+f'_cf{cf:0.1f}Hz_ftsOpt_mleOpt'
            
            df = pd.read_csv(dataDirSim+fileName+'.csv')
            data = df.to_numpy()
            time,lmtc,_,fsee = data.T[0:4]
            Wmech = -integrate.trapezoid(fsee,lmtc) # [J]
            AMPO.append(Wmech*cf) # [W]
            
            cf_imposed.append(cf)
            fts_opt.append(np.argmin(lmtc)/(len(lmtc)-1)) # [ ]
            mle_opt.append((np.max(lmtc)-np.min(lmtc))) # [m]
        except:
            None
            
    AMPO = np.array(AMPO)/sf
    cf_imposed = np.array(cf_imposed)
    fts_opt = np.array(fts_opt)
    mle_opt = np.array(mle_opt)  
      
    axs[0,0].plot(cf_imposed,AMPO*1e3,color=colorSet[iMus],marker=symbols[iMus],ms=markersize,clip_on=False)
    axs[1,0].plot(cf_imposed,fts_opt,color=colorSet[iMus],marker=symbols[iMus],ms=markersize,clip_on=False)
    axs[2,0].plot(cf_imposed,mle_opt*1e3,color=colorSet[iMus],marker=symbols[iMus],ms=markersize,clip_on=False)
    
    #%% Imposed FTS
    fts_set = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    AMPO, fts_imposed, cf_opt, mle_opt = [], [], [], []
    for fts in fts_set:
        try:
            fileName = mus+f'_cfOpt_fts{fts:{"0.2f"}}_mleOpt'
        
            df = pd.read_csv(dataDirSim+fileName+'.csv')
            data = df.to_numpy()
            time,lmtc,stim,fsee = data.T[0:4]
            Wmech = -integrate.trapezoid(fsee,lmtc) # [J]
            AMPO.append(Wmech/time[-1]) # [W]
            
            fts_imposed.append(fts)
            cf_opt.append(1/time[-1])
            mle_opt.append((np.max(lmtc)-np.min(lmtc))) # [m]
        except:
            None
    
    AMPO = np.array(AMPO)/sf
    fts_imposed = np.array(fts_imposed)
    cf_opt = np.array(cf_opt)
    mle_opt = np.array(mle_opt)
    
    axs[0,1].plot(fts_imposed,AMPO*1e3,color=colorSet[iMus],marker=symbols[iMus],ms=markersize,clip_on=False)
    axs[1,1].plot(fts_imposed,mle_opt*1e3,color=colorSet[iMus],marker=symbols[iMus],ms=markersize,clip_on=False)
    axs[2,1].plot(fts_imposed,cf_opt,color=colorSet[iMus],marker=symbols[iMus],ms=markersize,clip_on=False)
    
    #%% Imposed MLE
    mle_set = np.arange(2,11,1)*1e-3
    AMPO, mle_imposed, cf_opt, fts_opt = [], [], [], []
    for mle in mle_set:
        try:
            fileName = mus+f'_cfOpt_ftsOpt_mle{mle*1e3:{"0.1f"}}mm'
            df = pd.read_csv(dataDirSim+fileName+'.csv')
            data = df.to_numpy()
            time,lmtc,stim,fsee = data.T[0:4]
            Wmech = -integrate.trapezoid(fsee,lmtc) # [J]
            AMPO.append(Wmech/time[-1]) # [W]
            
            mle_imposed.append(mle)
            cf_opt.append(1/time[-1])
            fts_opt.append(np.argmin(lmtc)/(len(lmtc)-1)) # [ ]
        except:
            continue
    
    AMPO = np.array(AMPO)/sf
    mle_imposed = np.array(mle_imposed)
    cf_opt = np.array(cf_opt)
    fts_opt = np.array(fts_opt)
        
    axs[0,2].plot(mle_imposed*1e3,AMPO*1e3,color=colorSet[iMus],marker=symbols[iMus],ms=markersize,clip_on=False)
    axs[1,2].plot(mle_imposed*1e3,cf_opt,color=colorSet[iMus],marker=symbols[iMus],ms=markersize,clip_on=False)
    axs[2,2].plot(mle_imposed*1e3,fts_opt,color=colorSet[iMus],marker=symbols[iMus],ms=markersize,clip_on=False)
    
#%%
legend = axs[1,2].legend(['1', '2', '3'],
                       title='Rat',
                       title_fontproperties={'weight': 'bold'},
                       loc='upper right',
                       bbox_to_anchor=(1.05, 1.06),
                       alignment='right',
                       handlelength=0.8,
                       handletextpad=0.5,
                       labelspacing=0.2)

# Effect of CF
for ax in axs[:,0]:
    ax.set_xlim(1,6)
    ax.set_xticks([2,4,6])
    ax.set_xticks([1,3,5], minor=True)

# Effect of FTS
for ax in axs[:,1]:
    ax.set_xlim(0.25,1)
    ax.set_xticks([0.25,0.50,0.75,1.00])
    ax.set_xticks([0.375,0.625,0.875], minor=True)

# Effect of MLE
for ax in axs[:,2]:
    ax.set_xlim(2,11)
    ax.set_xticks([2,6,10])
    ax.set_xticks([4,8], minor=True)

# Effect on AMPO
for ax in axs[0,:]:
    ax.set_ylim(37.5,187.5)
    ax.set_yticks([50,100,150])
    ax.set_yticks([75,125,175], minor=True)

# Effect on FTS
for ax in [axs[1,0], axs[2,2]]:
    ax.set_ylim(0.835,0.885)
    ax.set_yticks([0.84,0.86,0.88])
    ax.set_yticks([0.85,0.87], minor=True)
ax = axs[1,0]
ax.set_ylim(0.835,0.885)
ax.set_yticks([0.84,0.86,0.88])
ax.set_yticks([0.85,0.87], minor=True)
ax = axs[2,2]
ax.set_ylim(0.835,0.855)
ax.set_yticks([0.84,0.85])
ax.set_yticks([0.845], minor=True)

# Effect on MLE
for ax in [axs[1,1], axs[2,0]]:
    ax.set_ylim(4.5,11.5)
    ax.set_yticks([6,8,10])
    ax.set_yticks([5,7,9], minor=True)
ax = axs[1,1]
ax.set_ylim(4.5,9)
ax.set_yticks([6,8])
ax.set_yticks([5,7,9], minor=True)
ax = axs[2,0]
ax.set_ylim(3.5,13.5)
ax.set_yticks([4,8,12])
ax.set_yticks([6,10], minor=True)

# Effect on CF
for ax in [axs[1,2], axs[2,1]]:
    ax.set_ylim(1,9)
    ax.set_yticks([2,4,6,8])
    ax.set_yticks([1,3,5,7,9], minor=True)
ax = axs[1,2]
ax.set_ylim(2.5,9.5)
ax.set_yticks([4,6,8])
ax.set_yticks([3,5,7,9], minor=True)
ax = axs[2,1]
ax.set_ylim(1.75,4.25)
ax.set_yticks([2,3,4])
ax.set_yticks([2.5,3.5], minor=True)

#%% Labels etc.
axs[2,0].set_xlabel('CF [Hz]')
axs[2,1].set_xlabel('FTS [ ]')
axs[2,2].set_xlabel('$\Delta L_{MTC}$ [mm]')
axs[0,0].set_ylabel('AMPO [mW]')
axs[0,1].set_ylabel('AMPO [mW]')
axs[0,2].set_ylabel('AMPO [mW]')
axs[1,0].set_ylabel('FTS [ ]')
axs[2,0].set_ylabel('$\Delta L_{MTC}$ [mm]')
axs[1,1].set_ylabel('$\Delta L_{MTC}$ [mm]')
axs[2,1].set_ylabel('CF [Hz]')
axs[1,2].set_ylabel('CF [Hz]')
axs[2,2].set_ylabel('FTS [ ]')

for ax in axs.flatten():
    ax.spines['left'].set_position(('outward', 12)) 
    ax.yaxis.labelpad = 0

fig.align_labels()

#%%
labels = ['A1','B1','C1', 'A2','B2','C2', 'A3','B3','C3']
cust_fig.add_labels(fig, axs.flatten(), labels, -15/72)
    
# %% Show and save
plt.show()
# # fig.savefig("r_interrelationships.png", bbox_inches="tight", pad_inches=0, dpi=600)
fig.savefig("r_interrelationships.pdf", bbox_inches="tight", pad_inches=0)
# # fig.savefig("r_interrelationships.svg", bbox_inches="tight", pad_inches=0)

# %% Checks
if len(sys.argv) > 1:
    check_size = sys.argv[1]
else:
    check_size = True  

if check_size == True or check_size == 'True':
    cust_fig.report_axes_size(fig,axs)
    # cust_fig.report_fig_size("r_interrelationships.png") 
    cust_fig.report_fig_size("r_interrelationships.pdf") 
    # cust_fig.report_fig_size("r_interrelationships.svg") 