#| label: fig-r-ampo-cf-fts
#| fig-cap: Experimentally measured influence of cycle frequency and FTS on AMPO. AMPO as a function of cycle frequency, 
#|   with a fixed FTS of 0.5 at an MTC length excursion of 4 mm (A) and 8 mm (B). AMPO as a function of FTS, with a fixed 
#|   cycle frequency of 3 Hz at an MTC length excursion of 4 mm (C) and 8 mm (D). Each combination of cycle frequency, FTS 
#|   and MTC length excursion was performed twice. The first trial with suboptimal muscle stimulation duration and the second 
#|   trial with an improved muscle stimulation duration.

#%% Load packages & set directories
import os, sys
import pandas as pd
import numpy as np
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

fig = plt.figure(figsize=(15.92/2.54+0.001, 10.32/2.54), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)
gs = fig.add_gridspec(2,2)
axs = [fig.add_subplot(gs[i]) for i in range(0,gs.ncols*gs.nrows)]

#%%
colorSet = plt.rcParams['axes.prop_cycle'].by_key()['color']
colorSet[0] = '#000000'

lines = []
for iMus,(mus,clr) in enumerate(zip(['GMe1','GMe2','GMe3'],colorSet)):
    dataDirMus = os.path.join(dataDir,mus,'')
    dataExp = pd.read_excel(dataDirMus+str(mus)+'_dataAMPO.xlsx').to_numpy()[:,5:].astype(float)
    
    dataExp[dataExp<1] = np.nan
    dataSSC_PA_T1 = np.nanmean(dataExp[0:3,:],axis=0)
    dataSSC_PA_T2 = np.nanmean(dataExp[3:6,:],axis=0)
    dataSSC_PB_T1 = np.nanmean(dataExp[6:9,:],axis=0)
    dataSSC_PB_T2 = np.nanmean(dataExp[9:12,:],axis=0)
        
    for iExp,exp in enumerate(['SSC_PA', 'SSC_PB']):
        if exp == 'SSC_PA': # 2mm amplitude
            iRow = np.r_[0,1,2]
            
            # CF
            iCol = np.r_[1,2,3,4,5]-1
            # Trial 1
            AMPO = dataSSC_PA_T1[iCol]
            axs[iExp].scatter([1,2,3,4,5],AMPO,color=clr, marker='x',s=20, clip_on=False)
            AMPO = dataSSC_PA_T2[iCol]
            axs[iExp].scatter([1,2,3,4,5],AMPO,color=clr, marker='.',s=40, clip_on=False)
            
            # FTS
            iCol = np.r_[6,7,5,8,9]-1 
            # Trial 1
            AMPO = dataSSC_PA_T1[iCol]
            axs[iExp+2].scatter([0.8,0.65,0.50,0.35,0.20],AMPO,color=clr, marker='x',s=20, clip_on=False)
            # Trial 2
            AMPO = dataSSC_PA_T2[iCol]
            axs[iExp+2].scatter([0.8,0.65,0.50,0.35,0.20],AMPO,color=clr, marker='.',s=40, clip_on=False)
            
        elif exp == 'SSC_PB': # 4mm amplitude
            iRow = np.r_[6,7,8] 
            
            # CF
            iCol = np.r_[1,2,3,4,5]-1
            # Trial 1
            AMPO = dataSSC_PB_T1[iCol]
            l1 = axs[iExp].scatter([1.0,1.5,2.0,2.5,3.0],AMPO,color=clr, marker='x',s=20, clip_on=False)
            AMPO = dataSSC_PB_T2[iCol]
            l2 = axs[iExp].scatter([1.0,1.5,2.0,2.5,3.0],AMPO,color=clr, marker='.',s=40, clip_on=False)
            lines.append(l1)
            
            if iMus == 0:
                l = [l1, l2]
            
            # FTS
            iCol = np.r_[6,7,5,8,9]-1 
            # Trial 1
            AMPO = dataSSC_PB_T1[iCol]
            axs[iExp+2].scatter([0.8,0.65,0.50,0.35,0.20],AMPO,color=clr, marker='x',s=20, clip_on=False)
            # Trial 2
            AMPO = dataSSC_PB_T2[iCol]
            axs[iExp+2].scatter([0.8,0.65,0.50,0.35,0.20],AMPO,color=clr, marker='.',s=40, clip_on=False)
        
legend = axs[1].legend(lines,
                       ['1', '2', '3'],
                       loc='lower right',
                       title='Rat',
                       title_fontproperties={'weight': 'bold'},
                       alignment='right')
legend = axs[3].legend(l,
                       ['1', '2'],
                       loc='lower right',
                       title='Trial',
                       title_fontproperties={'weight': 'bold'},
                       alignment='right')

# #%% Plot
axs[0].set_xlabel('Cycle frequency [Hz]')
axs[1].set_xlabel('Cycle frequency [Hz]')
axs[2].set_xlabel('FTS [ ]')
axs[3].set_xlabel('FTS [ ]')
axs[0].set_ylabel('AMPO [mW]')
axs[2].set_ylabel('AMPO [mW]')

# Subplot 0,0: Imposed CF - 4mm MLE
ax = axs[0]
ax.set_xlim(1,5)
ax.set_xticks([1,2,3,4,5])
# ax.set_ylim(0,97.5)
# ax.set_yticks([0,30,60,90])
# ax.set_yticks([15,45,75], minor=True)
ax.set_ylim(0,130)
ax.set_yticks([0,40,80,120])
ax.set_yticks([20,60,100], minor=True)

# Subplot 0,1: Iposed CF - 8mm MLE
ax = axs[1]
ax.set_xlim(1,3)
ax.set_xticks([1,1.5,2,2.5,3])
# ax.set_ylim(0,97.5)
# ax.set_yticks([0,30,60,90])
# ax.set_yticks([15,45,75], minor=True)
ax.set_ylim(0,130)
ax.set_yticks([0,40,80,120])
ax.set_yticks([20,60,100], minor=True)

# Subplot 1,0: Imposed FTS - 4mm MLE
ax = axs[2]
ax.set_xlim(0.2,0.8)
ax.set_xticks([0.2,0.35,0.5,0.65,0.8])
ax.set_ylim(0,130)
ax.set_yticks([0,40,80,120])
ax.set_yticks([20,60,100], minor=True)

# Subplot 1,1: Imposed FTS - 8mm MLE
ax = axs[3]
ax.set_xlim(0.2,0.8)
ax.set_xticks([0.2,0.35,0.5,0.65,0.8])
ax.set_ylim(0,130)
ax.set_yticks([0,40,80,120])
ax.set_yticks([20,60,100], minor=True)

#%%
for ax in [axs[0], axs[1], axs[2], axs[3]]:
    ax.spines['left'].set_position(('outward', 12))
    
fig.align_ylabels(axs)
cust_fig.add_labels(fig,axs,['A','B','C', 'D'],-15/72)

# %% Show and save
plt.show()
# fig.savefig("r_ampo_cf_fts.png", bbox_inches="tight", pad_inches=0, dpi=600)
fig.savefig("r_ampo_cf_fts.pdf", bbox_inches="tight", pad_inches=0)
# fig.savefig("r_ampo_cf_fts.svg", bbox_inches="tight", pad_inches=0)

# %% Checks
if len(sys.argv) > 1:
    check_size = sys.argv[1]
else:
    check_size = True  

if check_size == True or check_size == 'True':
    cust_fig.report_axes_size(fig,axs)
    # cust_fig.report_fig_size("r_ampo_cf_fts.png") 
    cust_fig.report_fig_size("r_ampo_cf_fts.pdf") 
    # cust_fig.report_fig_size("r_ampo_cf_fts.svg")