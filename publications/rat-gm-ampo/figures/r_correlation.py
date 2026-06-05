#| label: fig-r-correlation
#| fig-cap: Comparison of experimentally measured and predicted AMPO. Measured AMPO was
#|   derived from experimentally observed MTC length and GM force. Predicted AMPO was derived using a Hill
#|   MTC-type model, with experimentally measured MTC length and stimulation over time as inputs to predict
#|   GM force. Each dot represents AMPO of one full cycle where muscle stimulation was present, such that
#|   there are three dots for each experimental SSC condition. The nearly perfect correlation demonstrates that a
#|   Hill-type MTC model can accurately predict the influence of MTC length and stimulation over time on
#|   AMPO.

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
cust_fig.style(plt, fontname='Minion Pro',fontsize=11,grid=True)

fig = plt.figure(figsize=(15.92/3/2.54, 5.07/2.54), constrained_layout=True) # 3:2 ratio
fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)
gs = fig.add_gridspec(1,1)
axs = [fig.add_subplot(gs[i]) for i in range(0,gs.ncols*gs.nrows)]

colorSet = plt.rcParams['axes.prop_cycle'].by_key()['color']
colorSet[0] = '#000000'
symbols = ['o','^', 's']

#%%
for mus,clr,symbol in zip(['GMe1','GMe2','GMe3'],colorSet,symbols):
    dataDirMus = os.path.join(dataDir,mus,'')
    dataExp = pd.read_excel(dataDirMus+str(mus)+'_dataAMPO.xlsx').to_numpy()[:,5:].astype(float)
    simsExp = pd.read_excel(dataDirMus+str(mus)+'_simsAMPO.xlsx').to_numpy()[:,5:].astype(float)
    
    dataExp[dataExp<1] = np.nan
    simsExp[simsExp<1] = np.nan
        
    axs[0].plot(dataExp.flatten(),simsExp.flatten(),symbol,ms=1,color=clr)

legend = axs[0].legend(['1', '2', '3'],
                       title='Rat',
                       title_fontproperties={'weight': 'bold'},
                       loc='lower right',
                       alignment='right',
                       handlelength=0.8,
                       handletextpad=0.5,
                       labelspacing=0.2)

    
#%% Plot
axs[0].set_xlabel(r'Measured AMPO [mW]')
axs[0].set_ylabel(r'Predicted AMPO [mW]')

axs[0].set_xlim(0,162.5)
axs[0].set_xticks([0,25,50,75,100,125,150])
axs[0].set_xticklabels(['0','','50','','100','','150'])
axs[0].set_ylim(0,162.5)
axs[0].set_yticks([0,25,50,75,100,125,150])
axs[0].set_yticklabels(['0','','50','','100','','150'])

# %% Show and save
plt.show()
# fig.savefig("r_correlation.png", bbox_inches="tight", pad_inches=0, dpi=600)
fig.savefig("r_correlation.pdf", bbox_inches="tight", pad_inches=0)
# fig.savefig("r_correlation.svg", bbox_inches="tight", pad_inches=0)

# %% Checks
if len(sys.argv) > 1:
    check_size = sys.argv[1]
else:
    check_size = True  

if check_size == True or check_size == 'True':
    cust_fig.report_axes_size(fig,axs)
    # cust_fig.report_fig_size("r_correlation.png") 
    cust_fig.report_fig_size("r_correlation.pdf") 
    # cust_fig.report_fig_size("r_correlation.svg")