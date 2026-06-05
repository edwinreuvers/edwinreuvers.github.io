#| label: fig-m-conditions
#| fig-cap: Representation of experimentally investigated SSCs. Thirteen combinations of cycle frequency and FTS were 
#|   tested at two distinct MTC length excursions (4 mm and 8 mm), resulting in a total of 26 experimental SSC conditions.

#%% Load packages & set directories
import sys
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

fig = plt.figure(figsize=(15.92/2/2.54+0.084, (15.92/2+1.74)/2.54), constrained_layout=True)
gs = fig.add_gridspec(1, 1)
axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(gs.ncols)] for i in range(gs.nrows)])
ax = axs[0,0]

#%%    
cfCondA  = [1,2,3,4,5,3,3,3,3,5,4,2,1]
cfCondB  = [1,1.5,2,2.5,3,2,2,2,2,3,2.5,1.5,1]
ftsCond = [0.5, 0.5, 0.5, 0.5, 0.5, 0.8, 0.65, 0.35, 0.20, 0.80, 0.65, 0.35, 0.20]

for idx,(cfA,cfB,fts) in enumerate(zip(cfCondA,cfCondB,ftsCond)):
    ax.text(cfA, fts, str(idx+1), bbox=dict(facecolor='white',boxstyle='circle'),ha='center',va='center')
    ax.text(1, 0.5, "1", bbox=dict(facecolor='white',boxstyle='circle'),ha='center',va='center')
    ax.text(2, 0.5, "2", bbox=dict(facecolor='white',boxstyle='circle'),ha='center',va='center')
    ax.text(3, 0.5, "3", bbox=dict(facecolor='white',boxstyle='circle'),ha='center',va='center')
    ax.text(4, 0.5, "4", bbox=dict(facecolor='white',boxstyle='circle'),ha='center',va='center')
    ax.text(5, 0.5, "5", bbox=dict(facecolor='white',boxstyle='circle'),ha='center',va='center')
    ax.text(3, 0.8, "6", bbox=dict(facecolor='white',boxstyle='circle'),ha='center',va='center')
    ax.text(3, 0.65, "7", bbox=dict(facecolor='white',boxstyle='circle'),ha='center',va='center')
    ax.text(3, 0.35, "8", bbox=dict(facecolor='white',boxstyle='circle'),ha='center',va='center')
    ax.text(3, 0.2, "9", bbox=dict(facecolor='white',boxstyle='circle'),ha='center',va='center')
    ax.text(5, 0.8, "10", bbox=dict(facecolor='white',boxstyle='circle'),ha='center',va='center')
    ax.text(4, 0.65, "11", bbox=dict(facecolor='white',boxstyle='circle'),ha='center',va='center')
    ax.text(2, 0.35, "12", bbox=dict(facecolor='white',boxstyle='circle'),ha='center',va='center')
    ax.text(1, 0.2, "13", bbox=dict(facecolor='white',boxstyle='circle'),ha='center',va='center')
 
ax.set_ylabel('FTS [ ]')
ax.set_xlabel('Cycle frequency [Hz] \n SSCs with 4 mm MTC length excursion')

ax.set_xlim(0.5,5.5)
ax.set_xticks([1,2,3,4,5])
ax.set_ylim(0.1,0.9)
ax.set_yticks([0.2,0.35,0.5,0.65,0.8])

axB = ax.twiny()
ax = axB
ax.set_xlim(0.5,5.5)
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(['1','1.5','2','2.5','3'])
ax.set_xlabel('Cycle frequency [Hz] \n SSCs with 8 mm MTC length excursion')
ax.spines['top'].set_visible(True)

# labels = ['A','B']
# cust_fig.add_labels(fig, axs.flatten(), labels)

# %% Show and save
plt.show()
# fig.savefig("m_conditions.png", bbox_inches="tight", pad_inches=0, dpi=600)
fig.savefig("m_conditions.pdf", bbox_inches="tight", pad_inches=0)
# fig.savefig("m_conditions.svg", bbox_inches="tight", pad_inches=0)

# %% Checks
if len(sys.argv) > 1:
    check_size = sys.argv[1]
else:
    check_size = True  

if check_size == True or check_size == 'True':
    cust_fig.report_axes_size(fig,axs)
    # cust_fig.report_fig_size("m_conditions.png") 
    cust_fig.report_fig_size("m_conditions.pdf") 
    # cust_fig.report_fig_size("m_conditions.svg")