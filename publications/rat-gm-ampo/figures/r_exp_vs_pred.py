#| label: fig-r-exp-vs-pred
#| fig-cap: Comparison of experimentally measured and predicted GM force over time. Predicted
#|   GM force over time was derived using a Hill MTC-type model, with experimentally measured MTC length
#|   and stimulation over time as inputs. Predicted SEE force closely matched experimentally measured GM force
#|   during activation but was slightly higher than experimentally measured SEE force in most conditions. Top:
#|   MTC length over time. Second: Muscle stimulation over time, with maximal stimulation during the periods
#|   indicated by the black bars and no stimulation elsewhere. Third: SEE force over time. Bottom: SEE force as
#|   a function of MTC length (‘the work loop’). A) SSC with a cycle frequency of 3 Hz, a FTS of 0.5 and an
#|   MTC length excursion of 4 mm. B) SSC with a cycle frequency of 2 Hz, a FTS of 0.5 and an MTC length
#|   excursion of 8 mm.

#%% Load packages & set directories
import os, glob, sys, pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
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

#%%
cust_fig.style(plt, fontname='Minion Pro',fontsize=11,grid=False)
fig = plt.figure(figsize=(15.92/2.54+51/600, 10.9/2.54), constrained_layout=True) # 3:2 ratio
gs = fig.add_gridspec(4, 2, height_ratios=[5.0,1,35,35],wspace=0.1)
axs = [fig.add_subplot(gs[i]) for i in range(0,gs.ncols*gs.nrows)]

colorSet = plt.rcParams['axes.prop_cycle'].by_key()['color']
colorSet[0] = '#000000'

#%% SSC: CF = 3 Hz, FTS = 0.5 Hz, MLE = 4mm
iCond = 5
fileExp = sorted(glob.glob(os.path.join(dataDir,mus,'dataExp','SSC_PA','*.csv')))[iCond]
fileSim = sorted(glob.glob(os.path.join(dataDir,mus,'simsExp','SSC_PA','*.csv')))[iCond]

if fileExp[-19:-4] != fileSim[-22:-7]:
    print(fileExp[-19:-4])
    print(fileSim[-22:-7])
    print('Datafile and simulation not the same!')
    sys.exit()
if fileExp[-19:-4] != 'GMe1_SSC_PA03_2': 
    print('Wrong files loaded..')
    sys.exit()

# Experimental data
df = pd.read_csv(fileExp)
data = df.to_numpy()
time,lmtc,stim,fsee = data.T[0:4]

iMin = signal.find_peaks(-lmtc[150:],distance=200)[0]+150
iMax = signal.find_peaks(lmtc[150:],distance=200)[0]+150
iSel = slice(iMin[0],iMax[3])
tOff = time[iMin[0]]
time = time[iSel]-tOff
lmtc = lmtc[iSel]
fsee = fsee[iSel]
stim = stim[iSel]

axs[0].plot(time,lmtc*1e3, color=colorSet[0])
axs[4].plot(time,fsee, color=colorSet[0])
axs[6].plot(lmtc*1e3,fsee, color=colorSet[0])

# Simulated data
df = pd.read_csv(fileSim)
data = df.to_numpy()
time,lmtc,stim,fsee = data.T[0:4]

tOff = time[iMin[0]]
time = time[iSel]-tOff
lmtc = lmtc[iSel]
fsee = fsee[iSel]
stim = stim[iSel]

# axs[0].plot(time,lmtc*1e3,'--', color=colorSet[1])
axs[4].plot(time,fsee,'--', color=colorSet[1])
axs[6].plot(lmtc*1e3,fsee, '--', color=colorSet[1])

tStimOn, tStimOff = stimulation.get_stim_timing(time,stim)
cust_fig.plot_stim(axs[2],tStimOn[0],tStimOff[0],y=0, lw=1/3)
cust_fig.plot_stim(axs[2],tStimOn[1],tStimOff[1],y=0, lw=1/3)
cust_fig.plot_stim(axs[2],tStimOn[2],tStimOff[2],y=0, lw=1/3)

for ax in [axs[0], axs[2], axs[4]]:
    ax.set_xlim(time[0],time[-1])

#%% SSC: CF = 2 Hz, FTS = 0.5 Hz, MLE = 8mm
iCond = 6
fileExp = sorted(glob.glob(os.path.join(dataDir,mus,'dataExp','SSC_PB','*.csv')))[iCond]
fileSim = sorted(glob.glob(os.path.join(dataDir,mus,'simsExp','SSC_PB','*.csv')))[iCond]

if fileExp[-19:-4] != fileSim[-22:-7]:
    print(fileExp[-19:-4])
    print(fileSim[-22:-7])
    print('Datafile and simulation not the same!')
    sys.exit()
if fileExp[-19:-4] != 'GMe1_SSC_PB03_2': 
    print('Wrong files loaded..')
    sys.exit()

# Experimental data
df = pd.read_csv(fileExp)
data = df.to_numpy()
time,lmtc,stim,fsee = data.T[0:4]

iMin = signal.find_peaks(-lmtc[150:],distance=200)[0]+150
iMax = signal.find_peaks(lmtc[150:],distance=200)[0]+150
iSel = slice(iMin[0],iMax[3])
tOff = time[iMin[0]]
time = time[iSel]-tOff
lmtc = lmtc[iSel]
fsee = fsee[iSel]
stim = stim[iSel]

axs[1].plot(time,lmtc*1e3, color=colorSet[0])
l1, = axs[5].plot(time,fsee, color=colorSet[0], label='Measured')
axs[7].plot(lmtc*1e3,fsee, color=colorSet[0])

# Simulated data
df = pd.read_csv(fileSim)
data = df.to_numpy()
time,lmtc,stim,fsee = data.T[0:4]

tOff = time[iMin[0]]
time = time[iSel]-tOff
lmtc = lmtc[iSel]
fsee = fsee[iSel]
stim = stim[iSel]

# axs[1].plot(time,lmtc*1e3,'--', color=colorSet[1])
l2, = axs[5].plot(time,fsee,'--', color=colorSet[1], label='Predicted')
axs[7].plot(lmtc*1e3,fsee,'--', color=colorSet[1])

tStimOn, tStimOff = stimulation.get_stim_timing(time,stim)
cust_fig.plot_stim(axs[3],tStimOn[0],tStimOff[0],y=0, lw=1/3)
cust_fig.plot_stim(axs[3],tStimOn[1],tStimOff[1],y=0, lw=1/3)
cust_fig.plot_stim(axs[3],tStimOn[2],tStimOff[2],y=0, lw=1/3)

for ax in [axs[1], axs[3], axs[5]]:
    ax.set_xlim(time[0],time[-1])

axs[5].legend(['Measured', 'Predicted'],
    loc='center',
    bbox_to_anchor=(0.525, 0.5),  # shift left (into space between axes) and center vertically
    bbox_transform=fig.transFigure,
    handlelength=0.8,
    handletextpad=0.5,
    labelspacing=0.2,
)

#%% Labels etc.
# Lmtc(t) - SSC_PA
ax = axs[0]
ax.set_ylim(axs[1].get_ylim())
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
y_min, y_max = 41.5, 45.5
ax.set_ylim(y_min, y_max)
y_range = (y_max-y_min)
ax.plot([0, 0], [y_min, y_max], color='black', lw=1)
ax.set_yticks([(y_min+y_max)/2])
ax.set_yticklabels([f'{y_range:.1f} mm'])
ax.tick_params(
    axis='both',       # both x and y axes
    which='both',      # both major and minor ticks
    bottom=False,      # remove ticks on bottom
    top=False,         # remove ticks on top
    left=False,        # remove ticks on left
    right=False        # remove ticks on right
)

# Lmtc(t) - SSC_PB
ax = axs[1]
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
y_min, y_max = 39.5, 47.5
ax.set_ylim(y_min, y_max)
y_range = (y_max-y_min)
ax.plot([0, 0], [y_min, y_max], color='black', lw=1)
ax.set_yticks([(y_min+y_max)/2])
ax.set_yticklabels([f'{y_range:.1f} mm'])
ax.tick_params(
    axis='both',       # both x and y axes
    which='both',      # both major and minor ticks
    bottom=False,      # remove ticks on bottom
    top=False,         # remove ticks on top
    left=False,        # remove ticks on left
    right=False        # remove ticks on right
)



# STIM(t) - SSC_PA
ax = axs[2]
ax.set_ylim(-0.5,0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# STIM(t) - SSC_PB
ax = axs[3]
ax.set_ylim(-0.5,0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Fsee(t) - SSC_PA
ax = axs[4]
ax.set_xticks([0,0.4,0.8])
ax.set_xticks([0.2,0.6,1.0],minor=True)
ax.set_ylim(0,9.5)
ax.set_yticks([0,4,8])
ax.set_yticks([2,6],minor=True)
ax.set_ylabel('$F_{SEE}$ [N]')

# Fsee(t) - SSC_PB
ax = axs[5]
ax.set_xticks([0,0.5,1.0,1.5])
ax.set_xticks([0.25,0.75,1.25],minor=True)
ax.set_ylim(0,9.5)
ax.set_yticks([0,4,8])
ax.set_yticks([2,6],minor=True)
# ax.set_ylabel('$F_{SEE}$ [N]')

axs[4].set_xlabel('Time [s]')
axs[5].set_xlabel('Time [s]')

# Fsee(Lmtc) - SSC_PA
ax = axs[6]
ax.set_xlim(41.25,45.75)
ax.set_xticks([42,43,44,45])
ax.set_xticks([41.5,42.5,43.5,44.5,45.5],minor=True)
ax.set_xlabel('$L_{MTC}$ [mm]')
ax.set_ylim(0,9.5)
ax.set_yticks([0,4,8])
ax.set_yticks([2,6],minor=True)
ax.set_ylabel('$F_{SEE}$ [N]')

# Fsee(Lmtc) - SSC_PB
ax = axs[7]
ax.set_xlim(39.25,47.5)
ax.set_xticks([40,42,44,46])
ax.set_xticks([41,43,45,47],minor=True)
ax.set_xlabel('$L_{MTC}$ [mm]')
ax.set_ylim(0,9.5)
ax.set_yticks([0,4,8])
ax.set_yticks([2,6],minor=True)

# fig.align_labels()
cust_fig.add_labels(fig, axs, ['A', 'B'])

# %% Show and save
plt.show()
# fig.savefig("r_exp_vs_pred2.png", bbox_inches="tight", pad_inches=0, dpi=600)
fig.savefig("r_exp_vs_pred.pdf", bbox_inches="tight", pad_inches=0)
# fig.savefig("r_exp_vs_pred2.svg", bbox_inches="tight", pad_inches=0)

# %% Checks
if len(sys.argv) > 1:
    check_size = sys.argv[1]
else:
    check_size = True  

if check_size == True or check_size == 'True':
    cust_fig.report_axes_size(fig,axs)
    # cust_fig.report_fig_size("r_exp_vs_pred2.png") 
    cust_fig.report_fig_size("r_exp_vs_pred.pdf") 
    # cust_fig.report_fig_size("r_exp_vs_pred2.svg")