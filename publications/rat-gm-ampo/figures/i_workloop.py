#| label: fig-i-workloop
#| fig-cap: An example of a work loop. In a work loop, MTC force is plotted against MTC length (change). The area enclosed 
#|   by the work loop represents the net mechanical work produced during a full cycle (C), which is the sum of the positive 
#|   mechanical work during MTC shortening (A) and the negative mechanical work during MTC lengthening (B). The arrows indicate 
#|   the direction of the work loop over time.

#%% Load packages & set directories
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import cust_fig, hillmodel

plt.close('all')

#%% Load muscle parameters
mus = 'GMe3'
parFile = os.path.join(dataDir,mus,mus+'_IM.pkl')
muspar = pickle.load(open(parFile, 'rb'))[0]
eseerelmax = (muspar['fmax']/muspar['ksee'])**0.5/muspar['lsee0']
lmtcOpt = muspar['lce_opt']+(1+eseerelmax)*muspar['lsee0']
lmtc0 = lmtcOpt+0.5e-3 # [m] MTC-length at t=0

#%% Generate SIN data
cf = 4
amp = 2*1e-3
   
time = np.linspace(0,5/cf,1000)
lmtc = np.cos(time*2*np.pi*cf)*amp+lmtc0

# Simlation to obtain fsee(t)
gamma0 = muspar['gamma_0']
lcerel0 = hillmodel.force_eq(lmtc[0],gamma0,muspar)[1]
c_in = {}
c_in['time'] = time
c_in['lmtc'] = lmtc
c_in['t_stim'] = np.array([[0, 0.3/cf], [1/cf, 1.3/cf], [2/cf, 2.3/cf], [3/cf, 3.3/cf], [4/cf, 4.3/cf]])

solstr = hillmodel.solve_simu_mtc(gamma0,lcerel0,muspar,c_in)[1]
time, lmtc, stim, gamma, lcerel, q, lsee, lpee, fisomrel, fsee, fpee, fce, fcerel, vcerel = solstr

#%%
cust_fig.style(plt, fontname='Minion Pro',fontsize=11,grid='on')

fig = plt.figure(figsize=(15.92/2.54+0.084, (3.34)/2.54), constrained_layout=True)
gs = fig.add_gridspec(1,3)
axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(gs.ncols)] for i in range(gs.nrows)])

iStart = np.argmin(abs(time-4/cf))
iStop = np.argmin(abs(time-5/cf))
time = time[iStart:iStop]-time[iStart]
lmtc = lmtc[iStart:iStop]
fsee = fsee[iStart:iStop]+2
iPeak = signal.find_peaks(-lmtc,distance=200)[0][0]

# W+
lmtcPos = np.hstack((lmtc[0], lmtc[:iPeak], lmtc[iPeak]))
fseePos = np.hstack((0, fsee[:iPeak], 0))
axs[0,0].plot(lmtc[:iPeak],fsee[:iPeak],'k')
axs[0,0].fill(lmtcPos,fseePos,'#bfbfbf')

# W-
lmtcNeg = np.hstack((lmtc[iPeak], lmtc[iPeak:], lmtc[-1]))
fseeNeg = np.hstack((0, fsee[iPeak:], 0))
axs[0,1].plot(lmtc[iPeak:],fsee[iPeak:],'k')
axs[0,1].fill(lmtcNeg,fseeNeg,'#bfbfbf')

# Wnet
axs[0,2].plot(lmtc,fsee,'k')
axs[0,2].fill(lmtc,fsee,'#bfbfbf')

# Text
axs[0,0].text(np.median(lmtc),fsee[50]/2,'Positive work',ha='center',va='center',)
axs[0,1].text(np.median(lmtc),fsee[150]/2-0.21,'Negative work',ha='center',va='center',)
axs[0,2].text(np.median(lmtc),fsee[50]+(fsee[150]-fsee[50])/2,'Net work',ha='center',va='center',)

# Labels
axs[0,0].set_xlabel('MTC length')
axs[0,1].set_xlabel('MTC length')
axs[0,2].set_xlabel('MTC length')
axs[0,0].set_ylabel('MTC force')
# axs[0,1].set_ylabel('Muscle force')
# axs[0,2].set_ylabel('Muscle force')

# Limits & ticks
axs[0,0].set_xticks([])
axs[0,1].set_xticks([])
axs[0,2].set_xticks([])

axs[0,0].set_yticks([0])
axs[0,1].set_yticks([0])
axs[0,2].set_yticks([0])

axs[0,0].tick_params(direction='out', length=0)
axs[0,1].tick_params(direction='out', length=0)
axs[0,2].tick_params(direction='out', length=0)

axs[0,0].set_xlim(lmtc.min()-0.0003,lmtc.max()+0.0003)
axs[0,1].set_xlim(lmtc.min()-0.0003,lmtc.max()+0.0003)
axs[0,2].set_xlim(lmtc.min()-0.0003,lmtc.max()+0.0003)

axs[0,0].set_ylim(0,10)
axs[0,1].set_ylim(0,10)
axs[0,2].set_ylim(0,10)

# x1, x2 = lmtc[60], lmtc[40]
# y1, y2 = fsee[60]+1, fsee[40]+1
# axs[0].annotate("", xy=(x1, y1), xytext=(x2, y2),arrowprops=dict(arrowstyle="->"))

# x1, x2 = lmtc[140], lmtc[160]
# y1, y2 = fsee[140]+1, fsee[160]+1
# axs[1].annotate("", xy=(x1, y1), xytext=(x2, y2),arrowprops=dict(arrowstyle="<-"))

for i in [25, 50, 75]:
    x1, x2 = lmtc[i], lmtc[i+1]
    y1, y2 = fsee[i], fsee[i+1]
    axs[0,0].annotate("", xy=(x1, y1), xytext=(x2, y2),arrowprops=dict(arrowstyle="<-"))

for i in [125, 150, 175]:
    x1, x2 = lmtc[i], lmtc[i+1]
    y1, y2 = fsee[i], fsee[i+1]
    axs[0,1].annotate("", xy=(x1, y1), xytext=(x2, y2),arrowprops=dict(arrowstyle="<-"))

for i in [25,50,75, 125,150,175]:
    x1, x2 = lmtc[i], lmtc[i+1]
    y1, y2 = fsee[i], fsee[i+1]
    axs[0,2].annotate("", xy=(x1, y1), xytext=(x2, y2),arrowprops=dict(arrowstyle="<-"))


# Add the '-' sign between Plot 1 and Plot 2
# axs[0].text(1.1, 0.5, '–', fontsize=30, va='center', ha='center', transform=axs[0].transAxes, weight='bold')

# Add the '=' sign between Plot 2 and Plot 3
# axs[1].text(1.1, 0.5, '=', fontsize=30, va='center', ha='center', transform=axs[1].transAxes)

labels = ['A','B','C']
cust_fig.add_labels(fig, axs.flatten(), labels)

# %% Show and save
plt.show()
# fig.savefig("i_workloop.png", bbox_inches="tight", pad_inches=0, dpi=600)
fig.savefig("i_workloop.pdf", bbox_inches="tight", pad_inches=0)
# fig.savefig("i_workloop.svg", bbox_inches="tight", pad_inches=0)

# %% Checks
if len(sys.argv) > 1:
    check_size = sys.argv[1]
else:
    check_size = True  

if check_size == True or check_size == 'True':
    cust_fig.report_axes_size(fig,axs)
    # cust_fig.report_fig_size("i_workloop.png") 
    cust_fig.report_fig_size("i_workloop.pdf") 
    # cust_fig.report_fig_size("i_workloop.svg")
