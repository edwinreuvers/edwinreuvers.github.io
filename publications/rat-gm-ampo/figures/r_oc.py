#| label: fig-r-oc
#| fig-cap: Predicted CE behaviour for maximally attainable AMPO, shown at three distinct
#|   cycle frequencies for rat 1. CE length (A), CE force (B), CE velocity (C) and instantaneous mechanical
#|   power output (IMPO) (D) as a function of normalised time (i.e. time divided by the cycle duration). CE
#|   stimulation was maximal during the period indicated by the coloured bars and fully off elsewhere. The right
#|   labels depict the normalised values, where CE length and CE velocity are normalised to optimum CE length,
#|   CE force is normalised to maximal isometric CE force and IMPO is normalised to the product of optimum
#|   CE length and maximal isometric CE force. Based on CE length and force over time, and the SEE
#|   properties, the MTC length over time can be calculated. The resulting MTC length over time substantially
#|   differs between a typical SEE slack length of 28 mm (E) and a short SEE slack length of 3 mm (F).

#%% Load packages & set directories
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

import cust_fig, hillmodel

plt.close('all')

#%%
cust_fig.style(plt, fontname='Minion Pro',fontsize=11,grid=False)
# fig = plt.figure(figsize=(15.92/2.54+0.085, 13.36/2.54), constrained_layout=True) # 3:2 ratio
fig = plt.figure(figsize=(15.92/2.54+0.085, 15.07/2.54), constrained_layout=True) # 3:2 ratio
# fig = plt.figure(figsize=(15.92/2.54+0.085, 15.92/2.54/3+0.085), constrained_layout=True) # 3:1 ratio
gs = fig.add_gridspec(6, 2, height_ratios=[1e-3,1.02,8.4,8.4,1e-3,8.4], wspace=0)
axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(gs.ncols)] for i in range(gs.nrows)])
axs = axs.flatten().tolist()

# These are 'fake axis' to create title above the two 'blocks'
axs[9].remove(); del(axs[9])
axs[8].remove(); del(axs[8])
axs[1].remove(); del(axs[1])
axs[0].remove(); del(axs[0])

#%% Subplot 1,1
mus = 'GMe1'
parFile = os.path.join(dataDir,mus,mus+'_IM.pkl')
muspar = pickle.load(open(parFile, 'rb'))[0]
musparLong  = muspar.copy()
musparShort = muspar.copy()
muspar_lsee0_new = 3e-3
musparShort['ksee'] = musparShort['ksee']*(musparShort['lsee0']/muspar_lsee0_new)**2
musparShort['lsee0'] = muspar_lsee0_new
dataDirSim = os.path.join(dataDir,mus,'simsOC','')
fmax, lce_opt = muspar['fmax'], muspar['lce_opt']

cfSet = [2.5, 3.5, 4.5]
cfSet = [2.0, 3.5, 5]

colorSet = plt.rcParams['axes.prop_cycle'].by_key()['color']
colorSet[0] = colorSet[1]
colorSet[1] = '#000000'
# colorSet[1] = '#1f77b4'
for idx,cf in enumerate(cfSet):
    fileName = mus+f'_cf{cf:{"0.1f"}}Hz_ftsOpt_mleOpt'  
    df = pd.read_csv(dataDirSim+fileName+'.csv', sep=',')
    data = df.to_numpy()
    time,lmtc,stim,fsee,gamma,lcerel = data.T[0:6]
    
    lce = lcerel*lce_opt
    vce = np.gradient(lce,time)
       
    time = time/time[-1]
    time = np.concatenate((time-time[-1],time,time+time[-1]))
    lce = np.concatenate((lce,lce,lce))
    fsee = np.concatenate((fsee,fsee,fsee))
    vce = np.concatenate((vce,vce,vce))
    stim = np.concatenate((stim,stim,stim))
    lpee = lce
    fpee = hillmodel.lee2force(0,lpee,muspar)[1]
    fce = fsee-fpee
    IMPO = -fce*vce
    
    stim[stim>=0.5] = 1.0; 
    stim[stim<0.5] = 0
    tStimOn = time[np.where(np.diff(stim)>0.1)[0]]
    tStimOff = time[np.where(np.diff(stim)<-0.1)[0]]
    cust_fig.plot_stim(axs[0],tStimOn[0],tStimOff[1],y=-idx, lw=1/3, color=colorSet[idx])
    cust_fig.plot_stim(axs[0],tStimOn[1],tStimOff[2],y=-idx, lw=1/3 ,color=colorSet[idx])
    
    # Lmtc(t): Long SEE
    lseeLong = hillmodel.force2lee(fsee,fpee,musparLong)[0]
    lmtcLong = lce + lseeLong

    # Lmtc(t): Short SEE
    lseeShort = hillmodel.force2lee(fsee,fpee,musparShort)[0]
    lmtcShort = lce + lseeShort
    
    # plotStim(axs[1],tStimOn[0],tStimOff[1],y=-idx, lw=1/3, color=colorSet[idx])
    # plotStim(axs[1],tStimOn[1],tStimOff[2],y=-idx, lw=1/3 ,color=colorSet[idx])
    
    axs[1].plot(time,stim, color=colorSet[idx]) # plotted but not visible!
    
    if tStimOff[-1] > 1.75: 
        print('Warning, tStimoff exceed plotted values!!')
    
    axs[2].plot(time,lce*1e3, color=colorSet[idx])
    axs[3].plot(time,fce, color=colorSet[idx])
    axs[4].plot(time,vce*1e3, color=colorSet[idx])
    axs[5].plot(time,IMPO*1e3, color=colorSet[idx])
    
    axs[6].plot(time,lmtcLong*1e3, color=colorSet[idx])
    axs[7].plot(time,lmtcShort*1e3, color=colorSet[idx])
  
legend = axs[1].legend(["2.0 Hz", "3.5 Hz", "5.0 Hz"],
                       loc='center',
                       ncol=3,
                       bbox_to_anchor=(0.5, 0.5),  # slight offset above the axes
                       handlelength=0.8,
                       handletextpad=0.5,
                       labelspacing=0.2)


for ax in [axs[0], axs[1], axs[2], axs[3], axs[4], axs[5], axs[6], axs[7]]:
    ax.set_xlim(-0.25,1.75)

# stim(t)
# axs[0].autoscale(enable=True, axis='y', tight=True)
axs[0].set_ylim(-7/3,1/3)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].spines['bottom'].set_visible(False)
axs[0].spines['left'].set_visible(False)
axs[0].set_xticks([]);
axs[0].set_yticks([]); 
axs[1].autoscale(enable=True, axis='y', tight=True)
axs[1].set_ylim(500,900) # range way outside what is plotted, only for legend purpose!
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['bottom'].set_visible(False)
axs[1].spines['left'].set_visible(False)
axs[1].set_xticks([]);
axs[1].set_yticks([]); 

# lce(t)    
axs[2].set_ylim(7,20)
axs[2].set_yticks([10,14,18])
axs[2].set_yticks([12,16],minor=True)
axs[2].set_ylabel('$L_{CE}$ [mm]')
ax2T = axs[2].twinx()
ax2T.spines['right'].set_visible(True)
ax2T.plot(time,lce/lce_opt, linewidth=1, linestyle ='--', color='k', alpha=0)
ax2T.set_ylim((axs[2].get_ylim()[0]/1e3/lce_opt, axs[2].get_ylim()[1]/1e3/lce_opt)) 
ax2T.set_yticks([0.6,1.0,1.4])
ax2T.set_yticks([0.8,1.2],minor=True)
ax2T.set_ylabel('$L_{CE}^{rel}$ [ ]')
 
# fsee(t)
ax = axs[3]
ax.set_ylim(0,11)
ax.set_yticks([0,4,8])
ax.set_yticks([2,6],minor=True)
ax.set_ylabel('$F_{CE}$ [N]')
axFsee = ax.twinx()
axFsee.spines['right'].set_visible(True)
axFsee.plot(time,fce/fmax, linewidth=1, linestyle ='--', color='k', alpha=0)
axFsee.set_ylim((ax.get_ylim()[0]/fmax, ax.get_ylim()[1]/fmax)) 
axFsee.set_yticks([0,0.2,0.4,0.6])
axFsee.set_yticks([0.1,0.3,0.5,0.7],minor=True)
axFsee.set_ylabel('$F_{CE}^{rel}$ [ ]')

# vce(t)
ax = axs[4]
ax.set_ylim(-75,270)
ax.set_yticks([0,100,200])
ax.set_yticks([-50,50,150,250],minor=True)
ax.set_ylabel('$V_{CE}$ [mm/s]')
axVce = ax.twinx()
axVce.spines['right'].set_visible(True)
axVce.plot(time,vce/lce_opt, linewidth=1, linestyle ='--', color='k', alpha=0)
axVce.set_ylim((ax.get_ylim()[0]/1e3/lce_opt, ax.get_ylim()[1]/1e3/lce_opt)) 
axVce.set_yticks([0,8,16])
axVce.set_yticks([-4,4,12],minor=True)
axVce.set_ylabel('$V_{CE}^{rel}$ [1/s]')

# IMPO(t)
ax = axs[5]
ax.set_ylim(-350,350)
ax.set_yticks([-200,0,200])
ax.set_yticks([-300,-100,0,100,300],minor=True)
ax.set_ylabel('IMPO [mW]')
axImpo = ax.twinx()
axImpo.spines['right'].set_visible(True)
axImpo.plot(time,IMPO/(lce_opt*fmax), linewidth=1, linestyle ='--', color='k', alpha=0)
axImpo.set_ylim((ax.get_ylim()[0]/1e3/(lce_opt*fmax), ax.get_ylim()[1]/1e3/(lce_opt*fmax))) 
axImpo.set_yticks([-1,0,1])
axImpo.set_yticks([-1.5,-0.5,0.5,1.5],minor=True)
axImpo.set_ylabel('rel IMPO [1/s]')

# Lmtc Long (t)
ax = axs[6]
ax.set_title('Typical SEE slack length (28 mm)', loc="left")
ax.set_ylim(38,51)
ax.set_yticks([40,44,48])
ax.set_yticks([42,46,50],minor=True)
ax.set_ylabel('MTC length [mm]')

# Lmtc Long (t)
ax = axs[7]
ax.set_title('Short SEE slack length (3 mm)', loc="left")
ax.set_ylim(10,23)
ax.set_yticks([12,16,20])
ax.set_yticks([10,14,18,22],minor=True)
ax.set_ylabel('MTC length [mm]')

axs[2].plot([-2,2],[lce_opt*1e3,lce_opt*1e3],'k--',alpha=0.25,lw=0.5,zorder=-100)
axs[4].plot([-2,2],[0,0],'k--',alpha=0.25,lw=0.5,zorder=-100)
axs[5].plot([-2,2],[0,0],'k--',alpha=0.25,lw=0.5,zorder=-100)

# axs[4].set_xlabel('Normalised time [s]')
# axs[5].set_xlabel('Normalised time [s]')
axs[6].set_xlabel('Normalised time [s]')
axs[7].set_xlabel('Normalised time [s]')

axsTitle = fig.add_subplot(gs[0, :])
axsTitle.set_title('CE Dynamics', fontweight='bold', fontsize='large')
axsTitle.set_frame_on(False)
axsTitle.axis('off')

axsTitle = fig.add_subplot(gs[4, :])
axsTitle.set_title('MTC Dynamics', fontweight='bold', fontsize='large')
axsTitle.set_frame_on(False)
axsTitle.axis('off')

fig.align_labels()
cust_fig.add_labels(fig, axs, ['', '', 'A', 'B', 'C', 'D', 'E', 'F'])

# %% Show and save
plt.show()
# fig.savefig("r_oc.png", bbox_inches="tight", pad_inches=0, dpi=600)
fig.savefig("r_oc.pdf", bbox_inches="tight", pad_inches=0)
# fig.savefig("r_oc.svg", bbox_inches="tight", pad_inches=0)

# %% Checks
if len(sys.argv) > 1:
    check_size = sys.argv[1]
else:
    check_size = True  

if check_size == True or check_size == 'True':
    cust_fig.report_axes_size(fig,axs)
    # cust_fig.report_fig_size("r_oc.png") 
    cust_fig.report_fig_size("r_oc.pdf") 
    # cust_fig.report_fig_size("r_oc.svg")