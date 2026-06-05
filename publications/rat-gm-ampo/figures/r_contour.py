#| label: fig-r-contour
#| fig-cap: Predicted maximally attainable AMPO as a function of cycle frequency and MTC
#|   length excursion, shown for four distinct FTS values. The maximally attainable AMPO (in mW),
#|   averaged across three rats, are is depicted as contour lines. The open triangles indicate the location of peak
#|   AMPO at each FTS corresponding to the optimal combination of cycle frequency and MTC length excursion
#|   for each FTS, with the corresponding peak AMPO labelled at the top-left of the triangle

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

import cust_fig, helpers, interpolation, stats

plt.close('all')

#%%
cf_set = np.arange(0.5,6.1,0.5) # [Hz] n = 12
mle_set = np.arange(2,11.1,1)*1e-3 # [m] n = 11

#%%
customlay = cust_fig.style(plt, fontname='Minion Pro',fontsize=11,grid=False)
customlay['xtick.direction'] = 'out' # cause of contour plot
customlay['ytick.direction'] = 'out'
# plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update(customlay)

# fig = plt.figure(figsize=(15.92/2.54+0.085, 15.92/2.54+0.363), constrained_layout=True) # 3:2 ratio
fig = plt.figure(figsize=(15.92/2/2.54+51/600, 8.86/2.54), constrained_layout=True) # 3:2 ratio
gs = fig.add_gridspec(2,2)
# axs = [fig.add_subplot(gs[i]) for i in range(0,gs.ncols*gs.nrows)]
axs = [fig.add_subplot(gs[0, 0])]
axs.append(fig.add_subplot(gs[0, 1], sharex=axs[0]))
axs.append(fig.add_subplot(gs[1, 0], sharex=axs[0]))
axs.append(fig.add_subplot(gs[1, 1], sharex=axs[0]))

for iFts, fts in enumerate([0.25, 0.50, 0.75, 0.85]):
    AMPOsets, AMPOfines, sf = [], [], []
    for iMus,mus in enumerate(['GMe1', 'GMe2', 'GMe3']):
        # Load muspar
        parFile = os.path.join(dataDir,mus,mus+'_IM.pkl')
        muspar = pickle.load(open(parFile, 'rb'))[0]
        # sf.append(muspar['lce_opt']*muspar['fmax'])
        sf.append(1)
                
        # Load and interpolate data
        dataDirSim = os.path.join(dataDir,mus,'simsCV','')
        AMPOset = helpers.load_sims(cf_set,fts,mle_set,mus,dataDirSim)
        AMPOfine,(cfFine,mleFine) = interpolation.do_3d(AMPOset,(cf_set,mle_set),N=100,method='cubic')
           
        # Append to list for all muscles
        AMPOset = AMPOset/sf[iMus]
        AMPOsets.append(AMPOset)
        AMPOfine = AMPOfine/sf[iMus]
        AMPOfines.append(AMPOfine)
    
    #%% Calculate average   
    AMPOsets        = np.dstack(AMPOsets)
    AMPOfines       = np.dstack(AMPOfines)
    meanAMPOsets    = np.mean(AMPOsets,2)
    meanAMPOsets    = meanAMPOsets*np.mean(sf)
    meanAMPOfines   = np.mean(AMPOfines,2)
    meanAMPOfines   = meanAMPOfines*np.mean(sf)
    
    #%%
    iRow, iCol = np.unravel_index(np.nanargmax(meanAMPOsets), meanAMPOsets.shape)
    
    # Compute optimum, but interpolate over smaller interval..
    AMPOsets, AMPOfines, sf = [], [], []
    for iMus,mus in enumerate(['GMe1', 'GMe2', 'GMe3']):
        # Load muspar
        parFile = os.path.join(dataDir,mus,mus+'_IM.pkl')
        muspar = pickle.load(open(parFile, 'rb'))[0]
        sf.append(muspar['lce_opt']*muspar['fmax'])
                
        # Load and interpolate data
        dataDirSim = os.path.join(dataDir,mus,'simsCV','')
        AMPOset = helpers.load_sims(cf_set[iRow-1:iRow+2],fts,mle_set[iCol-1:iCol+2],mus,dataDirSim)
        AMPOfine,(cfFine_s,mleFine_s) = interpolation.do_3d(AMPOset,(cf_set[iRow-1:iRow+2],mle_set[iCol-1:iCol+2]),N=100,method='cubic')
           
        # Append to list for all muscles
        AMPOfine = AMPOfine/sf[iMus]
        AMPOfines.append(AMPOfine)
    AMPOfines       = np.dstack(AMPOfines)
    meanAMPOfines_s   = np.mean(AMPOfines,2)
    meanAMPOfines_s = meanAMPOfines_s*np.mean(sf)
    
    AMPOmax, (cfOpt, mleOpt) = stats.find_max(meanAMPOfines_s,(cfFine_s,mleFine_s))
    
    cfOpt2 = 0
    mleOpt2 = 0
    for iMus,mus in enumerate(['GMe1', 'GMe2', 'GMe3']):
        dataDirSim = os.path.join(dataDir,mus,'simsCV','')

        fileName = mus+f'_cfOpt_fts{fts:{"0.2f"}}_mleOpt'
        df = pd.read_csv(dataDirSim+fileName+'.csv')
        data = df.to_numpy()
        time,lmtc,stim,fsee = data.T[0:4]
        cfOpt2 += (1/time[-1])/3
        mleOpt2 += (lmtc.max()-lmtc.min())/3

    # cfOpt = cfOpt2
    # ampOpt = ampOpt2
    
    # print("AMPO = %1.2f mW" % (AMPOmax*1e3))
    # print("CF = %1.2f Hz" % cfOpt)
    # print("MLE = %1.2f mm" % (mleOpt*1e3))
    
    #%% Make figure 
    if fts == 0.50 or fts == 0.75:
        contour_levels  = np.arange(10,AMPOmax*1e3,20)
    else:
        contour_levels  = np.arange(0,AMPOmax*1e3,20)
    
    cmap = plt.get_cmap('gray_r')
    cmap = cust_fig.truncate_colormap(cmap, 0.25, 1)
    
    iAx = iFts
    CS = axs[iAx].contour(cfFine,mleFine*1e3,meanAMPOfines*1e3, contour_levels,cmap=cmap)
    # axs[iAx].plot(cfOpt,ampOpt*1e3*2,'kx',markersize=4)
    axs[iAx].scatter(cfOpt,mleOpt*1e3, marker='^', facecolors='none', edgecolors='k', s=15)
    axs[iAx].text(cfOpt-5.5*0.02,mleOpt*1e3+9*0.02, f'{AMPOmax*1e3:0.0f}', fontsize='x-small', va='bottom', ha='right')
    
    axs[iAx].set_xlim([0.5, axs[iAx].get_xlim()[1]])
    axs[iAx].set_title(f'FTS = {fts:0.2f}')
    
    a = (11-2)/5.5  # Example: y = x
    b = 2-0.5*a
    
    def line_func(x):
        return a * x + b

    # Function to compute intersection between segment and line
    def segment_line_intersection(p1, p2, a, b):
        x1, y1 = p1
        x2, y2 = p2
        # Represent line segment as p + t*r, intersect with line y = ax + b
        denom = (y2 - y1) - a * (x2 - x1)
        if denom == 0:
            return None  # Parallel
        t = ((a * x1 + b) - y1) / denom
        if 0 <= t <= 1:
            x_int = x1 + t * (x2 - x1)
            y_int = y1 + t * (y2 - y1)
            if np.isclose(y_int, a * x_int + b):
                return x_int, y_int
        return None

    # Find intersection points to use for manual labels
    manual_locations = []

    for i, segs in enumerate(CS.allsegs):
        for seg in segs:
            for j in range(len(seg) - 1):
                p1, p2 = seg[j], seg[j + 1]
                pt = segment_line_intersection(p1, p2, a, b)
                if pt is not None:
                    manual_locations.append(pt)
    # if fts == 0.50:
    #     del(manual_locations[11])
    #     del(manual_locations[10])
    
    
    if manual_locations:
        axs[iAx].clabel(CS, fmt = '%2.0f',fontsize='small', manual=manual_locations)

    # axs[iAx].annotate(f'{AMPOmax*1e3:0.0f}', xy=(cfOpt+5.5*0.02,ampOpt*1e3*2-9*0.02), xycoords='data', xytext=(11, -19), 
                # textcoords='offset points', arrowprops=dict(arrowstyle="->",lw=0.5, connectionstyle="arc3,rad=-.4"), fontsize=9)


#%%    
fig.supxlabel('Cycle frequency [Hz]', fontsize='medium')
fig.supylabel('MTC length excursion [mm]', fontsize='medium')

for ax in axs:
    ax.set_xlim(0.5,6)
    ax.set_ylim(2,11)
    ax.set_xticks([1.0,2.0,3.0,4.0,5.0,6.0])
    ax.set_xticks([1.5,2.5,3.5,4.5,5.5], minor=True)   
    ax.set_yticks([2.0,4.0,6.0,8.0,10])
    ax.set_yticks([3,5,7,9,11], minor=True)

    
fig.align_ylabels(axs)
cust_fig.add_labels(fig,axs,['A','B','C', 'D'])

# %% Show and save
plt.show()
# fig.savefig("r_contour.png", bbox_inches="tight", pad_inches=0, dpi=600)
fig.savefig("r_contour.pdf", bbox_inches="tight", pad_inches=0)
# fig.savefig("r_contour.svg", bbox_inches="tight", pad_inches=0)

# %% Checks
if len(sys.argv) > 1:
    check_size = sys.argv[1]
else:
    check_size = True  

if check_size == True or check_size == 'True':
    cust_fig.report_axes_size(fig,axs)
    # cust_fig.report_fig_size("r_contour.png") 
    cust_fig.report_fig_size("r_contour.pdf") 
    # cust_fig.report_fig_size("r_contour.svg")