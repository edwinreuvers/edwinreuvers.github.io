#| label: suppfig-s-prelim
#| fig-cap: Preliminary predictions of maximal AMPO as a function of cycle frequency and FTS. 
#|   The contour lines depict the maximally attainable AMPO (in mW) averaged over three parameter 
#|   sets. These preliminary predictions were derived with a Hill-type MTC model with parameters 
#|   obtained from literature [@van_zandwijk_twitch_1996] and were used to inform the selection 
#|   of experimental SSC conditions.

#%%
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import cust_fig, helpers, interpolation

plt.close('all')

#%% Make figure
customlay = cust_fig.style(plt, fontname='Minion Pro',fontsize=11,grid=False)
customlay['xtick.direction'] = 'out' # cause of contour plot
customlay['ytick.direction'] = 'out'
# plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update(customlay)

fig = plt.figure(figsize=(15.92/2.54+51/600, (15.92/2-1.6)/2.54), constrained_layout=True) # 3:2 ratio
gs = fig.add_gridspec(1,2)
# axs = [fig.add_subplot(gs[i]) for i in range(0,gs.ncols*gs.nrows)]
axs = [fig.add_subplot(gs[0, 0])]
axs.append(fig.add_subplot(gs[0, 1]))

#%% Loop
ftsSet = np.arange(0.05, 0.96, 0.05)
for iAmp, amp in enumerate([2e-3 , 4e-3]):
    if amp == 2e-3:
        cfSet = np.arange(0.4, 6.1, 0.2)
    elif amp == 4e-3:
        cfSet = np.arange(0.4, 4.1, 0.2)
    else:
        breakpoint()
    AMPOsets, AMPOfines, sf = [], [], []
    for iMus,mus in enumerate(['GMz1', 'GMz2', 'GMz3']):
        # Load muspar
        # parFile = os.path.join(dataDir,mus,mus+'_IM.pkl')
        # muspar = pickle.load(open(parFile, 'rb'))[0]
        # sf.append(muspar['lce_opt']*muspar['fmax'])
        sf.append(1)
    
        filepaths = [[os.path.join(dataDir,'prelim', mus, f'{mus}_amp{amp*1e3:0.1f}mm_cf{cf:0.1f}Hz_fts{fts:0.2f}.csv') for fts in ftsSet] for cf in cfSet] 
        AMPOset = helpers.get_ampo(filepaths)
        AMPOfine,(cfFine,ftsFine) = interpolation.do_3d(AMPOset,(cfSet,ftsSet),N=19,method='cubic')
           
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
    
    #%% Make figure 
    AMPOmax = np.nanmax(AMPOfine)
    contour_levels  = np.arange(10,AMPOmax*1e3,10)
    
    cmap = plt.get_cmap('gray_r')
    cmap = cust_fig.truncate_colormap(cmap, 0.25, 1)
    
    iAx = iAmp
    CS = axs[iAx].contour(cfFine,ftsFine,AMPOfine*1e3,contour_levels,cmap=cmap)
    
    x1, x2 = 0.5, [6,4][iAmp]
    y1, y2 = 0.05, 0.95
    
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    
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
        axs[iAx].clabel(CS, fmt = '%2.0f',fontsize=9, manual=manual_locations)
    
#%%    
fig.supxlabel('Cycle frequency [Hz]', fontsize=11)
fig.supylabel('FTS [ ]', fontsize=11)

axs[0].set_title('MTC length excursion = 4 mm')
axs[1].set_title('MTC length excursion = 8 mm')

for ax in axs:
    ax.set_ylim(0.05,0.95)
    ax.set_xticks([1.0,2.0,3.0,4.0,5.0,6.0])
    ax.set_xticks([1.5,2.5,3.5,4.5,5.5], minor=True)   
    # ax.set_yticks([2.0,4.0,6.0,8.0,10])
    # ax.set_yticks([3,5,7,9,11], minor=True)

axs[0].set_xlim(0.5,6)
axs[1].set_xlim(0.5,4) 

cust_fig.add_labels(fig, axs, ['A','B'])

# %% Show and save
plt.show()
# fig.savefig("s_prelim.png", bbox_inches="tight", pad_inches=0, dpi=600)
fig.savefig("s_prelim.pdf", bbox_inches="tight", pad_inches=0)
# fig.savefig("s_prelim.svg", bbox_inches="tight", pad_inches=0)

# %% Checks
if len(sys.argv) > 1:
    check_size = sys.argv[1]
else:
    check_size = True  

if check_size == True or check_size == 'True':
    cust_fig.report_axes_size(fig,axs)
    # cust_fig.report_fig_size("s_prelim.png") 
    cust_fig.report_fig_size("s_prelim.pdf") 
    # cust_fig.report_fig_size("s_prelim.svg")
