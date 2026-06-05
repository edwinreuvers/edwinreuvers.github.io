"""
This script analyses the constant-velocity SSC predictions after stimulation
timing and SSC parameters have been optimised.

Specifically, the following steps were taken:

-   Load the optimised predictions for all specimens.
-   Interpolate AMPO over a finer cycle frequency, FTS and MTC length excursion
    grid.
-   Average the interpolated AMPO landscape across specimens.
-   Find the parameter combination yielding maximal AMPO.
-   Estimate the 95%-of-maximum ranges for cycle frequency, FTS and MTC length
    excursion.

Custom functions used:

-   `helpers.load_sims(cf_set, fts_set, mle_set, mus, data_dir)`
    :   Load simulation files and compute AMPO values for the requested SSC
        parameter grid.
-   `interpolation.do_4d(data, grid, **kwargs)`
    :   Interpolate the AMPO grid onto a finer parameter grid.
-   `stats.find_max(data, grid)`
    :   Find the maximum value in the interpolated AMPO grid and return the
        corresponding SSC parameter values.
"""

#%% Load packages & set directories
import os, sys, pickle
import numpy as np
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import helpers, interpolation, stats

#%% Load and interpolate AMPO grids
cfSet   = [0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5, 5.5, 6]
ftsSet  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
mleSet  = [2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 10e-3, 11e-3, 12e-3]

# Better to use a smaller set around optimum
cfSet   = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
ftsSet  = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
mleSet  = [6e-3, 7e-3, 8e-3, 9e-3, 10e-3, 11e-3]

AMPOsets, AMPOfines, sf = [], [], []
for iMus,mus in enumerate(['GMe1', 'GMe2', 'GMe3']):
    # Load muspar
    parFile = os.path.join(dataDir,mus,mus+'_IM.pkl')
    muspar = pickle.load(open(parFile, 'rb'))[0]
    sf.append(1)
    # sf.append(muspar['lce_opt']*muspar['fmax'])
       
    # Load data   
    dataDir_mus = os.path.join(dataDir,mus,'simsCV')
    AMPOset = helpers.load_sims(cfSet, ftsSet, mleSet,mus,dataDir_mus)
    AMPOfine,(cfFine,ftsFine,mleFine) = interpolation.do_4d(AMPOset,(cfSet,ftsSet,mleSet),N=100,method='linear')    
        
    # Find maximum  
    AMPOset = AMPOset/sf[iMus]
    AMPOfine = AMPOfine/sf[iMus]
    AMPOsets.append(AMPOset)
    AMPOfines.append(AMPOfine)

#%% Average predictions across specimens
AMPOsets = np.stack(AMPOsets, axis=0)
AMPOfines = np.stack(AMPOfines, axis=0)

AMPOmean = np.mean(AMPOfines,0)
AMPOmean = AMPOmean*np.mean(sf)

#%% Find peak AMPO
# Find maximum
AMPOmax, (cfOpt, ftsOpt, mleOpt)  = stats.find_max(AMPOmean,(cfFine,ftsFine, mleFine))

print("AMPO = %1.2f mW" % (AMPOmax*1e3))
print("CF = %1.2f Hz" % cfOpt)
print("FTS = %1.2f" % ftsOpt)
print("MLE = %1.2f mm" % (mleOpt*1e3))

#%% Estimate 95%-of-maximum ranges
iRow, iCol, iDep = np.unravel_index(np.nanargmax(AMPOmean), AMPOmean.shape)

maxAMPO = np.nanmax(AMPOmean)

# 95%
AMPO_CF = AMPOmean[:,iRow,iDep] # @opt cf
idx = np.where(AMPO_CF>0.95*maxAMPO)
cf95 = cfFine[idx]
cfRange = (cf95[0], cf95[-1])
print(cfRange)

AMPO_FTS = AMPOmean[iRow,:,iDep] # @opt fts
idx = np.where(AMPO_FTS>0.95*maxAMPO)
fts95 = ftsFine[idx]
ftsRange = (fts95[0], fts95[-1])
print(ftsRange)

AMPO_AMP = AMPOmean[iRow,iCol,:] # @opt amp
idx = np.where(AMPO_AMP>0.95*maxAMPO)
mle95 = mleFine[idx]
mleRange = (mle95[0]*1e3, mle95[-1]*1e3)
print(mleRange)



