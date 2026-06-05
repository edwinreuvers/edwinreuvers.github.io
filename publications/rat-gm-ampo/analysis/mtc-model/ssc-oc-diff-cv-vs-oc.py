"""
This script quantifies how much AMPO increases when MTC length over time is
optimised freely, compared with SSCs that impose constant MTC shortening and
lengthening velocities.

For each specimen, AMPO is loaded for matching constant-velocity and
optimal-control simulations. Ratios are computed separately for sweeps with
imposed cycle frequency, imposed FTS and imposed MTC length excursion. The
printed values report the mean percentage increase of the optimal-control
solutions over the constant-velocity solutions.
"""

#%% Load packages & set directories
import os, sys
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import helpers

plt.close('all')
	    
#%% Compare simulations with imposed cycle frequency
cfSet = np.arange(1.0,6.1,0.5) # Hz

d = []
for mus in ['GMe1', 'GMe2', 'GMe3']:
    dataFolder = os.path.join(dataDir,mus,'simsCV','')
    filepaths_cf    = [os.path.join(dataFolder, f'{mus}_cf{cf:0.1f}Hz_ftsOpt_mleOpt.csv') for cf in cfSet]
    AMPOssc = helpers.get_ampo(filepaths_cf)
    
    dataFolder = os.path.join(dataDir,mus,'simsOC','')
    filepaths_cf    = [os.path.join(dataFolder, f'{mus}_cf{cf:0.1f}Hz_ftsOpt_mleOpt.csv') for cf in cfSet]
    AMPOoc = helpers.get_ampo(filepaths_cf)

    d.append(AMPOoc/AMPOssc)
avg,std = (np.nanmean(d)-1)*100, np.nanstd(d)*100
print(f'For imposed cycle frequency OC is {avg:0.1f} +- {std:0.1f}% higher')

#%% Compare simulations with imposed FTS
ftsSet = np.arange(0.25,0.96,0.05)

d = []
for mus in ['GMe1', 'GMe2', 'GMe3']:
    dataFolder = os.path.join(dataDir,mus,'simsCV','')
    filepaths_fts   = [os.path.join(dataFolder, f'{mus}_cfOpt_fts{fts:0.2f}_mleOpt.csv') for fts in ftsSet]
    AMPOssc = helpers.get_ampo(filepaths_fts)
    
    dataFolder = os.path.join(dataDir,mus,'simsOC','')
    filepaths_fts   = [os.path.join(dataFolder, f'{mus}_cfOpt_fts{fts:0.2f}_mleOpt.csv') for fts in ftsSet]
    AMPOoc = helpers.get_ampo(filepaths_fts)

    d.append(AMPOoc/AMPOssc)
avg,std = (np.nanmean(d)-1)*100, np.nanstd(d)*100
print(f'For imposed FTS OC is {avg:0.1f} +- {std:0.1f}% higher')

#%% Compare simulations with imposed MTC length excursion
mleSet = np.arange(1,12.1,1) # mm

d = []
for mus in ['GMe1', 'GMe2', 'GMe3']:
    dataFolder = os.path.join(dataDir,mus,'simsCV','')
    filepaths_mle   = [os.path.join(dataFolder, f'{mus}_cfOpt_ftsOpt_mle{mle:0.1f}mm.csv') for mle in mleSet]
    AMPOssc = helpers.get_ampo(filepaths_mle)
    
    dataFolder = os.path.join(dataDir,mus,'simsOC','')
    filepaths_mle   = [os.path.join(dataFolder, f'{mus}_cfOpt_ftsOpt_mle{mle:04.1f}mm.csv') for mle in mleSet]
    AMPOoc = helpers.get_ampo(filepaths_mle)

    d.append(AMPOoc/AMPOssc)
avg,std = (np.nanmean(d)-1)*100, np.nanstd(d)*100
print(f'For imposed MTC length excursion OC is {avg:0.1f} +- {std:0.1f}% higher')
