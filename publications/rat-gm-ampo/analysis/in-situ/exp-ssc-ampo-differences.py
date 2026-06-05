"""
This script computes selected percentage differences in experimentally measured
AMPO for reporting in the manuscript.

The calculations summarise how AMPO changes with cycle frequency and FTS for
the 4 mm and 8 mm MTC length excursion conditions. Quadratic fits are used only
to estimate the approximate cycle frequency at which AMPO peaks for FTS = 0.5.
"""

#%% Load packages & set directories
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

plt.close('all')

import stats

#%% Estimate optimum cycle frequency at 4 mm MTC length excursion
AMPO = np.empty((0,5))
for mus,clr in zip(['GMe1','GMe2','GMe3'],['k','r','g','b']):
    filepath = os.path.join(dataDir,mus,str(mus)+'_dataAMPO.xlsx')
    dataExp = pd.read_excel(filepath).to_numpy()[:,5:].astype(float)
    
    iCol = np.r_[1,2,3,4,5]-1 
    dataExp[dataExp<1] = np.nan
    AMPO = np.vstack((AMPO,dataExp[3:6,iCol]))

idx = np.isfinite(AMPO)
cf = np.tile([1,2,3,4,5],(9,1))
coef = np.polyfit(cf[idx].flatten(),AMPO[idx].flatten(),2)

fig, ax = plt.subplots()   
ax.plot(cf.T,AMPO.T)

cf = np.linspace(1,5,1000)
AMPOfit = np.polyval(coef,cf)
ax.plot(cf,AMPOfit,'--')

iMax = np.argmax(AMPOfit)
print(f"For FTS = 0.5 and MLE = 4 mm, optimum cycle frequency ≈ {cf[iMax]:.1f} Hz")

#%% Estimate optimum cycle frequency at 8 mm MTC length excursion
AMPO = np.empty((0,5))
for mus,clr in zip(['GMe1','GMe2','GMe3'],['k','r','g','b']):
    filepath = os.path.join(dataDir,mus,str(mus)+'_dataAMPO.xlsx')
    dataExp = pd.read_excel(filepath).to_numpy()[:,5:].astype(float)
    
    iCol = np.r_[1,2,3,4,5]-1 
    dataExp[dataExp<1] = np.nan
    AMPO = np.vstack((AMPO,dataExp[9:12,iCol]))

idx = np.isfinite(AMPO)
cf = np.tile([1.0,1.5,2.0,2.5,3.0],(9,1))
coef = np.polyfit(cf[idx].flatten(),AMPO[idx].flatten(),2)

fig, ax = plt.subplots()   
ax.plot(cf.T,AMPO.T)

cf = np.linspace(1,3,1000)
AMPOfit = np.polyval(coef,cf)
ax.plot(cf,AMPOfit,'--')

iMax = np.argmax(AMPOfit)
print(f"For FTS = 0.5 and MLE = 8 mm, optimum cycle frequency ≈ {cf[iMax]:.1f} Hz")

#%% Compute FTS effects at 4 mm MTC length excursion
print()
print('MTC length excursion = 4 mm')

AMPO = np.empty((0,13))
for mus,clr in zip(['GMe1','GMe2','GMe3'],['k','r','g','b']):
    filepath = os.path.join(dataDir,mus,str(mus)+'_dataAMPO.xlsx')
    dataExp = pd.read_excel(filepath).to_numpy()[:,5:].astype(float)
    
    dataExp[dataExp<1] = np.nan
    AMPO = np.vstack((AMPO,dataExp[3:6,:]))

# FTS 0.20 -> 0.50 - 4 mm @ 3Hz
pDiff = np.nanmean(stats.pdiff(AMPO[:,2],AMPO[:,8]))
print(f"FTS 0.20 -> 0.50 - 4mm@3Hz: {pDiff:.1f} %")

# FTS 0.50 -> 0.80 - 4 mm @ 3Hz
pDiff = np.nanmean(stats.pdiff(AMPO[:,5],AMPO[:,2]))
print(f"FTS 0.50 -> 0.80 - 4mm@3Hz: {pDiff:.1f} %")

# FTS 0.65 -> 0.80 - 4 mm @ 3Hz
pDiff = np.nanmean(stats.pdiff(AMPO[:,5],AMPO[:,6]))
print(f"FTS 0.65 -> 0.80 - 4mm@3Hz: {pDiff:.1f} %")

# FTS 0.50 -> 0.80 - 4 mm @ 5Hz
pDiff = np.nanmean(stats.pdiff(AMPO[:,9],AMPO[:,4]))
print(f"FTS 0.50 -> 0.80 - 4mm@5Hz: {pDiff:.1f} %")

#%% Compute FTS effects at 8 mm MTC length excursion
print()
print('MTC length excursion = 8 mm')
AMPO = np.empty((0,13))
for mus,clr in zip(['GMe1','GMe2','GMe3'],['k','r','g','b']):
    filepath = os.path.join(dataDir,mus,str(mus)+'_dataAMPO.xlsx')
    dataExp = pd.read_excel(filepath).to_numpy()[:,5:].astype(float)
    
    dataExp[dataExp<1] = np.nan
    AMPO = np.vstack((AMPO,dataExp[9:12,:]))

# FTS 0.20 -> 0.50 - 8 mm @ 2Hz
pDiff = np.nanmean(stats.pdiff(AMPO[:,2],AMPO[:,8]))
print(f"FTS 0.20 -> 0.50 - 8mm@2Hz: {pDiff:.1f} %")

# FTS 0.50 -> 0.80 - 8 mm @ 2Hz
pDiff = np.nanmean(stats.pdiff(AMPO[:,5],AMPO[:,2]))
print(f"FTS 0.50 -> 0.80 - 8mm@2Hz: {pDiff:.1f} %")

# FTS 0.65 -> 0.80 - 8 mm @ 2Hz
pDiff = np.nanmean(stats.pdiff(AMPO[:,5],AMPO[:,6]))
print(f"FTS 0.65 -> 0.80 - 8mm@2Hz: {pDiff:.1f} %")

# FTS 0.50 -> 0.80 - 4 mm @ 3Hz
pDiff = np.nanmean(stats.pdiff(AMPO[:,9],AMPO[:,4]))
print(f"FTS 0.65 -> 0.80 - 8mm@3Hz: {pDiff:.1f} %")








