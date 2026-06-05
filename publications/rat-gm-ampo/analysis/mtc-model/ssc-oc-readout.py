"""
This script selects the representative optimal-control solution from repeated
optimisation attempts.

The optimal-control predictions are generated multiple times for each imposed
SSC parameter value because the solver can converge to slightly different
solutions depending on the initial guess. For each parameter value, this script
loads the repeated solutions, computes mechanical work over the cycle and keeps
the solution with the highest mechanical work. The selected solution is written
to the main `simsOC` folder, where it can be used by figure scripts and by the
comparison with constant-velocity SSCs.

The printed values report the percentage difference between the worst and best
successful repeated solution for each imposed parameter value. These values are
used as a quick readout of how much the repeated solutions differed.
"""

#%% Load packages, directories and muscle parameters
import os, sys, pickle
import numpy as np
import pandas as pd
from scipy import integrate
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'

mus = 'GMe3'
parFile = os.path.join(dataDir, mus, mus + '_IM.pkl')
muspar = pickle.load(open(parFile, 'rb'))[0]

#%% Select best solutions for imposed cycle frequency
for cf in np.arange(1,8.1,0.5):
    Wset = []
    for i in range(1,3):
        filepath = os.path.join(dataDir,mus,'simsOC', 'it', f"{mus}_cf{cf:0.1f}Hz_ftsOpt_mleOpt_it{i:02d}.csv")
        
        try:
            df = pd.read_csv(filepath)
            _, _, _, fsee, _, lcerel = df.to_numpy().T
            fce = fsee  # TEMP
            Wmech = -integrate.trapezoid(fce, lcerel * muspar['lce_opt'])
            Wset.append(Wmech)
        except:
            Wset.append(np.nan)
    
    # Find and save Max
    iMax = np.nanargmax(Wset)+1; iMin = np.nanargmin(Wset)+1
    print((Wset[iMin-1]-Wset[iMax-1])/Wset[iMax-1]*100)
    filepath = os.path.join(dataDir,mus,'simsOC', 'it', f"{mus}_cf{cf:0.1f}Hz_ftsOpt_mleOpt_it{iMax:02d}.csv")
    df = pd.read_csv(filepath)
    
    filepath = os.path.join(dataDir,mus,'simsOC', f"{mus}_cf{cf:0.1f}Hz_ftsOpt_mleOpt.csv")
    df.to_csv(filepath,index=False,header=['Time [s]','Lmtc [m]','STIM [ ]', 'Fsee [N]', 'Gamma [ ]','Lcerel [ ]'])

#%% Select best solutions for imposed FTS
for fts in np.arange(0.05,0.96,0.05):
    Wset = []
    for i in range(1,11):
        filepath = os.path.join(dataDir,mus,'simsOC', 'it', f"{mus}_cfOpt_fts{fts:0.2f}_mleOpt_it{i:02d}.csv")
        
        try:
            df = pd.read_csv(filepath)
            _, _, _, fsee, _, lcerel = df.to_numpy().T
            fce = fsee  # TEMP
            Wmech = -integrate.trapezoid(fce, lcerel * muspar['lce_opt'])
            Wset.append(Wmech)
        except:
            Wset.append(np.nan)

    # Find and save Max
    iMax = np.nanargmax(Wset)+1; iMin = np.nanargmin(Wset)+1
    print((Wset[iMin-1]-Wset[iMax-1])/Wset[iMax-1]*100)
    filepath = os.path.join(dataDir,mus,'simsOC', 'it', f"{mus}_cfOpt_fts{fts:0.2f}_mleOpt_it{iMax:02d}.csv")
    df = pd.read_csv(filepath)
    
    filepath = os.path.join(dataDir,mus,'simsOC', f"{mus}_cfOpt_fts{fts:0.2f}_mleOpt.csv")
    df.to_csv(filepath,index=False,header=['Time [s]','Lmtc [m]','STIM [ ]', 'Fsee [N]', 'Gamma [ ]','Lcerel [ ]'])
    
#%% Select best solutions for imposed MTC length excursion
for mle in np.arange(2,11.1,1):
    Wset = []
    for i in range(1,4):
        filepath = os.path.join(dataDir,mus,'simsOC', 'it', f"{mus}_cfOpt_ftsOpt_mle{mle:04.1f}mm_it{i:02d}.csv")
        
        try:
            df = pd.read_csv(filepath)
            _, _, _, fsee, _, lcerel = df.to_numpy().T
            fce = fsee  # TEMP
            Wmech = -integrate.trapezoid(fce, lcerel * muspar['lce_opt'])
            Wset.append(Wmech)
        except:
            Wset.append(np.nan)

    # Find and save Max
    iMax = np.nanargmax(Wset)+1; iMin = np.nanargmin(Wset)+1
    print((Wset[iMin-1]-Wset[iMax-1])/Wset[iMax-1]*100)
    filepath = os.path.join(dataDir,mus,'simsOC', 'it', f"{mus}_cfOpt_ftsOpt_mle{mle:04.1f}mm_it{iMax:02d}.csv")
    df = pd.read_csv(filepath)
    
    filepath = os.path.join(dataDir,mus,'simsOC', f"{mus}_cfOpt_ftsOpt_mle{mle:04.1f}mm.csv")
    df.to_csv(filepath,index=False,header=['Time [s]','Lmtc [m]','STIM [ ]', 'Fsee [N]', 'Gamma [ ]','Lcerel [ ]'])
    
#%% Select illustrative Rat 1 solutions with imposed FTS at 3.5 Hz
cf = 3.5
if mus == 'GMe1':
    for fts in [0.50, 0.95]:
        Wset = []
        for i in range(1,6):
            filepath = os.path.join(dataDir,mus,'simsOC', 'it', f"{mus}_cf{cf:0.1f}Hz_fts{fts:0.2f}_mleOpt_it{i:02d}.csv")
            
            try:
                df = pd.read_csv(filepath)
                _, _, _, fsee, _, lcerel = df.to_numpy().T
                fce = fsee  # TEMP
                Wmech = -integrate.trapezoid(fce, lcerel * muspar['lce_opt'])
                Wset.append(Wmech)
            except:
                Wset.append(np.nan)
    
        # Find and save Max
        iMax = np.nanargmax(Wset)+1; iMin = np.nanargmin(Wset)+1
        print((Wset[iMin-1]-Wset[iMax-1])/Wset[iMax-1]*100)
        filepath = os.path.join(dataDir,mus,'simsOC', 'it', f"{mus}_cf{cf:0.1f}Hz_fts{fts:0.2f}_mleOpt_it{iMax:02d}.csv")
        df = pd.read_csv(filepath)
        
        filepath = os.path.join(dataDir,mus,'simsOC', f"{mus}_cf{cf:0.1f}Hz_fts{fts:0.2f}_mleOpt.csv")
        df.to_csv(filepath,index=False,header=['Time [s]','Lmtc [m]','STIM [ ]', 'Fsee [N]', 'Gamma [ ]','Lcerel [ ]'])
        
#%% Select illustrative Rat 1 solutions with imposed MTC length excursion at 3.5 Hz
if mus == 'GMe1':
    for mle in [2, 10]:
        Wset = []
        for i in range(1,6):
            filepath = os.path.join(dataDir,mus,'simsOC', 'it', f"{mus}_cf{cf:0.1f}Hz_ftsOpt_mle{mle:04.1f}mm_it{i:02d}.csv")
            
            try:
                df = pd.read_csv(filepath)
                _, _, _, fsee, _, lcerel = df.to_numpy().T
                fce = fsee  # TEMP
                Wmech = -integrate.trapezoid(fce, lcerel * muspar['lce_opt'])
                Wset.append(Wmech)
            except:
                Wset.append(np.nan)
    
        # Find and save Max
        iMax = np.nanargmax(Wset)+1; iMin = np.nanargmin(Wset)+1
        print((Wset[iMin-1]-Wset[iMax-1])/Wset[iMax-1]*100)
        filepath = os.path.join(dataDir,mus,'simsOC', 'it', f"{mus}_cf{cf:0.1f}Hz_ftsOpt_mle{mle:04.1f}mm_it{iMax:02d}.csv")
        df = pd.read_csv(filepath)
        
        filepath = os.path.join(dataDir,mus,'simsOC', f"{mus}_cf{cf:0.1f}Hz_ftsOpt_mle{mle:04.1f}mm.csv")
        df.to_csv(filepath,index=False,header=['Time [s]','Lmtc [m]','STIM [ ]', 'Fsee [N]', 'Gamma [ ]','Lcerel [ ]'])
    
