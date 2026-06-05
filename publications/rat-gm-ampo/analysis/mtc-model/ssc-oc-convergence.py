"""
This script checks whether the optimal-control predictions are robust to the
random initial guesses used during optimisation.

For every specimen and imposed SSC parameter value, five optimisation attempts
were run by `ssc-oc-run-predictions.py`. This script reloads those attempts,
computes AMPO for each successful solution and summarises the convergence
quality using:

-   the number of failed or missing optimisation runs;
-   the percentage spread between repeated successful solutions;
-   the overall success rate across cycle frequency, FTS and MTC length
    excursion sweeps.
"""

#%% Load packages & set directories
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
sys.path.append(str(funcDir))

#%% Check convergence for imposed cycle frequency
iters = range(1, 6)
muscles = ['GMe1', 'GMe2', 'GMe3']

iTotal = 0
iFail = 0
pct_diffs = []

#%% Imposed CF
# Define CFs, iterations, and mus
cf_vals = np.arange(1, 6.1, 0.5)
cols = [f"{cf:.1f}" for cf in cf_vals]

# Initialize DataFrame (3D by adding 'mus' as another index)
df_cf = pd.DataFrame(index=iters, columns=pd.MultiIndex.from_product([muscles, cols], names=['mus', 'cf']), dtype=float)

# Loop over mus, cf, and iterations
for mus in muscles:
    parFile = os.path.join(dataDir, mus, mus + '_IM.pkl')
    muspar = pickle.load(open(parFile, 'rb'))[0]
    dataDirSim = os.path.join(dataDir, mus, 'simsOC', 'it', '')
    for cf_str in cols:
        cf = float(cf_str)
        AMPOset = []
        for i in iters:
            filepath = os.path.join(dataDirSim, f"{mus}_cf{cf_str}Hz_ftsOpt_mleOpt_it{i:02d}.csv")
            try:
                df = pd.read_csv(filepath)
                _, _, _, fsee, _, lcerel = df.to_numpy().T
                fce = fsee  # TEMP
                Wmech = -integrate.cumulative_trapezoid(fce, lcerel * muspar['lce_opt'])
                df_cf.loc[i, (mus, cf_str)] = Wmech[-1] * 1e3 * cf
            except:
                continue  # leave as NaN
        # Compute stats if possible
        if len(AMPOset) >= 2:
            AMPOset = -np.sort(-np.array(AMPOset))
            max_val = max(AMPOset)
            pct_diff = [(max_val - val) / max_val * 100 for val in AMPOset][1:]
            pct_diffs.append(pct_diff)   
        
# Compute stats (here stats are computed across the 2D slice per mus)
stats = pd.DataFrame({
    'percent_diff': ((df_cf.max(axis=0) - df_cf.min(axis=0)) / df_cf.max(axis=0) * 100).round(2),
    'nan_count': df_cf.isna().sum(axis=0)
}).T

# Append stats to each level of 'mus'
df_cf = pd.concat([df_cf, stats])

# Calculate total nan_count and highest percent_diff
total_nan_count = stats.loc['nan_count'].sum()
highest_percent_diff = stats.loc['percent_diff'].max()
n_opt = len(iters)*len(cf_vals)*len(muscles)

# Print total nan_count and highest percent_diff
print("\nTotal nan_count across all mus and cf combinations:", total_nan_count)
print(f"Succes rate: {(1-total_nan_count/n_opt)*100:0.2f}%")
print("Highest percent_diff across all mus and cf combinations:", highest_percent_diff)

iTotal += n_opt
iFail += total_nan_count

#%% Check convergence for imposed FTS
# Define FTSs, iterations, and mus
fts_vals = np.arange(0.05, 0.96, 0.05)
cols = [f"{fts:.2f}" for fts in fts_vals]

# Initialize DataFrame (3D by adding 'mus' as another index)
df_fts = pd.DataFrame(index=iters, columns=pd.MultiIndex.from_product([muscles, cols], names=['mus', 'cf']), dtype=float)

# Loop over mus, cf, and iterations
for mus in muscles:
    parFile = os.path.join(dataDir, mus, mus + '_IM.pkl')
    muspar = pickle.load(open(parFile, 'rb'))[0]
    dataDirSim = os.path.join(dataDir, mus, 'simsOC', 'it', '')
    for fts_str in cols:
        fts = float(fts_str)
        AMPOset = []
        for i in iters:
            filepath = os.path.join(dataDirSim, f"{mus}_cfOpt_fts{fts:0.2f}_mleOpt_it{i:02d}.csv")
            try:
                df = pd.read_csv(filepath)
                time, _, _, fsee, _, lcerel = df.to_numpy().T
                fce = fsee  # TEMP
                Wmech = -integrate.cumulative_trapezoid(fce, lcerel * muspar['lce_opt'])
                df_fts.loc[i, (mus, fts_str)] = Wmech[-1] * 1e3 / time[-1]
            except:
                continue  # leave as NaN
        # Compute stats if possible
        if len(AMPOset) >= 2:
            AMPOset = -np.sort(-np.array(AMPOset))
            max_val = max(AMPOset)
            pct_diff = [(max_val - val) / max_val * 100 for val in AMPOset][1:]
            pct_diffs.append(pct_diff)        

# Compute stats (here stats are computed across the 2D slice per mus)
stats = pd.DataFrame({
    'percent_diff': ((df_fts.max(axis=0) - df_fts.min(axis=0)) / df_fts.max(axis=0) * 100).round(2),
    'nan_count': df_fts.isna().sum(axis=0)
}).T

# Append stats to each level of 'mus'
df_fts = pd.concat([df_fts, stats])

# Calculate total nan_count and highest percent_diff
total_nan_count = stats.loc['nan_count'].sum()
highest_percent_diff = stats.loc['percent_diff'].max()
n_opt = len(iters)*len(fts_vals)*len(muscles)

# Print total nan_count and highest percent_diff
print("\nTotal nan_count across all mus and fts combinations:", total_nan_count)
print(f"Succes rate: {(1-total_nan_count/n_opt)*100:0.2f}%")
print("Highest percent_diff across all mus and fts combinations:", highest_percent_diff)

iTotal += n_opt
iFail += total_nan_count

#%% Check convergence for imposed MTC length excursion
mle_vals = np.arange(1, 11.1, 1)
cols = [f'{mle:04.1f}' for mle in mle_vals]

# Initialize DataFrame (3D by adding 'mus' as another index)
df_mle = pd.DataFrame(index=iters, columns=pd.MultiIndex.from_product([muscles, cols], names=['mus', 'mle']), dtype=float)

# Main loop
for mus in muscles:
    parFile = os.path.join(dataDir, mus, mus + '_IM.pkl')
    muspar = pickle.load(open(parFile, 'rb'))[0]
    dataDirSim = os.path.join(dataDir, mus, 'simsOC', 'it', '')
    
    for mle_str in cols:
        mle = float(mle_str)
        AMPOset = []
        for i in iters:
            filepath = os.path.join(dataDirSim, f"{mus}_cfOpt_ftsOpt_mle{mle:04.1f}mm_it{i:02d}.csv")
            try:
                df = pd.read_csv(filepath)
                time, _, _, fsee, _, lcerel = df.to_numpy().T
                fce = fsee  # TEMP
                AMPO = -integrate.trapezoid(fce, lcerel * muspar['lce_opt']) * 1e3 / time[-1]
                AMPOset.append(AMPO)
                df_mle.loc[i, (mus, mle_str)] = AMPO
            except:
                continue  # leave as NaN
        # Compute stats if possible
        if len(AMPOset) >= 2:
            AMPOset = -np.sort(-np.array(AMPOset))
            max_val = max(AMPOset)
            pct_diff = [(max_val - val) / max_val * 100 for val in AMPOset][1:]
            pct_diffs.append(pct_diff)        

# Compute stats (here stats are computed across the 2D slice per mus)
stats = pd.DataFrame({
    'percent_diff': ((df_mle.max(axis=0) - df_mle.min(axis=0)) / df_mle.max(axis=0) * 100).round(2),
    'nan_count': df_mle.isna().sum(axis=0),
}).T

# Append stats to each level of 'mus'
df_mle = pd.concat([df_mle, stats])

# Calculate total nan_count and highest percent_diff
total_nan_count = stats.loc['nan_count'].sum()
highest_percent_diff = stats.loc['percent_diff'].max()
n_opt = len(iters)*len(mle_vals)*len(muscles)

# Print total nan_count and highest percent_diff
print("\nTotal nan_count across all mus and fts combinations:", total_nan_count)
print(f"Succes rate: {(1-total_nan_count/n_opt)*100:0.2f}%")
print("Highest percent_diff across all mus and fts combinations:", highest_percent_diff)

iTotal += n_opt
iFail += total_nan_count

#%% Summarise convergence across all optimisation runs
pct_diffs = [item for sublist in pct_diffs for item in sublist]
print(np.mean(pct_diffs))
print(np.std(pct_diffs))
print(iFail/iTotal*100) # thus 98.2% converged

# iTotal = 615, iFail = 11. Thus 604 succesfull..
