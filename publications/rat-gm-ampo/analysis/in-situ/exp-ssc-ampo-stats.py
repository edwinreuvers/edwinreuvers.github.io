"""
This script tests whether cycle frequency and FTS affect experimentally
measured AMPO.

The AMPO spreadsheet is reshaped into a long-format dataframe for the four
subsets shown in the experimental AMPO figure. For each subset, a linear
mixed-effects model with linear and quadratic terms is fitted, using specimen
as random effect.
"""

#%% Load packages & set directories
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import mixedlm
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

#%% Make dataframe
data = {
    'specimen': [],
    'cf': [],
    'fts': [],
    'mle': [],
    'trial': [],
    'AMPO': [],
    'subset': [],
}
    
for iMus,mus in enumerate(['GMe1','GMe2','GMe3']):
    dataDirMus = os.path.join(dataDir,mus,'')
    dataExp = pd.read_excel(dataDirMus+str(mus)+'_dataAMPO.xlsx').to_numpy()[:,5:].astype(float)
    dataExp[dataExp < 1] = np.nan
        
    # Panel A: 4mm MLE - effect of CF
    cf = np.matlib.repmat([1,2,3,4,5],6,1) # cf
    fts = np.matlib.repmat([0.5],6,5) # fts
    mle = np.matlib.repmat([4],6,5)
    trial = np.repeat([1, 2], 3)[:, None].repeat(5, axis=1)
    AMPO = dataExp[0:6,0:5]
    
    data['specimen'].extend([mus]*30)
    data['cf'].extend(cf.flatten())
    data['fts'].extend(fts.flatten())
    data['mle'].extend(mle.flatten())
    data['trial'].extend(trial.flatten())
    data['AMPO'].extend(AMPO.flatten())
    data['subset'].extend(['A']*30)
    
    # Panel B: 8mm MLE - effect of FTS
    cf = np.matlib.repmat([1,1.5,2,2.5,3],6,1) # cf
    fts = np.matlib.repmat([0.5],6,5) # fts
    mle = np.matlib.repmat([8],6,5)
    trial = np.repeat([1, 2], 3)[:, None].repeat(5, axis=1)
    AMPO = dataExp[6:12,0:5]
    
    data['specimen'].extend([mus]*30)
    data['cf'].extend(cf.flatten())
    data['fts'].extend(fts.flatten())
    data['mle'].extend(mle.flatten())
    data['trial'].extend(trial.flatten())
    data['AMPO'].extend(AMPO.flatten())
    data['subset'].extend(['B']*30)
    
    # Panel C: 4mm MLE - effect of CF
    cf = np.matlib.repmat([3],6,5) # cf
    fts = np.matlib.repmat([0.80,0.65,0.50,0.35,0.20],6,1) # fts
    mle = np.matlib.repmat([4],6,5)
    trial = np.repeat([1, 2], 3)[:, None].repeat(5, axis=1)
    AMPO = dataExp[0:6,[5,6,2,7,8]]
    
    data['specimen'].extend([mus]*30)
    data['cf'].extend(cf.flatten())
    data['fts'].extend(fts.flatten())
    data['mle'].extend(mle.flatten())
    data['trial'].extend(trial.flatten())
    data['AMPO'].extend(AMPO.flatten())
    data['subset'].extend(['C']*30)

    # Panel D: 8mm MLE - effect of FTS
    cf = np.matlib.repmat([2],6,5) # cf
    fts = np.matlib.repmat([0.80,0.65,0.50,0.35,0.20],6,1) # fts
    mle = np.matlib.repmat([8],6,5)
    trial = np.repeat([1, 2], 3)[:, None].repeat(5, axis=1)
    AMPO = dataExp[6:12,[5,6,2,7,8]]
    
    data['specimen'].extend([mus]*30)
    data['cf'].extend(cf.flatten())
    data['fts'].extend(fts.flatten())
    data['mle'].extend(mle.flatten())
    data['trial'].extend(trial.flatten())
    data['AMPO'].extend(AMPO.flatten())
    data['subset'].extend(['D']*30)

df = pd.DataFrame(data)

#%% Create function to run mixedLM
def run_mixed_model(df, subset, predictor, xlabel=None):
    df_sub = df[df['subset'] == subset].copy()
    df_sub = df_sub.dropna(subset=['AMPO'])

    # Model fit
    formula = f"AMPO ~ {predictor} + I({predictor}**2)"
    model = mixedlm(formula, df_sub, groups=df_sub["specimen"])
    result = model.fit()

    #print(f"\n=== Subset {subset} ({predictor}) ===")
    #print(result.summary())
    
    # P-values
    pvals = result.pvalues
    p_lin = pvals[predictor]
    p_quad = pvals[f'I({predictor} ** 2)']
    
    if xlabel is not None:
        # Prediction
        x_vals = np.linspace(df_sub[predictor].min(), df_sub[predictor].max(), 100)
        x2_vals = x_vals**2
    
        fe = result.fe_params
        y_pred = (
            fe['Intercept']
            + fe[predictor]*x_vals
            + fe[f'I({predictor} ** 2)']*x2_vals
        )
        
        # Plot
        plt.figure(figsize=(6,4))
    
        for mus in df_sub['specimen'].unique():
            df_mus = df_sub[df_sub['specimen'] == mus]
            plt.scatter(df_mus[predictor], df_mus['AMPO'], label=mus, alpha=0.7)
    
        plt.plot(x_vals, y_pred, 'k-', linewidth=2, label='Mixed model fit')
    
        plt.xlabel(xlabel)
        plt.ylabel('AMPO')
        plt.title(f'Subset {subset}: AMPO vs {predictor}')
        plt.ylim(0, np.ceil(np.max(df_sub['AMPO'])/10)*10)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return result, p_lin, p_quad

#%% Run linear mixed models
# Panel A & B: effect CF
result_A, pl_A, pq_A = run_mixed_model(df, 'A', 'cf') #, 'Cycle frequency [Hz]')
result_B, pl_B, pq_B = run_mixed_model(df, 'B', 'cf') #, 'Cycle frequency [Hz]')

# Panel C & D: effect FTS
result_C, pl_C, pq_C = run_mixed_model(df, 'C', 'fts') #, 'FTS')
result_D, pl_D, pq_D = run_mixed_model(df, 'D', 'fts') #, 'FTS')

# Store all p-values
pl = np.array([pl_A, pl_B, pl_C, pl_D]) # linear terms
pq = np.array([pq_A, pq_B, pq_C, pq_D]) # quadratic terms
p_all = np.array([pl, pq]) # all p-values
