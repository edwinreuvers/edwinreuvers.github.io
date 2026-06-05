#| label: supptbl-sscpa
#| tbl-cap: SSC parameters, stimulation durations, and measured AMPO of experimental stretch-shortening
#|   cycles with a 4 mm MTC length excursion. Stimulation onset was set at the start of MTC shortening in
#|   all conditions.

#%% Load packages & set directories
import os, sys
import numpy as np
import pandas as pd
from great_tables import GT, style, loc
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = os.path.join(baseDir,'data')
funcDir = os.path.join(baseDir,'analysis','functions')
sys.path.append(str(funcDir))

import stats , stimulation

#%% Set-up
exp = 'SSC_PA'
muscles = ['GMe1', 'GMe2', 'GMe3']

#%% Compute data values
# Motion parameters
if exp == 'SSC_PA':
    cf = np.array([1, 2, 3, 4, 5, 3, 3, 3, 3, 5, 4, 2, 1])
elif exp == 'SSC_PB':
    cf = np.array([1, 1.5, 2, 2.5, 3, 2, 2, 2, 2, 3, 2.5, 1.5, 1])

fts = np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.80, 0.65, 0.35, 0.20, 0.80, 0.65, 0.35, 0.20])
tShort = fts / cf * 1e3
tLeng = (1 - fts) / cf * 1e3

# Stimulation duration trial 1
iSuperscript = 1  
iTrial = 1
durStim1 = np.empty((len(muscles),len(cf)))
for iMus, mus in enumerate(muscles):      
    filepaths = [os.path.join(dataDir,mus,'dataExp',exp,f'{mus}_{exp}{iCond:02d}_{iTrial:01d}.csv') for iCond in range(1,14)]
    durStim  = stimulation.get_stim_dur(filepaths)
    durStim1[iMus,:] = [x*1e3 for x in durStim] # to ms

durStim1_str = [] 
for iCond in range(1,len(cf)+1):
    same,diff,i = stats.analyse_3similar(durStim1[:,iCond-1],1)
    if i == True:
        durStim1_str.append(stats.str_round(same,2))
    else:
        durStim1_str.append(stats.str_round(same,2)+f'<sup>{iSuperscript}</sup>')
        #print(f'{iSuperscript}: Cond {iCond:02d}, GMe{i+1} stimDuration = {diff:0.0f} ms')
        iSuperscript +=1
        
# Stimulation duration trial 2
iSuperscript = 1  
iTrial = 2
durStim2 = np.empty((len(muscles),len(cf)))
for iMus, mus in enumerate(muscles):   
    filepaths = [os.path.join(dataDir,mus,'dataExp',exp,f'{mus}_{exp}{iCond:02d}_{iTrial:01d}.csv') for iCond in range(1,14)]
    durStim  = stimulation.get_stim_dur(filepaths)
    durStim2[iMus,:] = [x*1e3 for x in durStim] # to ms

durStim2_str = [] 
for iCond in range(1,len(cf)+1):
    same,diff,i = stats.analyse_3similar(durStim2[:,iCond-1],1)
    if i == True:
        durStim2_str.append(stats.str_round(same,2))
    else:
        durStim2_str.append(stats.str_round(same,2)+f'<sup>{iSuperscript}</sup>')
        #print(f'{iSuperscript}: Cond {iCond:02d}, GMe{i+1} stimDuration = {diff:0.0f} ms')
        iSuperscript +=1
        
# AMPO of the rats:
AMPO = []
for mus in ['GMe1', 'GMe2', 'GMe3']:
    fileName = mus+'_dataAMPO'
    df = pd.read_excel(dataDir+'/'+mus+'/'+fileName+'.xlsx')
    ampoData = df.to_numpy()
    
    if exp == 'SSC_PA':
        t1 = np.mean(ampoData[0:3,5:],0)
        t2 = np.mean(ampoData[3:6,5:],0)
    elif exp == 'SSC_PB':
        t1 = np.mean(ampoData[6:9,5:],0)
        t2 = np.mean(ampoData[9:12,5:],0)
    t1 = [stats.str_round(x,2) for x in t1]
    t2 = [stats.str_round(x,2) for x in t2]
    
    AMPO.append(t1)
    AMPO.append(t2)
AMPO = np.array(AMPO)

#%%
# First 4 rows are calculated values; rest are placeholders (NaN)
data_values = np.full((12, len(cf)), np.nan, dtype=object)
data_values[0] = [stats.str_round(x,2) for x in cf]
data_values[1] = [stats.str_round(x,2) for x in fts]
data_values[2] = [stats.str_round(x,3) for x in tShort]
data_values[3] = [stats.str_round(x,3) for x in tLeng]
data_values[4] = durStim1_str
data_values[5] = durStim2_str
data_values[6:] = AMPO

# === Create DataFrame ===
descriptions = [
    'Cycle frequency', 'FTS', 'MTC shortening time', 'MTC lengthening time',
    'Trial 1', 'Trial 2',
    'Trial 1', 'Trial 2',
    'Trial 1', 'Trial 2',
    'Trial 1', 'Trial 2',
]

units = ['Hz', '-', 'ms', 'ms', 'ms', 'ms', 'mW', 'mW', 'mW', 'mW', 'mW', 'mW']
types = ['SSC parameters'] * 2 + ['MTC shortening and lengthening times'] * 2 + ['Stimulation durations'] * 2 + ['Measured AMPO of rat 1'] * 2 + ['Measured AMPO of rat 2'] * 2 + ['Measured AMPO of rat 3'] * 2
conds = [str(i) for i in range(1, 14)]

df = pd.DataFrame(
    data=np.column_stack([types, descriptions, units, data_values]),
    columns=['type', 'Description', 'Unit'] + conds
)

#%% TeX table
from great_tables import GT
from gt_tex import make_latex, insert_rows, fix_reference, replace_latex_table_cell, delete_rows, replace_superscripts

df_tex = df.copy()
df_tex = df_tex.drop('type', axis=1)

gt_table = (GT(df_tex)
    #.tab_stub(rowname_col="description", groupname_col="type")
    .cols_align(align='center') 
    .cols_align(align='left', columns=['Description'])
    .cols_label(Description='')
)

latex_str = make_latex(gt_table.as_latex())
add_rows = {
    0: r"  & & \multicolumn{13}{c|}{Condition}  \\ \hline",
    1: r"  \bfseries & \bfseries Unit & \bfseries 1 & \bfseries 2 & \bfseries 3 & \bfseries 4 & \bfseries 5 & \bfseries 6 & \bfseries 7 & \bfseries 8 & \bfseries 9 & \bfseries 10 & \bfseries 11 & \bfseries 12 & \bfseries 13 \\ \hline",
    2: r"  \multicolumn{15}{|l|}{\itshape SSC parameters} \\ \hline",
    5: r"  \multicolumn{15}{|l|}{\itshape MTC shortening and lengthening times} \\ \hline",
    8: r"  \multicolumn{15}{|l|}{\itshape Stimulation durations} \\ \hline",
    11: r"  \multicolumn{15}{|l|}{\itshape Measured AMPO of Rat 1} \\ \hline",
    14: r"  \multicolumn{15}{|l|}{\itshape Measured AMPO of Rat 2} \\ \hline",
    17: r"  \multicolumn{15}{|l|}{\itshape Measured AMPO of Rat 3} \\ \hline",
}
latex_str = delete_rows(latex_str, row_numbers=[0])
latex_str = insert_rows(latex_str, add_rows)

latex_str = replace_latex_table_cell(latex_str, row=8, col=0, new_text=r'Trial 1')
latex_str = replace_latex_table_cell(latex_str, row=9, col=0, new_text=r'Trial 2')
latex_str = replace_latex_table_cell(latex_str, row=10, col=0, new_text=r'Trial 1')
latex_str = replace_latex_table_cell(latex_str, row=11, col=0, new_text=r'Trial 2')
latex_str = replace_latex_table_cell(latex_str, row=12, col=0, new_text=r'Trial 1')
latex_str = replace_latex_table_cell(latex_str, row=13, col=0, new_text=r'Trial 2')

# Write to a .tex file
latex_str += (r"\break\hfill\footnotesize{"+ 
              r"\textsuperscript{1} For conditon 1: stimulation duration of rat 3 was 455 ms. "
              r"\textsuperscript{2} For conditon 9: stimulation duration of rat 3 was 23 ms. "
              r"\textsuperscript{3} For conditon 13: stimulation duration of rat 1 was 165 ms.}")

with open('supptbl-sscpa.tex', "w", encoding="utf-8") as f:
    f.write(latex_str)


#%% Great table
from great_tables import GT, md
df_gt = df.copy()

gt_table = (GT(df_gt)
    .tab_spanner(label = "Condition", columns = [f'{x}' for x in range(1,14)])
    .tab_stub(rowname_col="Description", groupname_col="type")
    .tab_style(style = style.text(style = "italic"), locations = loc.row_groups())
    .tab_source_note(
        source_note = md("<sup>1</sup> For conditon 1: stimulation duration of rat 3 was 455 ms.")
    )
    .tab_source_note(
        source_note = md("<sup>2</sup> For conditon 9: stimulation duration of rat 3 was 23 ms.")
    )
    .tab_source_note(
        source_note = md("<sup>3</sup> For conditon 13: stimulation duration of rat 1 was 165 ms.")
    )
)
gt_table


