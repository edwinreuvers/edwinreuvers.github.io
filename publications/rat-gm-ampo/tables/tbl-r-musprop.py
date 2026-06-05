
import os, sys, pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = os.path.join(baseDir,'data')
funcDir = os.path.join(baseDir,'analysis','functions')
sys.path.append(str(funcDir))

import hillmodel, stats

#%% Initialize dataframe
# Define the variable names and types
# varnames,z = [['Hill constant', 'a'], ['Hill constant', 'b']].T

varnames = [['arel',    '$a_{rel}$',        '-',        'Hill constant'], 
            ['brel',    '$b_{rel}$',        '-',        'Hill constant'], 
            ['fmax',    '$F_{CE}^{max}$',   'N',        'maximal isometric CE force'],
            ['kpee',    '$k_{PEE}$',        'N/mm<sup>2</sup>', 'PEE stiffness scaling factor'], 
            ['ksee',    '$k_{SEE}$',        'N/mm<sup>2</sup>', 'SEE stiffness scaling factor'], 
            ['lce_opt', '$L_{CE}^{opt}$',   'mm',       'CE optimum length'],
            ['lpee0',   '$L_{PEE}^0$',      'mm',       'PEE slack length'],
            ['lsee0',   '$L_{SEE}^0$',      'mm',       'SEE slack length'],
            ['tact',    r'$\tau_{act}$',    'ms',       'Calcium dynamics activation time constant'],
            ['tdeact',  r'$\tau_{deact}$',  'ms',       'Calcium dynamics deactivation time constant'],
            ['lmtcOpt', '$L_{MTC}^{opt}$',  'mm',       'MTC length yielding maximal isometric SEE force'],
            ['pmax',    '$P_{CE}^{max}$',   'mW',       'maximal instantaneous CE power'],
            ['tHRise',  '$t_{HRT}$',        'ms',       r"`half-rise time'"],
            ['vmax',    '$v_{CE}^{max}$',   'mm/s',     'maximal CE shortening velocity'],
            ['vceopt',  '$v_{CE}^{opt}$',   'mm/s',     'CE velocity at $P_{CE}^{max}$']]

variables,symbols,units,descriptions, = list(zip(*varnames))  # This transposes the list of lists

types = ['MP'] * 10 +['DM']*5
muscles = ['GMe1', 'GMe2', 'GMe3']

# Create the column names dynamically
columns = ['type','Description','Symbol','Unit'] + muscles

# Initialize the data dict with placeholder values
data = {col: [] for col in columns}

#%%  Fill the DataFrame
num_vars = len(variables)
num_gmes = len(muscles)

for iMus,mus in enumerate(muscles):
    parFile = os.path.join(dataDir,mus,mus+'_IM.pkl')
    muspar,dataQRout, dataSRout, dataACTout = pickle.load(open(parFile, 'rb'))
    
    # Compute arel & brel
    muspar['arel'] = muspar['a']/muspar['fmax']
    muspar['brel'] = muspar['b']/muspar['lce_opt']
    
    # Compute helper variables (or whatever we'll call it)
    muspar['lmtcOpt'] = (muspar['fmax']/muspar['ksee'])**0.5 + muspar['lsee0'] + muspar['lce_opt'] # [m] MTC length yielding maximal isometric SEE force
    muspar['vmax'] = muspar['b']/muspar['a']*muspar['fmax'] # [m/s] maximal CE shortening velocity
    muspar['vcenorm'] = muspar['a']*((1+muspar['fmax']/muspar['a'])**0.5-1)/muspar['fmax'] # [ ] normalised CE velocity at which instantaneous CE power is maximal 
    muspar['vceopt'] = muspar['vcenorm']*muspar['vmax']
    fcenorm = muspar['vcenorm'] # [ ] normalised CE force at which instantaneous CE power is maximal 
    muspar['pmax'] = muspar['vcenorm']*fcenorm*muspar['vmax']*muspar['fmax'] # [W] value of maximal instantaneous CE power
    _,_,gamma05 = hillmodel.act_state(np.nan,1,muspar) # [ ] [ ] value of gamma at which q=0.5
    muspar['tHRise'] = -muspar['tact']*np.log(1-gamma05) # [s] "half-rise time"
    
    for iVar,(var,description,symbol,unit,vartype) in enumerate(zip(variables,descriptions,symbols,units,types)):
        vartype = types[iVar]
        
        # Append data for each variable and its corresponding type
        if iMus == 0:  # Only append var and type once per variable
            data['type'].append(vartype)
            data['Description'].append(description)
            data['Symbol'].append(symbol)
            data['Unit'].append(unit)
        
        # Fill the GMe columns
        if var in ['lce_opt', 'lpee0', 'lsee0', 'tact', 'tdeact', 'lmtcOpt', 'pmax', 'tHRise', 'vmax', 'vceopt']:
            value = muspar[var]*1e3
        elif var in ['kpee', 'ksee']:
            value = muspar[var]/1e3
        else:
            value = muspar[var]
        data[mus].append(str(stats.str_round(value,3)))  # Adjust the logic as needed
        
# Create the DataFrame
df = pd.DataFrame(data)
df.map(str)
fileName = 'tbl_musprop'
df.to_csv(dataDir+'/'+fileName+'.csv', index=False, header=['type','description',
                    'symbol', 'unit', 'GMe1','GMe2','GMe3e'])

#%% TeX table
from great_tables import GT
from gt_tex import make_latex, delete_rows, insert_rows

df_tex = df.copy()
df_tex = df_tex.drop('type', axis=1)

gt_table = (GT(df_tex)
    #.tab_stub(rowname_col="description", groupname_col="type")
    .cols_align(align='center') 
    .cols_align(align='left', columns=['Description'])
    .cols_label(GMe1='1',GMe2='2',GMe3='3')
)

# Transform to LateX table
latex_str = make_latex(gt_table.as_latex())
latex_str = delete_rows(latex_str, row_numbers=[0])
add_rows = {
    0: r"  \bfseries Description & \bfseries Symbol & \bfseries Unit & \bfseries Rat 1 & \bfseries Rat 2 & \bfseries Rat 3 \\ \hline",
    1: r"  \multicolumn{6}{|l|}{\itshape MTC properties} \\ \hline",
    12: r"  \multicolumn{6}{|l|}{\itshape Derived metrics} \\ \hline"
}

latex_str = insert_rows(latex_str, add_rows)

print(latex_str)

# Write to a .tex file
with open("tbl-r-musprop.tex", "w", encoding="utf-8") as f:
    f.write(latex_str) 

#%% Great table
df_tex = df.copy()

