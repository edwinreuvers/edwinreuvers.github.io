# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 10:53:03 2025

@author: Edwin
"""

filenames = ['i_workloop.py', 
             'i_ssc_parameterisation.py',
             'm_conditions.py',
             'r_ampo_cf_fts.py',
             'r_exp_vs_pred.py',
             'r_correlation.py',
             'r_contour.py',
             'r_interrelationships.py',
             'r_oc.py',
             's_prelim.py',
             's_oc.py',
             's_ampo.py']

for filename in filenames:
    runfile(filename)