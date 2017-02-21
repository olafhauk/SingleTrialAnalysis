"""
=========================================================
mSSP: read projected time courses and compute grand-mean
across subjects (after mSSP_make_GM_timecourses.py)
OH July 2016
=========================================================

"""
import sys
sys.path.insert(1, '/imaging/local/software/anaconda/2.4.1/2/lib/python2.7/site-packages/sklearn/')
sys.path.insert(1, '/imaging/local/software/anaconda/2.4.1/2/lib/python2.7/site-packages/pysurfer/')
sys.path.insert(1, '/imaging/local/software/anaconda/2.4.1/2/lib/python2.7/site-packages/nibabel/')
sys.path.insert(1, '/imaging/local/software/mne_python/v0.12/')

import mne
import numpy as np
import matplotlib.pyplot as plt
import pickle
import mSSP_functions
from mSSP_functions import mSSP_plot_avg_timecourses

# file from which grand-mean structure will be read
GM_fname = '/group/erp/data/caroline.coutout/MEG/data/GM_results.pickle'
f_GM = open(GM_fname, 'r')

fig_path = '/group/erp/data/olaf.hauk/MEG/mSSP/results_figures'

print "Reading GM results from {0}".format(GM_fname)
# [TCS_snr_avg_GM, TCS_snr_avg_ALL, epochs] = pickle.load(f_GM)
[TCS_snr_avg_GM, _, epochs] = pickle.load(f_GM)


def mSSP_stack_data(TCS, params):
      # """stack data from different condition/estimators etc. into 
         # one np.array (e.g. for plotting)

    # Parameters
    # ----------
    # TCS: structure with projected time courses (e.g. from mSSP_make_GM_timecourses.py)
    #         TCS_snr_avg_GM[cond dict][win_str dict][nc list][ch_type dict]
    # params: list of lists
    #        parameters specifying data sets to be stacked
    #        every list items should be of the form [cond, win_str, nc], where:
    #           cond: list of list of strings
    #               condition names (e.g. 'lex'/'sem')
    #               all arguments should be lists of equal length
    #           win_str: list of str
    #               strings for estimators/target topographies (e.g. 'mSSP_50-200ms')
    #           nc: list of int
    #               component of estimator (e.g. mSSP components)
 
    # Returns
    # -------
    # data_stack: dict of np.arrays
    # stacked data as np.arrays per channel type as dict."""

    n_list = len(params)

    ch_types = TCS[params[0][0]][params[0][1]][params[0][2]].keys() # channel types

    data_stack = {} # channel types

    for ch_type in ch_types:

        data_list = (TCS[params[idx][0]][params[idx][1]][params[idx][2]][ch_type][0,:] for idx in range(n_list))

        data_stack[ch_type] = np.stack(data_list, axis=0)

    return data_stack
# Done function


param_list = [] # list with parameters for plotting separate figures

### 50-200ms
param_list.append([ \
fig_path + '/TCS_mSSP_MLE_BF_tw150-200ms', # figure name
['mSSP_tw150_nc0', 'mSSP_tw150_nc2', 'mSSP_tw150_nc5', 'mSSP_eye_tw150_nc0', 'MLE_tw150', 'BF_tw150'], # legend entries (for each next list entry)
['lex', 'mSSP_150-200ms', 0], # cond, win_str, n_comp
['lex', 'mSSP_150-200ms', 2],
['lex', 'mSSP_150-200ms', 5],
['lex', 'mSSP_eye_150-200ms', 0],
['lex', 'MLE_150-200ms', 0],
['lex', 'BF_150-200ms', 0],
] )


### 250-500ms
param_list.append([ \
fig_path + '/TCS_mSSP_MLE_BF_tw250-400ms',
['mSSP_tw250_nc0', 'mSSP_tw250_nc2', 'mSSP_tw250_nc5', 'mSSP_eye_tw250_nc0', 'MLE_tw250', 'BF_tw250'],
['lex', 'mSSP_250-400ms', 0],
['lex', 'mSSP_250-400ms', 2],
['lex', 'mSSP_250-400ms', 5],
['lex', 'mSSP_eye_250-400ms', 0],
['lex', 'MLE_250-400ms', 0],
['lex', 'BF_250-400ms', 0],
] )

figs = []
for [pp, params] in enumerate(param_list):
    print pp

    print "Stacking data."
    tcs_plot = mSSP_stack_data(TCS_snr_avg_GM, params[2:])

    fig_fname = params[0]
    legend = params[1]
    print "Plotting, saving figure to {0}.".format(fig_fname)
    t = epochs[params[2][0]].times
    myaxis = [t[0], t[-1], 0, None]
    fig = mSSP_plot_avg_timecourses(avgs=tcs_plot, t=t, titles='', legend=legend, fname_stem=fig_fname, myaxis=myaxis)
    figs.append(fig)