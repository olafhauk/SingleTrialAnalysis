"""
=========================================================
mSSP: read projected time courses and compute peak and
centre-of-gravity latencies for individual epochs
across subjects (after mSSP_proj_epochs.py)
OH August 2016
=========================================================

"""


print __doc__

# Russell's addition
import sys
sys.path.insert(1, '/imaging/local/software/anaconda/2.4.1/2/lib/python2.7/site-packages/sklearn/')
sys.path.insert(1, '/imaging/local/software/anaconda/2.4.1/2/lib/python2.7/site-packages/pysurfer/')
sys.path.insert(1, '/imaging/local/software/anaconda/2.4.1/2/lib/python2.7/site-packages/nibabel/')
sys.path.insert(1, '/imaging/local/software/mne_python/v0.12/')

import matplotlib
matplotlib.use('Agg') # possibly for running on cluster

import mne
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import mSSP_functions
reload(mSSP_functions)
from mSSP_functions import mSSP_peak_lats_proj_epochs


data_path = '/group/erp/data/caroline.coutout/MEG/data'

# File from which individual subject results will be read, projected timecourses
projections_fstem = '/group/erp/data/caroline.coutout/MEG/data/mSSP_proj_epochs_output'

# file to which result structure will be written
fname_out = '/group/erp/data/caroline.coutout/MEG/data/latencies_150-200ms_results.pickle'
f_out = open(fname_out, 'w')

# lexical and semantic decision tasks
cond_prefix = ['lex', 'sem']

# Latency windows for which peak latencies to be determined
lat_wins = [[0.05, 0.25]]

# estimator to use
estimator = 'MLE_150-200ms'

# Subject file information (ID, date, number, Go event)
subjs = [
    ['meg10_0047', '100317', '2', 1],
    ['meg10_0084', '100408', '4', 1],
    ['meg10_0085', '100409', '5', 2],
    ['meg10_0087', '100412', '7', 2],
    ['meg10_0088', '100412', '8', 1],
    ['meg10_0091', '100413', '9', 2],
    ['meg10_0094', '100414', '10', 1],
    ['meg10_0096', '100415', '11', 2],
    ['meg10_0098', '100415', '12', 1],
    ['meg10_00101', '100416', '13', 2],
    ['meg10_0101', '100416', '14', 1],
    ['meg10_102', '100419', '15', 2],
    ['meg10_00104', '100419', '16', 1],
    ['meg10_0106', '100420', '17', 2], 
    ['meg10_0108', '100420', '18', 1],
    ['meg10_0109', '100421', '19', 2],
    ['meg10_0110', '100422', '20', 1 ],
    ['meg10_0114', '100423', '22', 1]
]

n_subjs = len(subjs)

### initalise structure for average latencies

# Latency measures averaged: dicts for task, condition, latency window, channel type, latency measure, np.array for subjects
peak_lats_avg = {}
# keep latencies for epochs in lists for subjects
peak_lats_epo = {}

for [cci, cc] in enumerate(cond_prefix): # for tasks (lex/sem)

    peak_lats_avg[cc] = {} # for condition (words, liv etc.)
    peak_lats_epo[cc] = {}

    if cc == 'lex':
        conds = ['word', 'pseudo', 'go', 'nogo']
    elif cc == 'sem':
        conds = ['liv', 'nliv', 'go', 'nogo']

    for cond in conds:
 
        peak_lats_avg[cc][cond] = {} # for latency windows
        peak_lats_epo[cc][cond] = {}

        for lat_win in lat_wins:

            win_str = str(int(1000*lat_win[0])) + '-' + str(int(1000*lat_win[1])) + 'ms'                
            peak_lats_avg[cc][cond][win_str] = {} # channel types
            peak_lats_epo[cc][cond][win_str] = {}
            
            for ch_type in ['mag', 'grad', 'eeg']:

                peak_lats_avg[cc][cond][win_str][ch_type] = {} # peak/cof
                peak_lats_epo[cc][cond][win_str][ch_type] = {}
                
                for lat_type in ['peak', 'cof']: # peak/cof

                    peak_lats_avg[cc][cond][win_str][ch_type][lat_type] = np.zeros(n_subjs) # means across epochs per subject
                    peak_lats_epo[cc][cond][win_str][ch_type][lat_type] = [] # values per epoch as lists for subjects


print "Beginning"
for [ssi, ss] in enumerate(subjs):
    print ss[0]

    projections_fname = projections_fstem + '_' + ss[0] + '.pkl'
    f_proj = open(projections_fname, 'r')
    print "Un-pickling results from {0}".format(projections_fname)
    [TCS, TCS_snr, TCS_avg, TCS_snr_avg, Epochs] = pickle.load(f_proj)
    f_proj.close()

    for [cci, cc] in enumerate(cond_prefix): # for conditions
        print cc

        # projected epoch time courses for this subject, condition, and estimator
        proj_epochs = TCS[ss[0],cc][estimator][0]
        
        epochs = Epochs[cc]
        events = epochs.events

        for cond in peak_lats_avg[cc]:

            # find epoch indices for this condition
            events_cond = epochs[cond].events
            epo_idx = np.where(events[:,2] == events_cond[0,2])
            epo_idx = epo_idx[0] # from tuple to np.array

            print "Computing latencies for task {0} and condition {1}.".format(cc,cond)
            peak_lats_now = mSSP_peak_lats_proj_epochs(proj_epochs, epochs.times, lat_wins, epo_idx=epo_idx)

            for win_str in peak_lats_now:
                             
                for ch_type in peak_lats_now[win_str]:
                   
                    for lat_type in peak_lats_now[win_str][ch_type]: # peak/cof
                       
                        # finally: average latency across epochs, for first estimator, keep in np.array
                        peak_lats_avg[cc][cond][win_str][ch_type][lat_type][ssi] = peak_lats_now[win_str][ch_type][lat_type][0,:].mean()
                        # latency values across epochs, keep in lists
                        peak_lats_epo[cc][cond][win_str][ch_type][lat_type].append( peak_lats_now[win_str][ch_type][lat_type][0,:] )

print "Writing latency results to {0}".format(fname_out)
pickle.dump([peak_lats_avg, peak_lats_epo, subjs], f_out)
f_out.close()