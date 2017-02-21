"""
=========================================================
mSSP: read projected time courses and compute grand-mean
across subjects (after mSSP_proj_epochs.py)
OH July 2016
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

data_path = '/group/erp/data/caroline.coutout/MEG/data'

# File from which individual subject results will be read, projected timecourses
projections_fstem = '/group/erp/data/caroline.coutout/MEG/data/mSSP_proj_epochs_output'

# file to which grand-mean structure will be written
GM_fname = '/group/erp/data/caroline.coutout/MEG/data/GM_results.pickle'
f_GM = open(GM_fname, 'w')

# lexical and semantic decision tasks
cond_prefix = ['lex', 'sem']


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

# for qsub
if len(sys.argv)>1: # if in parallel mode
    print "Running subject(s) {0} now in PARALLEL mode".format(sys.argv)
    ss_idx = map(int, sys.argv[1:])
    subjs_new = []
    for ii,ss in enumerate(ss_idx): # a bit cumbersome because lists cannot be used as indices
        subjs_new.append(subjs[ss])
    subjs = subjs_new
else:
    print "Running now in SERIAL mode"


TCS_snr_avg_ALL = {} # lex/sem stacked arrays across subjects
TCS_snr_avg_GM = {} # lex/sem grand-mean across subjects
# in the end it will be TCS_snr_avg_GM[cond dict][win_str dict][nc list][ch_type dict]

n_subjs = len(subjs)


### Assembling data structure for all subjects
print "Assembling data structure for all subjects"
for [ssi, ss] in enumerate(subjs):
    print ss[0]

    projections_fname = projections_fstem + '_' + ss[0] + '.pkl'
    f_proj = open(projections_fname, 'r')
    print "Un-pickling results from {0}".format(projections_fname)
    [TCS, TCS_snr, TCS_avg, TCS_snr_avg, epochs] = pickle.load(f_proj)
    f_proj.close()

    if ssi==0: # initalise for first subject

        print "Initialising data structure"

        for [cci, cc] in enumerate(cond_prefix): # for conditions
            print cc
            
            # deepcopy, new structure will be changed
            TCS_snr_avg_ALL[cc] = copy.deepcopy(TCS_snr_avg[ss[0],cc]) # stacked subjects
            TCS_snr_avg_GM[cc] = copy.deepcopy(TCS_snr_avg[ss[0],cc]) # grand-mean across subjects

            n_tcs0, n_times0 = {}, {} # may differ across projection types, needed below

            for proj in TCS_snr_avg_GM[cc]: # dict for latency windows
                print proj

                for [pidx, data] in enumerate(TCS_snr_avg_GM[cc][proj]): # list of data sets

                    for ch_type in data: # channel types
                        
                        data_ch = data[ch_type]
                        n_tcs0[proj], n_times0[proj] = data_ch.shape

                        # initalise matrix containing all subjects per condition/estimator/data set/channe type
                        # 3D np.array: number of data sets x number of subjects x number of time samples
                        TCS_snr_avg_ALL[cc][proj][pidx][ch_type] = np.zeros([n_tcs0[proj],n_subjs,n_times0[proj]]) # stacked data sets
                        TCS_snr_avg_GM[cc][proj][pidx][ch_type] = np.zeros([n_tcs0[proj],n_times0[proj]]) # grand-mean data set

    print "Adding subject {0} to data structure.".format(ss[0])

    for [cci, cc] in enumerate(cond_prefix): # for conditions
        print cc
        
        for proj in TCS_snr_avg[ss[0],cc]: # dict for latency windows
            print proj

            for [pidx, data] in enumerate(TCS_snr_avg[ss[0],cc][proj]): # list of data sets

                for ch_type in data: # channel types
                    
                    data_ch = data[ch_type]
                    n_tcs, n_times = data_ch.shape

                    if [n_tcs, n_times] != [n_tcs0[proj], n_times0[proj]]:
                        print "!!! One subject has different dimensions: {0}".format(ss[0])
                        print "[{0},{1}] instead of [{2}, {3}].".format(n_tcs, n_times, n_tcs0[proj], n_times0[proj])
                    
                    TCS_snr_avg_ALL[cc][proj][pidx][ch_type][:,ssi,:] = data_ch

                    if (ssi==n_subjs-1): # if this was the last subject, let's average...
                        print "Averaging across {0} subjects".format(ssi+1)
                        data_all = TCS_snr_avg_ALL[cc][proj][pidx][ch_type]
                        TCS_snr_avg_GM[cc][proj][pidx][ch_type][:n_tcs,:n_times] = np.mean(data_all, 1) # average across subjects


print "Writing GM results to {0}".format(GM_fname)
pickle.dump([TCS_snr_avg_GM, TCS_snr_avg_ALL, epochs], f_GM)
f_GM.close()