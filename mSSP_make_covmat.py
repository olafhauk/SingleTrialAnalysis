"""
=========================================================
mSSP: make covariance matrices from epoched baselines
OH June 2016
=========================================================

"""
# Olaf Hauk, 2014

print __doc__

import sys
# for qsub: check matplotlib.use('Agg'), plt.ion(), plt.show(), do_show
sys.path.insert(1, '/imaging/local/software/anaconda/2.4.1/2/lib/python2.7/site-packages/sklearn/')
sys.path.insert(1, '/imaging/local/software/anaconda/2.4.1/2/lib/python2.7/site-packages/pysurfer/')
sys.path.insert(1, '/imaging/local/software/anaconda/2.4.1/2/lib/python2.7/site-packages/nibabel/')
sys.path.insert(1, '/imaging/local/software/mne_python/v0.13/')

import matplotlib
matplotlib.use('Agg') # possibly for running on cluster

import mne
import numpy as np
import pickle

print "Version:"
print mne.__version__

data_path = '/group/erp/data/caroline.coutout/MEG/data'
fig_path = '/group/erp/data/olaf.hauk/MEG/mSSP/figures'

# lexical and semantic decision tasks
cond_prefix = ['lex', 'sem']

# which channel types to process
pick_meg, pick_eeg, pick_eog = True, True, True

#time interval of interest
# tmin, tmax = -0.3, 0.0 # for MLE
tmin, tmax = 0.05, 0.250 # for BF

# interval for baseline correction
baseline = (None, 0.0)

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


for [ssi, ss] in enumerate(subjs):
    print ss[0]
    for [cci, cc] in enumerate(cond_prefix):            

        # reading epochs from file        
        epo_fname = data_path + '/' + ss[0] + '/' + ss[1] + '/' + cc + '_ds_raw-epo.fif'
        epochs = mne.read_epochs(epo_fname)

        noise_cov = mne.compute_covariance(epochs, tmin=tmin, tmax=tmax, method='auto')

        tmin_str = str(int(1000*tmin)) # time interval as string in ms
        tmax_str = str(int(1000*tmax))
        cov_fname = data_path + '/' + ss[0] + '/' + ss[1] + '/' + ss[0] + '_' + ss[2] + '_' + cc + '_' + tmin_str + '_' + tmax_str + '_auto-cov.fif'
        print "Writing covariance matrix to {0}".format(cov_fname)
        mne.write_cov(cov_fname, noise_cov)

        fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, epochs.info, show=False)

        fig_cov_fname = fig_path + '/' + ss[0] + '_' + ss[2] + '_' + cc + '_' + tmin_str + '_' + tmax_str + '_auto_cov.pdf'
        fig_cov.savefig(fig_cov_fname)

        fig_cov_fname = fig_path + '/' + ss[0] + '_' + ss[2] + '_' + cc + '_' + tmin_str + '_' + tmax_str + '_auto_covspec.pdf'
        fig_spectra.savefig(fig_cov_fname)

        matplotlib.pyplot.close('all')