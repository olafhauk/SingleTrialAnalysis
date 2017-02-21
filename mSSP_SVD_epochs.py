"""
=========================================================
mSSP: compute SVD within latency windows for epoched data
save first SVDs as dict containing all subjects
SVD applied to channel types separately
OH June 2016
=========================================================

"""
# Olaf Hauk, 2014

print __doc__

# Russell's addition
import sys
sys.path.append('/imaging/local/software/python_packages/nibabel/1.3.0')
sys.path.append('/imaging/local/software/python_packages/pysurfer/v0.3.1')

import matplotlib
matplotlib.use('Agg') # possibly for running on cluster

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import mne
import numpy as np

import pickle

# my functions
import mSSP_functions
from mSSP_functions import mSSP_SVD_on_Epochs

# reload(mSSP_functions) # in case it has changed

# where data are
data_path = '/group/erp/data/caroline.coutout/MEG/data'

# where figures with topos will be
fig_path = '/group/erp/data/olaf.hauk/MEG/mSSP/figures'

# write SVD results for all subject to one file:
svd_fstem = '/group/erp/data/caroline.coutout/MEG/data/SVD_epochs'

# lexical and semantic decision tasks
cond_prefix = ['lex', 'sem']

lat_wins = [[-0.3,0.], [0.05,0.2]] # latency windows for SVD

# which channel types to process
pick_meg, pick_eeg, pick_eog = True, True, False

# interval for baseline correction
baseline = (None, 0.0)

# number of SVD components per latency window
n_comp = 10

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


# dict for SVDs across all subjects and conditions
SVD_comps = {}
# dicts for eigenvalues and variance for subjects and conditions
SVD_eigen = {}
SVD_var = {}
# keep info for epochs
info_epo = {}
ch_names_epo = {}

for [ssi, ss] in enumerate(subjs):
    print ss[0]
    for [cci, cc] in enumerate(cond_prefix):        
       
        epo_fname = data_path + '/' + ss[0] + '/' + ss[1] + '/' + cc + '_ds_raw-epo.fif'
        print "Reading epochs from {0}".format(epo_fname)
        epochs = mne.read_epochs(epo_fname)
        epochs = epochs.pick_types(meg=pick_meg, eeg=pick_eeg, eog=pick_eog)
        info_epo[ss[0], cc] = epochs.info
        ch_names_epo[ss[0], cc] = epochs.ch_names

        # get first n_comp SVD components within latency window for channel types separately
        svd_comp, svd_s = mSSP_SVD_on_Epochs(epochs, lat_wins, n_comp=n_comp)


#         # compute variances for eigenvalues
# TO BE WRITTEN
#         svd_var = mSSP_var_from_eigen(svd_s, n_comp=n_comp)    
        
        # make evoked object with SVD components per subject
        svd_epochs = {}
        info = epochs.info
        for [twi, tw] in enumerate(lat_wins):
            
            win_str = str(int(1000*tw[0])) + '-' + str(int(1000*tw[1])) + 'ms'
            svd_epochs[win_str] = svd_comp[twi]

            # # Plot topographies per subject and condition for some epochs into PDFs
            # fig_fname = fig_path + '/' + 'SVDcomps_epochs_' + ss[0] + '_' + cc + '_' + win_str + '.pdf'
            # print "Saving figure to {0}.".format(fig_fname)
            # with PdfPages(fig_fname) as pdf:
            #     for ee in range(0,len(epochs),50):
                    
            #         # get data for plotting into evoked object
            #         evoked_now = mne.EvokedArray(svd_comp[twi][ee,:,:], info, tmin=0.0)
            #         evoked_now.comment = win_str

            #         pdftitle = str(int(tw[0])) + '-' + str(int(tw[1])) + ' ms - Epoch: ' + str(ee)
            #         pdf.attach_note(pdftitle)
            #         evoked_now.plot_topomap(ch_type='mag', show=False)
            #         pdf.savefig()
            #         plt.close()
            #         # plt.bar(range(n_comp), svd_var[twi]['mag'])
            #         # pdf.savefig()
            #         # plt.close()
            #         evoked_now.plot_topomap(ch_type='grad', show=False)
            #         pdf.savefig()
            #         plt.close()
            #         # plt.bar(range(n_comp), svd_var[twi]['grad'])
            #         # pdf.savefig()
            #         # plt.close()
            #         evoked_now.plot_topomap(ch_type='eeg', show=False)
            #         pdf.savefig()
            #         plt.close()
            #         # plt.bar(range(n_comp), svd_var[twi]['eeg'])
            #         # pdf.savefig()
            #         # plt.close()
        
        SVD_comps[ss[0], cc] = svd_epochs
        SVD_eigen[ss[0], cc] = svd_s
        # SVD_var[ss[0], cc] = svd_var
    
    # write dict with SVDs to file
    svd_fname = svd_fstem + '_' + ss[0] + '.pickle'
    f_svd = open(svd_fname, 'w')
    print "Pickling results to {0}".format(svd_fname)
    pickle.dump([SVD_comps, SVD_eigen, info_epo, ch_names_epo, subjs, lat_wins, baseline], f_svd)

    f_svd.close()

# matplotlib.mlab.close(all=True)