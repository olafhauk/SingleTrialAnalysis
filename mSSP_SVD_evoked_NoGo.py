"""
=========================================================
mSSP: compute SVD within latency windows for evoked data
for NoGo condition (target topographies)
save first SVDs as dict containing all subjects
SVD applied to channel types separately
OH June 2016
=========================================================

"""
# Olaf Hauk, 2014

print __doc__

# Russell's addition
import sys
# sys.path.append('/imaging/local/software/python_packages/nibabel/1.3.0')
# sys.path.append('/imaging/local/software/python_packages/pysurfer/v0.3.1')
sys.path.insert(1, '/imaging/local/software/anaconda/2.4.1/2/lib/python2.7/site-packages/sklearn/')
sys.path.insert(1, '/imaging/local/software/anaconda/2.4.1/2/lib/python2.7/site-packages/pysurfer/')
sys.path.insert(1, '/imaging/local/software/anaconda/2.4.1/2/lib/python2.7/site-packages/nibabel/')

import mne
import numpy as np
import matplotlib.pyplot as plt

import pickle

# my functions
import mSSP_functions
from mSSP_functions import mSSP_SVD_on_Evoked, mSSP_var_from_eigen
# reload(mSSP_functions) # in case it has changed
from matplotlib.backends.backend_pdf import PdfPages

# where data are
data_path = '/group/erp/data/caroline.coutout/MEG/data'

# where figures with topos will be
fig_path = '/group/erp/data/olaf.hauk/MEG/mSSP/figures'

# write SVD results for all subject to one file:
svd_fname = '/group/erp/data/caroline.coutout/MEG/data/SVD_windows_NoGo.pickle'
f_svd = open(svd_fname, 'w')

# lexical and semantic decision tasks
cond_prefix = ['lex', 'sem']

lat_wins = [[0.05,0.2], [0.15, 0.2], [0.25,0.4]] # for target topographies

# which channel types to process
pick_meg, pick_eeg, pick_eog = True, True, True

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

# dict for SVDs across all subjects and conditions
SVD_comps = {}
# dicts for eigenvalues and variance for subjects and conditions
SVD_eigen = {}
SVD_var = {}

for [ssi, ss] in enumerate(subjs):
    print ss[0]
    for [cci, cc] in enumerate(cond_prefix):
               
        # read NoGo evoked data       
        evoked_fname = data_path + '/' + ss[0] + '/' + ss[1] + '/' + ss[0] + '_' + ss[2] + cc + '_ds_raw_nogo-ave.fif'
        evoked_nogo = mne.read_evokeds(evoked_fname, baseline=baseline)

        # get first n_comp SVD components within latency window for channel types separately
        svd_comp, svd_s = mSSP_SVD_on_Evoked(evoked_nogo[0], lat_wins, n_comp=n_comp)

        # compute variances for eigenvalues
        svd_var = mSSP_var_from_eigen(svd_s, n_comp=n_comp)    
        
        # make evoked object with SVD components per subject
        svd_evoked = {}
        info = evoked_nogo[0].info
        for [twi, tw] in enumerate(lat_wins):
            evoked_now = mne.EvokedArray(svd_comp[twi], info, tmin=0.0)            
            win_str = str(int(1000*tw[0])) + '-' + str(int(1000*tw[1])) + 'ms'
            svd_evoked[win_str] = evoked_now
            svd_evoked[win_str].comment = win_str

            fig_fname = fig_path + '/' + 'SVDcomps_' + ss[0] + '_' + cc + '_' + win_str + '_NoGo.pdf'
            with PdfPages(fig_fname) as pdf:
                pdftitle = 'NoGo - ' + win_str
                pdf.attach_note(pdftitle)
                svd_evoked[win_str].plot_topomap(ch_type='mag', show=False)
                pdf.savefig()
                plt.close()
                plt.bar(range(n_comp), svd_var[twi]['mag'])
                pdf.savefig()
                plt.close()
                svd_evoked[win_str].plot_topomap(ch_type='grad', show=False)
                pdf.savefig()
                plt.close()
                plt.bar(range(n_comp), svd_var[twi]['grad'])
                pdf.savefig()
                plt.close()
                svd_evoked[win_str].plot_topomap(ch_type='eeg', show=False)
                pdf.savefig()
                plt.close()
                plt.bar(range(n_comp), svd_var[twi]['eeg'])
                pdf.savefig()
                plt.close()
        
        SVD_comps[ss[0], cc] = svd_evoked
        SVD_eigen[ss[0], cc] = svd_s
        SVD_var[ss[0], cc] = svd_var
    
# write dict with SVDs to file
print "Pickling to " + svd_fname
pickle.dump([SVD_comps, SVD_eigen, SVD_var, subjs, lat_wins, baseline], f_svd)

f_svd.close()

# matplotlib.mlab.close(all=True)