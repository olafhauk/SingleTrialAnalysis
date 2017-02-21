"""
=========================================================
mSSP: read raw data for each subject and condition
concatenate blocks per condition
compute and save epochs
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
sys.path.insert(1, '/imaging/local/software/mne_python/v0.12/')

import matplotlib
matplotlib.use('Agg') # possibly for running on cluster

import mne
import numpy as np


data_path = '/group/erp/data/caroline.coutout/MEG/data'
fig_path = '/group/erp/data/olaf.hauk/MEG/mSSP/figures'

# lexical and semantic decision tasks
cond_prefix = ['lex', 'sem']

# blocks within condition
block_id = ['A', 'B']

# which channel types to process
pick_meg, pick_eeg, pick_eog = True, True, True

# projector delay in s
stim_delay = 0.034 

#time interval of interest
tmin, tmax = -0.3, 1.0

# interval for baseline correction
baseline = (None, 0.0)

# artefact thresholds specified in loop below
# latency range for which rejection to be applied
reject_tmin, reject_tmax = -0.3, 0.1

# make bandpass filter (Butterworth 4-th order)
iir_params = dict(order=4, ftype='butter')
# iir_params = mne.filter.construct_iir_filter(iir_params, [0.1, 30], None, 1000, 'bandpass', return_copy=False)

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

# HPI channels can cause troube when movecomp didn't work for some files
HPIs = ['CHPI001', 'CHPI002', 'CHPI003', 'CHPI004', 'CHPI005', 'CHPI006', 'CHPI007', 'CHPI008', 'CHPI009']

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

epochs_cnt = np.zeros([len(subjs), 2]) # count good epochs per subject and condition
epochs_cnt.astype(int)

for [ssi, ss] in enumerate(subjs):
    print ss[0]
    for [cci, cc] in enumerate(cond_prefix):

        # events for epoching
        # code wds/psd/liv/nliv as well as go/nogo - a little wasteful but convenient later
        if ss[3] == 1:
            if cc == 'lex':
                event_id = {'word': 1, 'pseudo': 2, 'go': 1, 'nogo': 2}
            elif cc == 'sem':
                event_id = {'liv': 1, 'nliv': 2, 'go': 1, 'nogo': 2}
        elif ss[3] == 2:
            if cc == 'lex':
                event_id = {'word': 1, 'pseudo': 2, 'go': 2, 'nogo': 1}
            elif cc == 'sem':
                event_id = {'liv': 1, 'nliv': 2, 'go': 2, 'nogo': 1}


        raws = [] # blocks to be concatenated
        for bb in block_id:
            print bb
            # read and band-pass-filter fiff-data from one block
            raw_fname = data_path + '/' + ss[0] + '/' + ss[1] + '/' + cc + '_'  + bb + '_ds_raw.fif'
            raw_tmp = mne.io.read_raw_fif(raw_fname, preload=True)            
            # HPI channels may cause trouble if movecomp failed for one block
            raw_tmp.drop_channels(HPIs)
            raw_tmp.filter(l_freq=.1, h_freq=30.0, method='iir', iir_params=iir_params) # iir_params defined above
            raws.append( raw_tmp )
            
        # concatenate files for different blocks
        print "Concatenating files"
        raw = mne.concatenate_raws(raws, preload=True)

        # get event information and add projector delay
        print "Finding events"
        events = mne.find_events(raw, stim_channel='STI101', shortest_event=1)
        events[:,0] += np.int(np.round( raw.info['sfreq']*stim_delay )) # deal with stim delay

        picks = mne.pick_types(raw.info, meg=pick_meg, eeg=pick_eeg, eog=pick_eog)

        # artefact rejection threshold
        if (ss[0]=='meg10_0091') or (ss[0]=='meg10_0088'): # noisy EOG
            reject = dict(mag=4e-12, grad=200e-12, eeg=120e-6, eog=1.)
        else: # for everyone else
            reject = dict(mag=2e-12, grad=200e-12, eeg=100e-6, eog=120e-6)
        # reject = dict(mag=2e-12, grad=200e-12, eog=120e-6)

        # epoching
        print "Epoching"
        epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin,
                    tmax=tmax, reject=reject, reject_tmin=reject_tmin, reject_tmax=reject_tmax, picks=picks)
        epochs.drop_bad() # leaves only good epochs
        
        # keep track of number of good epochs
        epochs_cnt[ssi, cci] = len(epochs)
        print "Number of epochs (after drop): " + str(epochs_cnt[ssi, cci])

        # saving epochs to file
        epo_fname = data_path + '/' + ss[0] + '/' + ss[1] + '/' + cc + '_ds_raw-epo.fif'
        print "Saving epochs to " + epo_fname
        epochs.save(epo_fname)

        # writing figure with channels stats for epoch rejection        
        fig_epo = epochs.plot_drop_log(show=False)
        fig_epo_fname = fig_path + '/' + ss[0] + '_' + ss[2] + '_' + cc + '_drop_log.pdf'        
        print "Saving epochs drop log figure to " + fig_epo_fname
        fig_epo.savefig(fig_epo_fname)

epoch_cnt_file.close()

print "Done"

# matplotlib.mlab.close(all=True)