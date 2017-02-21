"""
=========================================================
mSSP: read raw data for each subject and condition
concatenate blocks per condition
plot events (mne.viz.plot_events) and save figure
OH June 2016
=========================================================

"""
# Olaf Hauk, 2014

print __doc__

# Russell's addition
import sys
sys.path.append('/imaging/local/software/python_packages/nibabel/1.3.0')
sys.path.append('/imaging/local/software/python_packages/pysurfer/v0.3.1')

from os.path import isfile

import mne
import numpy as np


data_path = '/group/erp/data/caroline.coutout/MEG/data'
fig_path = '/group/erp/data/olaf.hauk/MEG/mSSP/figures'


# lexical and semantic decision tasks
cond_prefix = ['lex', 'sem']

block_id = ['A', 'B']

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

for ss in subjs:
    print ss[0]
    for cc in cond_prefix:
        raws = [] # blocks to be concatenated
        for bb in block_id:
            print bb
            raw_fname = data_path + '/' + ss[0] + '/' + ss[1] + '/' + cc + '_'  + bb + '_ds_raw.fif'
            
            check_file = isfile(raw_fname)

            if check_file:
                print raw_fname
            else:
                print "No!{0}".format(raw_fname)

            raw_now = mne.io.read_raw_fif(raw_fname, preload=True)

            # HPI channels may cause trouble if movecomp failed for one block
            raw_now.drop_channels(HPIs)
                 
            # append block per condition and subject     
            raws.append( raw_now )
            
        raw = mne.concatenate_raws(raws, preload=None)

        events = mne.find_events(raw, stim_channel='STI101', shortest_event=1)

        # plot event information for combined blocks for all events
        fig = mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp, event_id=None, show=False)

        fig_fname = fig_path + '/' + 'events_' + ss[0] + '_' + cc +'.pdf'
        fig.savefig(fig_fname)