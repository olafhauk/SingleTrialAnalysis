"""
=========================================================
mSSP: average epochs for words, pseudos, liv, nliv, go, nogo
plot and save overlays
OH June 2016
=========================================================

"""

print __doc__

# Russell's addition
import sys
sys.path.append('/imaging/local/software/python_packages/nibabel/1.3.0')
sys.path.append('/imaging/local/software/python_packages/pysurfer/v0.3.1')

import mne
import numpy as np
import matplotlib.pyplot as plt


data_path = '/group/erp/data/caroline.coutout/MEG/data'
fig_path = '/group/erp/data/olaf.hauk/MEG/mSSP/figures'

# lexical and semantic decision tasks
cond_prefix = ['lex', 'sem']

# blocks within condition
block_id = ['A', 'B']

# which channel types to process
pick_meg, pick_eeg, pick_eog = True, True, True


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

# convenience function to plot ERP curves to file
def my_erp_plot(evoked_to_plot, fig_path, ss0, ss2, cc, filestring):
    fig_erp_fname = fig_path + '/ERP_' + ss0 + '_' + ss2 + '_' + cc + filestring
    window_title = "{0} {1} {2} {3}".format(ss0, ss2, cc, filestring)
    # spatial_colors=True (possible error: "overlapping positions", mne_check_eeg_locations)
    fig = evoked_to_plot.plot(gfp=True, window_title=window_title, spatial_colors=True, show=False) 
    fig.savefig(fig_erp_fname)
    plt.close()


epochs_cnt = np.zeros([len(subjs), 2]) # count good epochs per subject and condition
epochs_cnt.astype(int)

for [ssi, ss] in enumerate(subjs):
    print ss[0]
    for [cci, cc] in enumerate(cond_prefix):
        # reading epochs from file
        epo_fname = data_path + '/' + ss[0] + '/' + ss[1] + '/' + cc + '_ds_raw-epo.fif'
        epochs = mne.read_epochs(epo_fname)

        # picks = mne.pick_types(epochs.info, meg=pick_meg, eeg=pick_eeg, eog=pick_eog)

        if cc == 'lex':
            avg_word = epochs['word'].average()
            my_erp_plot(evoked_to_plot=avg_word, fig_path=fig_path, ss0=ss[0], ss2=ss[2], cc=cc, filestring='_word_erp.pdf')
            evoked_fname = data_path + '/' + ss[0] + '/' + ss[1] + '/' + ss[0] + '_' + ss[2] + cc + '_ds_raw_word-ave.fif'
            mne.write_evokeds(evoked_fname, avg_word)

            avg_pseudo = epochs['pseudo'].average()
            my_erp_plot(evoked_to_plot=avg_pseudo, fig_path=fig_path, ss0=ss[0], ss2=ss[2], cc=cc, filestring='_pseudo_erp.pdf')
            evoked_fname = data_path + '/' + ss[0] + '/' + ss[1] + '/' + ss[0] + '_' + ss[2] + cc + '_ds_raw_pseudo-ave.fif'
            mne.write_evokeds(evoked_fname, avg_pseudo)

        elif cc == 'sem':
            avg_liv = epochs['liv'].average()
            my_erp_plot(evoked_to_plot=avg_liv, fig_path=fig_path, ss0=ss[0], ss2=ss[2], cc=cc, filestring='_liv_erp.pdf')
            evoked_fname = data_path + '/' + ss[0] + '/' + ss[1] + '/' + ss[0] + '_' + ss[2] + cc + '_ds_raw_liv-ave.fif'
            mne.write_evokeds(evoked_fname, avg_liv)

            avg_nliv = epochs['nliv'].average()        
            my_erp_plot(evoked_to_plot=avg_nliv, fig_path=fig_path, ss0=ss[0], ss2=ss[2], cc=cc, filestring='_nliv_erp.pdf')
            evoked_fname = data_path + '/' + ss[0] + '/' + ss[1] + '/' + ss[0] + '_' + ss[2] + cc + '_ds_raw_nliv-ave.fif'
            mne.write_evokeds(evoked_fname, avg_nliv)

        avg_go = epochs['go'].average()
        my_erp_plot(evoked_to_plot=avg_go, fig_path=fig_path, ss0=ss[0], ss2=ss[2], cc=cc, filestring='_go_erp.pdf')
        evoked_fname = data_path + '/' + ss[0] + '/' + ss[1] + '/' + ss[0] + '_' + ss[2] + cc + '_ds_raw_go-ave.fif'
        mne.write_evokeds(evoked_fname, avg_go)

        avg_nogo = epochs['nogo'].average()
        my_erp_plot(evoked_to_plot=avg_nogo, fig_path=fig_path, ss0=ss[0], ss2=ss[2], cc=cc, filestring='_nogo_erp.pdf')
        evoked_fname = data_path + '/' + ss[0] + '/' + ss[1] + '/' + ss[0] + '_' + ss[2] + cc + '_ds_raw_nogo-ave.fif'
        mne.write_evokeds(evoked_fname, avg_nogo)

plt.close('all')