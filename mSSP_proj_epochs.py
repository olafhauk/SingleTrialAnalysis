"""
=========================================================
mSSP: compute single trial time courses by projecting
spatial filters on epochs
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
import mSSP_functions
reload(mSSP_functions)
from mSSP_functions import (mSSP_spatfilt_on_epochs, mSSP_avg_proj_timecourses, mSSP_SNR_proj_timecourses_epochs,
                            mSSP_make_beamformer, mSSP_compute_mSSP_estimators, mSSP_mSSP_on_epochs, mSSP_plot_avg_timecourses)

data_path = '/group/erp/data/caroline.coutout/MEG/data'
fig_path = '/group/erp/data/olaf.hauk/MEG/mSSP/figures'

# for pickled results
out_fstem = '/group/erp/data/caroline.coutout/MEG/data/mSSP_proj_epochs_output'

# file with SVD target components for projection (will be loaded)
svd_tar_fname = '/group/erp/data/caroline.coutout/MEG/data/SVD_windows_NoGo.pickle'
f_tar_svd = open(svd_tar_fname, 'r')

# file with SVD target components for eye blinks (will be loaded)
svd_eye_fname = '/group/erp/data/caroline.coutout/MEG/data/SVD_windows_Go.pickle'
f_eye_svd = open(svd_eye_fname, 'r')
t_win_eye = '500-1000ms' # dict index for eye blink topographies (SVD_comps_eye)

# file with SVD noise components per epoch
svd_epo_fstem = '/group/erp/data/caroline.coutout/MEG/data/SVD_epochs'

# Covariance matrix for MLE and BF estimators
cov_fname_MLE_spec = '_-300_0-cov.fif'
cov_fname_BF_spec = '_50_250-cov.fif'
# covariance matrix regualarisation parameter
cov_reg = 0.1

# lexical and semantic decision tasks
cond_prefix = ['lex', 'sem']

# which channel types to process
pick_meg, pick_eeg, pick_eog = True, True, True

# for SNR computation
baseline = [-0.2, 0]

# Maximum number of components in mSSP model
n_comp_mSSP = 6


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

# load SVD target components for projection
# careful: may override previous parameters
# SVD_comps is dict with svd_evoked with topographies for different latency windows
print "Unpickling " + svd_tar_fname
[SVD_comps_tar, SVD_eigen_tar, SVD_var_tar, _, lat_wins, _] = pickle.load(f_tar_svd)

# load eye-blink topographies (to be included in mSSP models)
print "Unpickling " + svd_eye_fname
[SVD_comps_eye, SVD_eigen_eye, SVD_eye_tar, _, lat_wins_eye, _] = pickle.load(f_eye_svd)


for [ssi, ss] in enumerate(subjs):
    print ss[0]

    # Load epoch noise topographies
    svd_epo_fname = svd_epo_fstem + '_' + ss[0] + '.pickle'
    f_svd_epo = open(svd_epo_fname, 'r')
    print "Un-pickling results from {0}".format(svd_epo_fname)
    [SVD_comps_epo, SVD_eigen_epo, SVD_info_epo, SVD_ch_names_epo, _, _, _] = pickle.load(f_svd_epo)

    # results per subject and condition as dict [ss,cc]
    TCS, TCS_snr, TCS_avg, TCS_snr_avg = {}, {}, {}, {}
    # in the end it will be: TCS[ss[0],cc dict][win_str dict][nc list][ch_type dict]

    Epochs = {} # keep epochs for tasks for pickle (lex/sem)

    for [cci, cc] in enumerate(cond_prefix):
        
        # results for each spatial filter type as dict per subject/condition
        TCS[ss[0],cc], TCS_snr[ss[0],cc], TCS_avg[ss[0],cc], TCS_snr_avg[ss[0],cc] = {}, {}, {}, {}

        svd_epochs = SVD_comps_epo[ss[0], cc]['-300-0ms']

        # channel names of SVDs for epochs
        ch_names_SVD_epo = SVD_ch_names_epo[ss[0],cc]

        # reading epochs from file        
        epo_fname = data_path + '/' + ss[0] + '/' + ss[1] + '/' + cc + '_ds_raw-epo.fif'
        print "Reading epochs from {0}".format(epo_fname)
        epochs = mne.read_epochs(epo_fname)
        t = epochs.times
        Epochs[cc] = epochs

        ### mSSP ###

        # initiatlise outputs of mSSP estimator function as dicts
        mSSP_sf, CorrCoef, CondNum = {}, {}, {} # dicts for different target topographies

        # get target topographies from SVD on evoked time windows
        # svd_evoked is dict with latency windows as elements
        for t_win in SVD_comps_tar[ss[0], cc]: # ['100-200ms'] etc.

            # target topography for one time window
            svd_evoked = SVD_comps_tar[ss[0], cc][t_win]

            # channel names of spatial filters depend on target
            ch_names_SVD_tar = svd_evoked.ch_names

            # initiatlise outputs of mSSP estimator function as lists (for different number of components)
            mSSP_sf[t_win], CorrCoef[t_win], CondNum[t_win] = [0]*n_comp_mSSP, [0]*n_comp_mSSP, [0]*n_comp_mSSP

            print "Compute mSSP estimator(s)"
            for nn in range(n_comp_mSSP):
                [mSSP_sf[t_win][nn], CorrCoef[t_win][nn], CondNum[t_win][nn]] = mSSP_compute_mSSP_estimators(
                                                    svd_evoked, svd_epochs, ch_names_SVD_epo, n_comp=nn, topo_add=[],
                                                    topo_add_idx=[0], get_corr=True)  

            print "Applying mSSP estimator to epochs"
            tcs_mSSP = [mSSP_mSSP_on_epochs(epochs, sf, ch_names_SVD_tar) for sf in mSSP_sf[t_win]]

            print "Converting time courses to SNRs"
            tcs_mSSP_snr = [mSSP_SNR_proj_timecourses_epochs(tcs, times=t, baseline=baseline) for tcs in tcs_mSSP]

            print "Averaging time courses"
            tcs_mSSP_avg = [mSSP_avg_proj_timecourses(tcs) for tcs in tcs_mSSP]
            print "Averaging SNR time courses"
            tcs_mSSP_snr_avg = [mSSP_avg_proj_timecourses(tcs, abs=True) for tcs in tcs_mSSP_snr]

            ### Plot
            print "Plotting time courses"

            legend = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

            # raw averaged projections
            for [idx, tcs] in enumerate(tcs_mSSP_avg):
                fig_fname = fig_path + '/' + 'projs_mSSP_' + ss[0] + '_' + cc + '_' + t_win + '_nc' + str(idx)
                titles = [ss[0] + ' ' + cc + ' | mSSP ' + t_win + ' nc' + str(idx)]
                fig = mSSP_plot_avg_timecourses(avgs=tcs, t=t, titles=titles, legend=legend, fname_stem=fig_fname)

            # SNR averaged projections
            for [idx, tcs] in enumerate(tcs_mSSP_snr_avg):
                fig_fname = fig_path + '/' + 'projs_mSSP_SNR_' + ss[0] + '_' + cc + '_' + t_win + '_nc' + str(idx)
                titles = [ss[0] + ' ' + cc + ' | mSSP SNR ' + t_win + ' nc' + str(idx)]
                fig = mSSP_plot_avg_timecourses(avgs=tcs, t=t, titles=titles, legend=legend, fname_stem=fig_fname)

            dict_idx = 'mSSP_' + t_win
            TCS[ss[0],cc][dict_idx] = tcs_mSSP
            TCS_snr[ss[0],cc][dict_idx] = tcs_mSSP_snr
            TCS_avg[ss[0],cc][dict_idx] = tcs_mSSP_avg
            TCS_snr_avg[ss[0],cc][dict_idx] = tcs_mSSP_snr_avg


        ### mSSP with eye blinks ###

        # initiatlise outputs of mSSP estimator function as dicts
        mSSP_sf, CorrCoef, CondNum = {}, {}, {} # dicts for different target topographies

        # get target topographies from SVD on evoked time windows
        # svd_evoked is dict with latency windows as elements
        for t_win in SVD_comps_tar[ss[0], cc]: # ['100-200ms'] etc.

            # target topography for one time window (Evoked)
            svd_evoked = SVD_comps_tar[ss[0], cc][t_win]

            # eye-blink topography for one time window (Evoked)
            svd_eye = SVD_comps_eye[ss[0], cc][t_win_eye]

            # channel names of spatial filters depend on target
            ch_names_SVD_tar = svd_evoked.ch_names

            # initiatlise outputs of mSSP estimator function as lists (for different number of components)
            mSSP_sf[t_win], CorrCoef[t_win], CondNum[t_win] = [0]*n_comp_mSSP, [0]*n_comp_mSSP, [0]*n_comp_mSSP
            # Note: topo_add_idx contains indices, not the number of components
            print "Compute mSSP estimator(s) with eye blink topography."
            for nn in range(n_comp_mSSP):
                [mSSP_sf[t_win][nn], CorrCoef[t_win][nn], CondNum[t_win][nn]] = mSSP_compute_mSSP_estimators(
                                                    svd_evoked, svd_epochs, ch_names_SVD_epo, n_comp=nn, topo_add=svd_eye,
                                                    topo_add_idx=[0], get_corr=True)  

            print "Applying mSSP estimator to epochs"
            tcs_mSSP = [mSSP_mSSP_on_epochs(epochs, sf, ch_names_SVD_tar) for sf in mSSP_sf[t_win]]

            print "Converting time courses to SNRs"
            tcs_mSSP_snr = [mSSP_SNR_proj_timecourses_epochs(tcs, times=t, baseline=baseline) for tcs in tcs_mSSP]

            print "Averaging time courses"
            tcs_mSSP_avg = [mSSP_avg_proj_timecourses(tcs) for tcs in tcs_mSSP]
            print "Averaging SNR time courses"
            tcs_mSSP_snr_avg = [mSSP_avg_proj_timecourses(tcs, abs=True) for tcs in tcs_mSSP_snr]

            ### Plot
            print "Plotting time courses"

            legend = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

            # raw averaged projections
            for [idx, tcs] in enumerate(tcs_mSSP_avg):
                fig_fname = fig_path + '/' + 'projs_mSSP_eye_' + ss[0] + '_' + cc + '_' + t_win + '_nc' + str(idx)
                titles = [ss[0] + ' ' + cc + ' | mSSP eye ' + t_win + ' nc' + str(idx)]
                fig = mSSP_plot_avg_timecourses(avgs=tcs, t=t, titles=titles, legend=legend, fname_stem=fig_fname)

            # SNR averaged projections
            for [idx, tcs] in enumerate(tcs_mSSP_snr_avg):
                fig_fname = fig_path + '/' + 'projs_mSSP_eye_SNR_' + ss[0] + '_' + cc + '_' + t_win + '_nc' + str(idx)
                titles = [ss[0] + ' ' + cc + ' | mSSP eye SNR ' + t_win + ' nc' + str(idx)]
                fig = mSSP_plot_avg_timecourses(avgs=tcs, t=t, titles=titles, legend=legend, fname_stem=fig_fname)

            dict_idx = 'mSSP_eye_' + t_win
            TCS[ss[0],cc][dict_idx] = tcs_mSSP
            TCS_snr[ss[0],cc][dict_idx] = tcs_mSSP_snr
            TCS_avg[ss[0],cc][dict_idx] = tcs_mSSP_avg
            TCS_snr_avg[ss[0],cc][dict_idx] = tcs_mSSP_snr_avg    


        # ### get peak latencies
        # # ep_dict = tcs_mSSP['50-200ms'][0]
        # # peak_lat = mSSP_peak_lats_proj_epochs(ep_dict, t, [0.05,0.2])
        # # dathist = np.histogram(data)
        # # plt.bar(1000*dathist[1][:-1], dathist[0], width=10)

        
        ### BEAMFORMERS ###

        print "MLE/Beamformer(s)"

        MLE_sf = {} # MLE for different targets as dict
        BF_sf = {} # Beamformer for different targets as dict

        # Covariance matrix file names
        cov_fname_MLE = data_path + '/' + ss[0] + '/' + ss[1] + '/' + ss[0] + '_' + ss[2] + '_' + cc + cov_fname_MLE_spec
        cov_fname_BF = data_path + '/' + ss[0] + '/' + ss[1] + '/' + ss[0] + '_' + ss[2] + '_' + cc + cov_fname_BF_spec

        # Covariance matrix for MLE estimator (baseline interval)
        print "Reading covariance matrix {0}".format(cov_fname_MLE)
        cov_MLE = mne.cov.read_cov(cov_fname_MLE)

        # Covariance matrix for BF estimator (post-stimulus interval)
        print "Reading covariance matrix {0}".format(cov_fname_BF)
        cov_BF = mne.cov.read_cov(cov_fname_BF)

        for t_win in SVD_comps_tar[ss[0], cc]: # ['100-200ms'] etc.

            # target topography for one time window
            svd_evoked = SVD_comps_tar[ss[0], cc][t_win]

            print "Creating MLE/beamformers"
            # output of function is Evoked object
            MLE_sf[t_win] = mSSP_make_beamformer(cov_MLE, svd_evoked, regpars=cov_reg)
            BF_sf[t_win] = mSSP_make_beamformer(cov_BF, svd_evoked, regpars=cov_reg)
        
            print "Projecting SVD components on epochs"
            tcs_MLE = mSSP_spatfilt_on_epochs(epochs, MLE_sf[t_win])
            tcs_BF = mSSP_spatfilt_on_epochs(epochs, BF_sf[t_win])

            # convert epoch time courses to SNRs
            print "Converting time courses to SNRs"
            tcs_MLE_snr = mSSP_SNR_proj_timecourses_epochs(tcs_MLE, times=t, baseline=baseline)
            tcs_BF_snr = mSSP_SNR_proj_timecourses_epochs(tcs_BF, times=t, baseline=baseline)

            # average across epoch arrays
            print "Averaging time courses"
            tcs_MLE_avg = mSSP_avg_proj_timecourses(tcs_MLE)
            tcs_BF_avg = mSSP_avg_proj_timecourses(tcs_BF)
            print "Averaging SNR time courses"
            tcs_MLE_snr_avg = mSSP_avg_proj_timecourses(tcs_MLE_snr, abs=True) # average absolute values for SNRs
            tcs_BF_snr_avg = mSSP_avg_proj_timecourses(tcs_BF_snr, abs=True)

            ### Plot
            print "Plotting time courses"
            
            legend = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        
            # MLE, raw averaged projections
            tcs = tcs_MLE_avg
            fig_fname = fig_path + '/' + 'projs_MLE_' + ss[0] + '_' + cc + '_' + t_win
            titles = [ss[0] + ' ' + cc + ' | MLE ' + t_win]
            fig = mSSP_plot_avg_timecourses(avgs=tcs, t=t, titles=titles, legend=legend, fname_stem=fig_fname)

            # SNR averaged projections
            tcs = tcs_MLE_snr_avg
            fig_fname = fig_path + '/' + 'projs_MLE_SNR_' + ss[0] + '_' + cc + '_' + t_win + '.pdf'
            titles = [ss[0] + ' ' + cc + ' | MLE SNR ' + t_win]
            fig = mSSP_plot_avg_timecourses(avgs=tcs, t=t, titles=titles, legend=legend, fname_stem=fig_fname)

            # BF, raw averaged projections
            tcs = tcs_BF_avg
            fig_fname = fig_path + '/' + 'projs_BF_' + ss[0] + '_' + cc + '_' + t_win + '.pdf'
            titles = [ss[0] + ' ' + cc + ' | BF ' + t_win]
            fig = mSSP_plot_avg_timecourses(avgs=tcs, t=t, titles=titles, legend=legend, fname_stem=fig_fname)

            # SNR averaged projections
            tcs = tcs_BF_snr_avg
            fig_fname = fig_path + '/' + 'projs_BF_SNR_' + ss[0] + '_' + cc + '_' + t_win + '.pdf'
            titles = [ss[0] + ' ' + cc + ' | BF SNR ' + t_win]
            fig = mSSP_plot_avg_timecourses(avgs=tcs, t=t, titles=titles, legend=legend, fname_stem=fig_fname)

            dict_idx = 'MLE_' + t_win
            TCS[ss[0],cc][dict_idx] = [tcs_MLE] # as list to make compatible with mSSP
            TCS_snr[ss[0],cc][dict_idx] = [tcs_MLE_snr]
            TCS_avg[ss[0],cc][dict_idx] = [tcs_MLE_avg]
            TCS_snr_avg[ss[0],cc][dict_idx] = [tcs_MLE_snr_avg]

            dict_idx = 'BF_' + t_win
            TCS[ss[0],cc][dict_idx] = [tcs_BF] # as list to make compatible with mSSP
            TCS_snr[ss[0],cc][dict_idx] = [tcs_BF_snr]
            TCS_avg[ss[0],cc][dict_idx] = [tcs_BF_avg]
            TCS_snr_avg[ss[0],cc][dict_idx] = [tcs_BF_snr_avg]

    out_fname = out_fstem + '_' + ss[0] + '.pkl'
    f_out = open(out_fname, 'w')
    print "Pickling results to {0}".format(out_fname)
    pickle.dump([TCS, TCS_snr, TCS_avg, TCS_snr_avg, Epochs], f_out)
    f_out.close()

plt.close('all')