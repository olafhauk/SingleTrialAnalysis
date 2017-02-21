
###
# To Do:
###

# check mne.cov.compute_whitener 


###
# This works so far:
###

# functions in mSSP_functions.py
# qsub via mSSP_submit_qsub.sh

# rename files, check fiff-files
import mSSP_check_eeg_electrodes
mSSP_check_eeg_electrodes

# read and plot events
import mSSP_visualise_events
mSSP_visualise_events

# epoch data (via qsub)
import mSSP_epochs
mSSP_epochs

# average epoched data
import mSSP_average_epochs
mSSP_average_epochs

# extract SVD components in time windows
import mSSP_SVD_evoked_NoGo    # for target topographies
import mSSP_SVD_evoked_Go      # for eye blink topographies

# compute SVDs for individual epochs, for mSSP estimators
import mSSP_SVD_epochs

# compute covariance matrices, for MLE and BF (via qsub)
import mSSP_make_covmat

# project spatial filters on epochs (via qsub)
import mSSP_proj_epochs

# compute grand-mean time courses
import mSSP_make_GM_timecourses

# plot grand-mean time courses to PDF
import mSSP_plot_GM_timecoures

# get peak latencies (for individual estimator)
import mSSP_get_latencies.py

# make figures for SNL poster (topomaps, histograms, curves)
import mSSP_make_figures_4pubs.py
