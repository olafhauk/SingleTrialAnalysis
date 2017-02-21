"""
=========================================================
mSSP: Make plots (e.g. topographies) for Poster (e.g. SNL)
=========================================================

"""
# Olaf Hauk, 2016

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
from scipy import stats
import matplotlib.pyplot as plt

import pickle

# output figures for SNL poster
fig_path = '/home/olaf/Posters/NLC London 2016/figures'

###
### Plot target topography for one subject

# # write SVD results for all subject to one file:
# svd_fname = '/group/erp/data/caroline.coutout/MEG/data/SVD_windows_NoGo.pickle'
# f_svd = open(svd_fname, 'r')

# [SVD_comps, SVD_eigen, SVD_var, subjs, lat_wins, baseline] = pickle.load(f_svd)

# ev = SVD_comps['meg10_0047', 'lex']['50-200ms']

# ev.plot_topomap(ch_type='eeg', show=True, vmin=250000, vmax=-250000, times=[0.001], size=10)


###
### Plot noise topographies from epoch baseline

# # write SVD results for all subject to one file:
# svd_fstem = '/group/erp/data/caroline.coutout/MEG/data/SVD_epochs'

# svd_fname = svd_fstem + '_meg10_0047.pickle'
# f_svd = open(svd_fname, 'r')

# print "Unpickling " + svd_fname
# [SVD_comps, SVD_eigen, info_epo, ch_names_epo, subjs, lat_wins, baseline] = pickle.load(f_svd)

# # where data are
# data_path = '/group/erp/data/caroline.coutout/MEG/data'

# epo_fname = data_path + '/meg10_0047/100317/lex_ds_raw-epo.fif'

# epochs = mne.read_epochs(epo_fname)

# evoked_now = mne.EvokedArray(SVD_comps['meg10_0047', 'lex']['-300-0ms'][0,:,:], epochs.info, tmin=0.0)

# evoked_now.plot_topomap(ch_type='eeg', show=True, times=[0.0, 0.004, 0.008, 0.012, 0.016], size=3)


###
### GM curves
# use figs[].show() from mSSP_plot_GM_timecourses.py

### Latency histograms
fname_out = '/group/erp/data/caroline.coutout/MEG/data/latencies_results.pickle'
f_out = open(fname_out, 'r')
print "Unpickling " + fname_out
[peak_lats_avg, peak_lats_epo, subjs] = pickle.load(f_out)
f_out.close()

# latency window in which peak was determined
# estimator was specified in mSSP_get_latencies.py
win_str = '50-250ms'

# peak_lats_avg['lex']['word'][win_str]['mag']['cof']

n_subjs = len(peak_lats_epo['lex']['word'][win_str]['mag']['cof'])

# concatenate arrays across all subjects
data = dict.fromkeys(peak_lats_epo) # lex/sem

for task in data:
    data[task] = dict.fromkeys(peak_lats_epo[task]) # word, pseudo, liv, nliv, go, nogo

    for cond in data[task]:

        data[task][cond] = dict.fromkeys(peak_lats_epo[task][cond][win_str]) # channel types

        for ch_type in data[task][cond]:

            data[task][cond][ch_type] = dict.fromkeys(peak_lats_epo[task][cond][win_str][ch_type])

            for lat_meas in data[task][cond][ch_type]:

                data[task][cond][ch_type][lat_meas] = np.concatenate( np.array( peak_lats_epo[task][cond][win_str][ch_type]['cof'] ) )



### t-tests
contrasts = [
                [['lex', 'word'], ['lex', 'pseudo']],
                [['lex', 'go'], ['lex', 'nogo']],
                [['sem', 'liv'], ['sem', 'nliv']],
                [['sem', 'go'], ['sem', 'nogo']],
                [['lex', 'go'], ['sem', 'go']],
                [['lex', 'nogo'], ['sem', 'nogo']]
]

for con in contrasts:
    print con
    a = peak_lats_avg[con[0][0]][con[0][1]][win_str]['grad']['cof']
    b = peak_lats_avg[con[1][0]][con[1][1]][win_str]['grad']['cof']
    t_res = stats.ttest_rel(a,b)
    print t_res

print "Go vs NoGo"
datlist = []
datlist.append( peak_lats_avg['lex']['go'][win_str]['grad']['cof'] )
datlist.append( peak_lats_avg['sem']['go'][win_str]['grad']['cof'] )
a = np.concatenate( datlist )
datlist = []
datlist.append( peak_lats_avg['lex']['nogo'][win_str]['grad']['cof'] )
datlist.append( peak_lats_avg['sem']['nogo'][win_str]['grad']['cof'] )
b = np.concatenate( datlist )

t_res = stats.ttest_rel(a,b)
print t_res

print "Lex vs Sem"
datlist = []
datlist.append( peak_lats_avg['lex']['go'][win_str]['grad']['cof'] )
datlist.append( peak_lats_avg['lex']['nogo'][win_str]['grad']['cof'] )
a = np.concatenate( datlist )
datlist = []
datlist.append( peak_lats_avg['sem']['go'][win_str]['grad']['cof'] )
datlist.append( peak_lats_avg['sem']['nogo'][win_str]['grad']['cof'] )
b = np.concatenate( datlist )

t_res = stats.ttest_rel(a,b)
print t_res



# word categories
dat = []
myplot = {}

myplot['cats'] = plt.figure()

dat.append(1000.*data['lex']['word']['grad']['cof'])
dat.append(1000.*data['lex']['pseudo']['grad']['cof'])
dat.append(1000.*data['sem']['liv']['grad']['cof'])
dat.append(1000.*data['sem']['nliv']['grad']['cof'])

legend = ['words', 'pseudowords', 'living', 'non-living']

plt.hist(dat, 20)
plt.legend(legend, loc='upper left')

fname = fig_path + '/histograms_cats_' + win_str + '.svg'
plt.savefig(fname)


# Go/NoGo
dat = []

myplot['gonogo'] = plt.figure()

dat.append(1000.*data['lex']['go']['grad']['cof'])
dat.append(1000.*data['lex']['nogo']['grad']['cof'])
dat.append(1000.*data['sem']['go']['grad']['cof'])
dat.append(1000.*data['sem']['nogo']['grad']['cof'])

legend = ['lex go', 'lex nogo', 'sem go', 'sem nogo']

plt.hist(dat, 20)
plt.legend(legend, loc='upper left')

fname = fig_path + '/histograms_gonogo_' + win_str + '.svg'
plt.savefig(fname)

# dathist = np.histogram(dat)

# plt.hist(1000*dat, 20, alpha=0.5, label='words')


# plt.bar(1000*dathist[1][:-1], dathist[0], width=20)
