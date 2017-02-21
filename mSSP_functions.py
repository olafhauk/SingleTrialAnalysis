from mne import pick_channels, pick_channels_cov, EvokedArray
from mne.utils import logger, verbose
from mne.cov import prepare_noise_cov
from mne.cov import regularize as cov_regularize
import numpy as np
from scipy.linalg import pinv, svd
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def mSSP_get_channel_types(ch_names):
    """ Get indices for channel types (mag, grad, eeg) from ch_names

    Parameters
    ----------
    ch_names: list of strings
        list of channel names (e.g. from evoked, epoch or raw object) 

    Returns
    -------
    chantypes_idx: dict of lists
        entries of dict are lists of indices for different channel types
    """
        
    MAG_idx = [cc for cc in range(len(ch_names)) if (ch_names[cc][:3]=='MEG'
                                                and ch_names[cc][-1:]=='1')]
    GRA_idx = [cc for cc in range(len(ch_names)) if (ch_names[cc][:3]=='MEG'
                    and (ch_names[cc][-1:]=='2' or ch_names[cc][-1:]=='3'))]
    EEG_idx = [cc for cc in range(len(ch_names)) if ch_names[cc][:3]=='EEG']

    chantypes_idx = {'mag': MAG_idx, 'grad': GRA_idx, 'eeg': EEG_idx}

    return chantypes_idx



def mSSP_get_mats_from_evoked(evoked, chtypes_idx):
    """ Get data matrices from evoked for channel types (mag, grad, eeg)

    Parameters
    ----------
    evoked: Instance of evoked
        evoked object with data
    chtypes_idx: dict of lists
        entries of dict are lists of indices for different channel types
        (e.g. from mSSP_get_channel_types())

    Returns
    -------
    mat_evk: dict of lists
        entries of dict are np.arrays with data for channel types
    """
    
    mat_evk = {}

    data = evoked.data

    for chtype in chtypes_idx:
        mat_evk[chtype] = data[chtypes_idx[chtype],:]

    return mat_evk


def mSSP_get_mats_from_epochs(epochs, chtypes_idx):
    """ Get data matrices from epochs for channel types (mag, grad, eeg)

    Parameters
    ----------
    epochs: Instance of Epoch
        epoch object with data
    chtypes_idx: dict of lists
        entries of dict are lists of indices for different channel types
        (e.g. from mSSP_get_channel_types())

    Returns
    -------
    mat_epo: dict of lists
        entries of dict are np.arrays with data for channel types.
        n_epo x n_chan x n_times.
    """
    
    mat_epo = {}

    data = epochs.get_data()
    
    for chtype in chtypes_idx:
        mat_epo[chtype] = data[:,chtypes_idx[chtype],:]

    return mat_epo


def mSSP_avg_proj_timecourses(epochs_dict, abs=False):
    """ average epochs as numpy arrays (e.g. projected time courses)

    Parameters
    ----------
    epochs_dict: dict of 3D numpy arrays
        dict elements are different channel types
        will be averaged along 2nd dimension (epochs)
    abs: Bool
        if True, absolute values will be averaged
        e.g. for SNRs
        defaults to False
    
    Returns
    -------
    avg_epo: dict of 2D numpy arrays
        averaged epochs (n_chan x n_times)
        n_chan can represent projected time courses for different estimators
    """
    
    avg_epo = {}
    for chantype in epochs_dict:
        if abs: # take absolute values for averaging
            data = np.abs(epochs_dict[chantype])
        else:
            data = epochs_dict[chantype]
        
        avg_epo[chantype] = np.average(data, 1)

    return avg_epo


def mSSP_SNR_proj_timecourses_epochs(epochs_dict, times, baseline=None):
    """ compute SNRs of projected time courses for epochs
        (e.g. from mSSP_spatfilt_on_epochs, mSSP_mSSP_on_epochs)

    Parameters
    ----------
    epochs_dict: dict of 2D or 3D numpy arrays
        dict elements are different channel types
        arrays with epochs:
        3D: n_chan x n_epochs x n_times
        2D: n_times x n_epochs
        n_chan can represent projected time courses for different estimators if 3D.
    times: list of float
        latencies for samples from which indices to be determined (e.g. epochs.times)
    baseline: list of float
        baseline interval in s
        if None, from beginning to 0s
    
    Returns
    -------
    SNR_epo: dict of 2D or 3D numpy arrays
        SNRs of epochs. Dimensions as in epochs_dict:
        3D: n_chan x n_epochs x n_times
        2D: n_times x n_epochs
        SNRs computed with respect to specified baseline."""

    if baseline == None:
        baseline = []
        baseline[0] = times[0] # first latency in epoch
        baseline[1] = 0.
    elif baseline[0] == None:
        baseline[0] = times[0]

    idx = mSSP_get_latency_indices(baseline, times)

    SNR_epo = {}
    
    for chantype in epochs_dict:
        data = epochs_dict[chantype]
        if data.ndim == 3: # for 3D arrays
            data_base = data[:,:,idx[0]:idx[1]+1]
            std_epo = np.std(data_base, 2)
            n_t = data.shape[2]
            std_epo_3D = np.tile(std_epo, [n_t,1,1])
            std_epo_3D = np.swapaxes(std_epo_3D, 0, 2) # put times at end
            std_epo_3D = np.swapaxes(std_epo_3D, 0, 1) # get chans and epochs in right order
            snr_epo = np.divide(data, std_epo_3D)                
        elif data.ndim == 2: # for 2D arrays
            data_base = data[idx[0]:idx[1]+1,:]
            std_epo = np.std(data_base, 0)
            n_t = data.shape[0]
            std_epo_2D = np.tile(std_epo, [n_t,1])
            snr_epo = np.divide(data, std_epo_2D)
        else:
            print "Don't know what to do with array of dimension {0}.".format(str(data.ndim))

        SNR_epo[chantype] = snr_epo

    return SNR_epo



def mSSP_peak_lats_proj_epochs(epochs_dict, times, lat_wins, epo_idx=[]):
    """ Compute peak latencies for projected time courses in epochs

    Parameters
    ----------
    epochs_dict: dict of 3D numpy arrays
        dict elements are different channel types
        arrays with epochs:
        3D: n_chan x n_epochs x n_times
        n_chan can represent projected time courses for different estimators if 3D.
    times: list of float
        latencies for samples in epochs (e.g. epochs.times)
    lat_wins : list of 2-element lists of float
        Latency windows (s) for which peak latencies to be computed.
    epo_idx: list of int
        indices to epochs to be considered.
        If empty, use all epochs.
    
    Returns
    -------
    Peak_Lats: dict of dict of 2D or 3D numpy arrays
        Peak latencies in epochs. Latency windows as dict, type of latency as dict,
        channel types as dict.
        Types of latency are 'peak' and 'cof' (centre-of-gravity)
        np.array dimensions as in epochs_dict: 3D: n_chan x n_epochs.
        n_epochs depends on epo_idx. 
        n_chan can represent time courses for different estimators."""

    # in case only one latency windows specified as one list element
    if len(lat_wins) == 2 and (type(lat_wins[0]) == float or type(lat_wins[0]) == int):
        lat_wins = [lat_wins]

    # initialise dict for latency windows
    Peak_Lats = {}

    for [win_idx, win] in enumerate(lat_wins):

        peak_lats = {} # initalise dict for channel types

        [idx1, idx2] = mSSP_get_latency_indices(win, times)

        t_win = times[idx1:idx2]
            
        for chantype in epochs_dict:            
            
            data = epochs_dict[chantype]                    

            n_sf = data.shape[0] # numer of spatial filters

            n_epo = data.shape[1]            

            if len(epo_idx)==0: # take all epochs
                epo_idx = range(n_epo)

            n_epo_idx = len(epo_idx)

            # different latency measures: peak and centre-of-gravity
            peak_lats[chantype] = dict(peak=np.zeros([n_sf, n_epo_idx]), cof=np.zeros([n_sf, n_epo_idx]))

            for ff in range(n_sf):

                for [eei, ee] in enumerate(epo_idx): # for all specified epochs

                    # one time course for lat window in one epoch
                    tcs_win = data[ff,eei,idx1:idx2]                    

                    tcs_win = np.abs(tcs_win)

                    max_idx = tcs_win.argmax() # index of max

                    max_lat = t_win[max_idx] # latency of max

                    peak_lats[chantype]['peak'][ff,eei] = max_lat

                    # centre-of-gravity latency
                    peak_lats[chantype]['cof'][ff,eei] = t_win.dot(tcs_win) / tcs_win.sum()

        win_str = str(int(1000*win[0])) + '-' + str(int(1000*win[1])) + 'ms'
        Peak_Lats[win_str] = peak_lats

    return Peak_Lats



def mSSP_var_from_eigen(svd_s, n_comp=None):
    """get variances associated with eigenvalues from SVD
        e.g. from mSSP_SVD_on_Evoked()

    Parameters
    ----------
    svd_s: list of dict of np.arrays
        as from mSSP_SVD_on_Evoked()
        singular values from SVD per window
        list: time window; dict: channel types        
    n_comp: int
        number of eigenvalues to be considered (all if None)

    Returns
    -------
    svd_var: list of dict of np.arrays
        variances for n_comp eigenvalues in svd_s"""

    svd_var = []
    for svd in svd_s:
        var = {}
        for chtype in svd:
            svd_pow = np.power(svd[chtype][:n_comp], 2)
            var[chtype] = 100. * svd_pow / svd_pow.sum()

        svd_var.append(var)

    return svd_var

def mSSP_get_latency_indices(lat_list, times):
    """get indices of latencies in sample list

    Parameters
    ----------
    lat_list: list of float
        as from mSSP_SVD_on_Evoked()
        latencies for which indices to be returned
        e.g. baseline interval
    times: list of float
        latencies of samples from which indices to be returned

    Returns
    -------
    lat_inds: list of int
        indices for elements of lat_list in times"""

    lat_inds = [np.abs(times-lat).argmin() for lat in lat_list]    

    return lat_inds



def mSSP_get_epoch_indices(epochs, cond):
    """get indices to epochs for specific condition

    Parameters
    ----------
    epochs: intance of Epochs
        Epochs object with events from which subset indices to be determined
    cond: int or string
        Specfies condition for which indices to be found.
        int: find event value.
        string: find event name.

    Returns
    -------
    idx: np.array of int
        indices to epochs for specified condition."""

    events = epochs.events

    if type(cond) == int:        
        idx = np.where(events[:,2]==cond)
    elif type(cond) == str:
        ev_id = epochs.event_id[cond]
        idx = np.where(events[:,2]==ev_id)
    else:
        print "Type of cond not valid."
    
    return idx[0]



def mSSP_SVD_on_Epochs(epochs, lat_wins, n_comp=10):
    """Extract first singular vectors for latency ranges in
        epoched data, for each epoch

    Parameters
    ----------
    epochs : instance of Epoch
        Epoched data to be subjected to SVD.
    lat_wins : list of 2-element lists of float
        Latency windows (s) for which SVD to be computed.
    n_comp: number of SVD components to be computed
    

    Returns
    -------
    svd_mat: list of np.arrays
        First n_comp singular vectors for each latency window.
        Values will be inserted according to indices of
        channels types mag/grad/eeg (if present)
    svd_S: list of dict of np.arrays
        singular values from SVD per window
        list: time window; dict: channel types"""

    import numpy as np
    from scipy import linalg    

    # epoch has dimensions n_eps x n_chan x n_times

    # get data matrix from epochs for different channel types
    ch_names_epo = epochs.ch_names
    chtyp_idx_epo = mSSP_get_channel_types(ch_names_epo)

    # Get data matrices from epochs for channel types (as dict: mag, grad, eeg)
    mat_epo = mSSP_get_mats_from_epochs(epochs, chtyp_idx_epo)

    n_epo = len(epochs)

    # overall number of channels
    n_chan = len(epochs.ch_names)    

    # in case only one latency windows specified as one list element
    if len(lat_wins) == 2 and (type(lat_wins[0]) == float or type(lat_wins[0]) == int):
        lat_wins = [lat_wins]

    # number of latency windows
    n_wins = len(lat_wins)

    # intitialise list of results
    svd_mat = []
    svd_S = []
        
    for [win_idx, win] in enumerate(lat_wins):

        print "Computing SVDs for epochs in time window " + str(1000*win[0]) + " to " + str(1000*win[1]) + "ms"

        [idx1, idx2] = mSSP_get_latency_indices(win, epochs.times)

        print idx1, idx2

        # results for this time window as np.array
        svd_mat_ch = np.zeros([n_epo, n_chan, n_comp])
        s_svd_ch = {}
                
        # apply SVD to latency window        
        for chantype in chtyp_idx_epo:

            # get data matrix for this channel type, reduce to latency window
            data = mat_epo[chantype][:,:,idx1:idx2]

            # initialise, number of components depends on channels and time points
            s_svd_ch[chantype] = np.zeros([n_epo, np.min([data.shape[1],data.shape[2]])])

            # subtract means across channels for each channel type before SVD
            data = data - data.mean(axis=1, keepdims=True)

            # compute SVDs by epoch
            for ee in range(n_epo):
                u_svd_ee, s_svd_ee, _ = linalg.svd(data[ee,:,:], full_matrices=False, compute_uv=True)

                ch_idx = chtyp_idx_epo[chantype]
                svd_mat_ch[ee,ch_idx,:] = u_svd_ee[:,:n_comp]

                s_svd_ch[chantype][ee,:] = s_svd_ee
                
        svd_mat.append(svd_mat_ch) 
        svd_S.append(s_svd_ch)

    return svd_mat, svd_S



def mSSP_SVD_on_Evoked(evoked, lat_wins, n_comp=10):
    """Extract first singular vectors for latency ranges in
    	evoked data

    Parameters
    ----------
    evoked : instance of Evoked
        Evoked data to be subjected to SVD.
    lat_wins : list of 2-element lists of float
        Latency windows (s) for which SVD to be computed.
    n_comp: number of SVD components to be computed
    

    Returns
    -------
    SVD_mat: list of np.arrays
        First n_comp singular vectors for each latency window.
        Values will be inserted according to indices of
        channels types mag/grad/eeg (if present)
    SVD_S: list of dict of np.arrays
        singular values from SVD per window
        list: time window; dict: channel types"""

    import numpy as np
    from scipy import linalg    

    # get data matrix
    data = evoked.data
    # get sample latencies (s)
    times = evoked.times

    ch_names = evoked.ch_names
    
    chantypes_idx = mSSP_get_channel_types(ch_names)

    # in case only one latency windows specified as one list element
    if len(lat_wins) == 2 and (type(lat_wins[0]) == float or type(lat_wins[0]) == int):
        lat_wins = [lat_wins]

    # number of latency windows
    n_wins = len(lat_wins)

    # list for first SVD components
    SVD_mat = []
    SVD_S = []
    
    for [win_idx, win] in enumerate(lat_wins):
        svd_mat = np.zeros([data.shape[0],n_comp]) # initialise for this time window

        idx1, idx2 = mSSP_get_latency_indices(win, evoked.times)
        data_win = data[:,idx1:idx2]

        if not data_win.any():
            print "\nNo data in latency window - did you specific latencies correctly (in s)?\n"

        # apply SVD to latency window
        s_svd = {} # singular values per channel type
        for chantype in chantypes_idx:
            ch_idx = chantypes_idx[chantype]
            if ch_idx:
                data_chan = data_win[ch_idx,:]
                # subtract means from columns for each channel type before SVD
                data_chan = data_chan - data_chan.mean(axis=0, keepdims=True)
                u_svd, s_svd[chantype], _ = linalg.svd(data_chan, full_matrices=False, compute_uv=True)

                # keep 1st singular vector                
                svd_mat[ch_idx,0:n_comp] = u_svd[:,0:n_comp]

        SVD_mat.append(svd_mat)
        SVD_S.append(s_svd)

    return SVD_mat, SVD_S



def mSSP_spatfilt_on_epochs(epochs, spatfilt):
    """Project spatial filters on epoched data, return projected time courses

    Parameters
    ----------
    epochs : instance of Epoch
        Epoched data for projection.
    spatfilt : instance of Evoked
        Evoked object contains spatial filters for sensor types that are present in epoch.
        
    Returns
    -------
    timecourses: dict of numpy arrays
        Saptial filters as np.arrays, each with dict for different channel types.
        Different time points in an evoked object will result in different rows of the 
        numpy array in timecourses.
        For example, if the evoked object contains 3 "time points" (e.g.
        corresponding to different parameter settings for each spatial filter type), then
        for example timecourses['grad'].shape is (3,n_epochs,n_times)."""

    # epoch has dimensions n_eps x n_chan x n_times
    # epochs._data # if preloaded

    # get data matrix from epochs for different channel types
    ch_names_epo = epochs.ch_names
    chtyp_idx_epo = mSSP_get_channel_types(ch_names_epo)
    
    # Get data matrices from epochs for channel types (as dict: mag, grad, eeg)
    mat_epo = mSSP_get_mats_from_epochs(epochs, chtyp_idx_epo)

    timecourses = {}
    # new_epochs =copy.deepcopy(epochs) # will contain projected time courses

    # get channel type indices for spatial filters
    ch_names_sf = spatfilt.ch_names 
    chtyp_idx_sf = mSSP_get_channel_types(ch_names_sf)
    # get matrices with spatial filters, as dict for different channel types
    mat_sf = mSSP_get_mats_from_evoked(spatfilt, chtyp_idx_sf)

    mat = {} # will contain projections for this spatial filter
    for chtype in mat_sf: # channel types as separate dict entries
        mat[chtype] = np.dot(mat_sf[chtype].T,mat_epo[chtype])

    # list for different spatial filters, each with dict for different channel types
    timecourses = mat

    return timecourses



def mSSP_mSSP_on_epochs(epochs, mSSP_sf, ch_names_sf):
    """Project mSSP spatial filters on epoched data, return projected time courses.

    Parameters
    ----------
    epochs : instance of Epoch
        Epoched data for projection.
    mSSP_sf : 2D np.array
        Contains mSSP spatial filters for each epoch (e.g. from mSSP_compute_mSSP_estimators).        
        n_chan x n_epo.
    ch_names_sf: list of strings
        channel names for mSSP_sf.

    Returns
    -------
    timecourses: dict of 3D numpy arrays
        For each channel type, contains array n_times x n_epo.
        3D even though currently only for one spatial filter."""

    # epoch has dimensions n_eps x n_chan x n_times
    # epochs._data # if preloaded

    # get data matrix from epochs for different channel types
    ch_names_epo = epochs.ch_names
    chtyp_idx_epo = mSSP_get_channel_types(ch_names_epo)

    chtyp_idx_sf = mSSP_get_channel_types(ch_names_sf)
    
    # Get data matrices from epochs for channel types (as dict: mag, grad, eeg)
    # for channels specified for spatial filter
    mat_epo = mSSP_get_mats_from_epochs(epochs, chtyp_idx_sf)

    proj_epo = {}
    # new_epochs =copy.deepcopy(epochs) # will contain projected time courses

    n_epo = len(epochs)

    for ch_type in chtyp_idx_epo:

        proj_epo[ch_type] = np.zeros([1, n_epo, len(epochs.times)]) # first dimension for compatibility

        for ee in range(n_epo):

            proj_epo[ch_type][0,ee,:] = np.dot(mSSP_sf[chtyp_idx_sf[ch_type],ee].T, mat_epo[ch_type][ee,:,:]).T

    timecourses = proj_epo

    return timecourses


def mSSP_invert_square_Tikhonov(mat, regpar):
    """Pseudo-invert square matrix using Tiknonov regularisation
       by adding diagonal before applying pinv

    Parameters
    ----------
    mat : 2D m x m numpy array
        Square matrix to be pseudo-inverted.
    regpar: float
        Regularisation parameter.
        Identity matrix will be weighted by lambda*trace(mat).
        Defaults to zero.

    Returns
    -------
    invmat: 2D numpy array
        Pseudo-inverted matrix."""

    [m, n] = mat.shape

    if m != n:
        print "Sorry - this is not a square matrix!"

    my_eye = np.eye(m)

        # sum of diagonal elements
    mat_trace = np.trace(mat)

    # add regularisation term
    mat_eye = mat + regpar*(mat_trace/m)*my_eye

    invmat = pinv(mat_eye)

    return invmat


def mSSP_invert_SVD_trunc(mat, n_comp):
    """Pseudo-invert matrix using SVD eigenvalue cut-off

    Parameters
    ----------
    mat : 2D numpy array
        Matrix to be pseudo-inverted.
    n_comp : int
        Number of eigenvalues to be retained.

    Returns
    -------
    invmat: 2D numpy array
        Pseudo-inverted matrix."""

    u_svd, s_svd, v_svd = svd(mat, full_matrices=False, compute_uv=True)

    n_s = len(s_svd)

    s_inv = np.zeros([n_s,n_s])

    idx = np.diag_indices(n_comp)

    s_inv[idx] = np.power(s_svd[:n_comp], -1)

    invmat = v_svd.T.dot(s_inv).dot(u_svd.T)

    return invmat



def mSSP_make_whitener(noise_cov, info, pca=False, rank=None,
                     verbose=None):
    """get whitening matrix from covariance matrix
       ripped from mne.minimum_norm.inverse._prepare_forward
       I still don't understand this - doesn't seem to work
       e.g. whitener.dot(noise_cov.data).dot(whitener) is not near to identity
    """
        
    ch_names = [c['ch_name'] for c in info['chs']
                if ((c['ch_name'] not in info['bads'] and
                     c['ch_name'] not in noise_cov['bads']) and
                    (c['ch_name'] in noise_cov.ch_names))]

    if not len(info['bads']) == len(noise_cov['bads']) or \
            not all(b in noise_cov['bads'] for b in info['bads']):
        logger.info('info["bads"] and noise_cov["bads"] do not match, '
                    'excluding bad channels from both')

    n_chan = len(ch_names)
    logger.info("Computing whitener with %d channels." % n_chan)

    #
    #   Handle noise cov
    #
    noise_cov = prepare_noise_cov(noise_cov, info, ch_names, rank)

    #   Omit the zeroes due to projection
    eig = noise_cov['eig']
    nzero = (eig > 0)
    n_nzero = sum(nzero)

    if pca:
        #   Rows of eigvec are the eigenvectors
        whitener = noise_cov['eigvec'][nzero] / np.sqrt(eig[nzero])[:, None]
        logger.info('Reducing data rank to %d' % n_nzero)
    else:
        whitener = np.zeros((n_chan, n_chan), dtype=np.float)
        whitener[nzero, nzero] = 1.0 / np.sqrt(eig[nzero])
        #   Rows of eigvec are the eigenvectors
        whitener = np.dot(whitener, noise_cov['eigvec'])

    logger.info('Total rank is %d' % n_nzero)

    return whitener, noise_cov, n_nzero


def mSSP_compute_mSSP_estimators(target, epochs_SVD, ch_names, n_comp, topo_add=[], topo_add_idx=[0], get_corr=True):
    """Compute mSSP spatial filters for individual epochs

    Parameters
    ----------
    target : instance of Evoked
        Target topographies for spatial filters.
    epochs_SVD : 3D np.array
        First n_comp singular vectors for each latency window.
        Values will be inserted according to indices of
        channels types mag/grad/eeg (if present).
    ch_names: list of strings
        list of channel names associated with epochs object from which epochs_SVD was obtained.
    n_comp: int
        number of components from epochs_SVD to be used for mSSP computation
    topo_add: instance of Evoked
        noise topography(ies) to be added to every epoch's mSSP model (e.g. eye blink)
    topo_add_idx: list of int
        indices of samples in topo_add to be included as added topographies
    get_corr: Bool
        if True, intercorrelations for design matrix will be computed.

    Returns
    -------
    mSSP_sf: 2D np.array
        spatial filters for target for all epochs.
        Values will be inserted according to indices of
        channels types mag/grad/eeg (if present) in info of target.
    CorrCoef: 2D np.array
        If get_corr==True: correlation coefficients for design matrices for epochs.
        n_epo x n_comp.
    CondNum: 1D np.array
        If get_corr==True: condition number of design matrices for epochs."""

    # epoch has dimensions n_eps x n_chan x n_times
    # epochs._data # if preloaded

    if len(set(ch_names) & set(target.ch_names)) != len(target.ch_names):
        print "Not all channels in target are present in info for epochs_SVD!"

    print "Computing mSSP estimator with {0} components".format(str(n_comp))

    chtyp_idx_epo = mSSP_get_channel_types(ch_names)

    chtyp_idx_tar = mSSP_get_channel_types(target.ch_names)

    n_epo = epochs_SVD.shape[0]

    data = target.data

    # array with mSSP spatial filters
    mSSP_sf = np.zeros([len(target.ch_names),n_epo])

    CorrCoef = np.array(0) # correlation coefficients per epoch if wanted
    CondNum = np.array(0) # condition numbers per epoch if wanted

    # add extra noise topographies if specified
    if topo_add:
        n_add = len(topo_add_idx)
        print "Adding {0} topographies.".format(n_add)
        chtyp_idx_add = mSSP_get_channel_types(topo_add.ch_names)
        data_add = topo_add.data[:,topo_add_idx]

    # if correlations requested
    if get_corr:
        n_comp_corr = n_comp + 1 # number of elements in row of correlation matrix
        if topo_add:
            n_comp_corr += n_add
        CorrCoef = np.zeros([n_epo,n_comp_corr])
        CondNum = np.zeros(n_epo) # condition number of mSSP model

    for ee in range(n_epo):
        for ch_type in chtyp_idx_tar:
            # matrix for target topography plus noise SVD components
            
            if topo_add: # if additional topographies specified                
                # target plus added topos plus noise topos
                X = np.zeros([len(chtyp_idx_epo[ch_type]),1+n_add+n_comp])
                X[:,0] = data[chtyp_idx_tar[ch_type],0]
                X[:,1:n_add+1] = data_add[chtyp_idx_add[ch_type],:]
                X[:,n_add+1:] = epochs_SVD[ee,chtyp_idx_epo[ch_type],0:n_comp]
            else:
                # target plus noise topos
                X = np.zeros([len(chtyp_idx_epo[ch_type]),1+n_comp])
                X[:,0] = data[chtyp_idx_tar[ch_type],0]
                X[:,1:] = epochs_SVD[ee,chtyp_idx_epo[ch_type],0:n_comp]

            # invert matrix
            Xinv = pinv(X)
            # first row of Xinv is desired estimator for this epoch
            mSSP_sf[chtyp_idx_tar[ch_type],ee] = Xinv[0,:].T

            if get_corr: # if correlations wanted
                corr = np.corrcoef(X, rowvar=0)                
                if corr.ndim == 0:
                    CorrCoef[ee,:] = corr
                else:
                    CorrCoef[ee,:] = corr[0,:]

                CondNum[ee] = np.linalg.cond(X)

                if np.mod(ee,100)==0: # print only every 100th epoch
                    # print CorrCoef[ee,:]
                    if CorrCoef.shape[1]>1:
                        print CorrCoef[ee,1:].max() # maximum non-diagonal value
                    print CondNum[ee]
                    
    return mSSP_sf, CorrCoef, CondNum


def mSSP_make_beamformer(cov, topos, regpars):
    """Make beamformer estimator given covariance matrix and topography(ies)
        inclulde MLE depending on covmat

    Parameters
    ----------
    cov : instance of Covariance matrix
        Covariance matrix for beamformer, assumed to be regularised.
    topos : instance of Evoked
        One beamformer per topography will be computed, with this toography as target.
    regpars : float or dict of float
        regularisation parameters for covariance matrices (per channel type)
        if float, one regpar for all channel types
        if dict, needs entries for all channel types in cov ('mag', 'grad', 'eeg')

    Returns
    -------
    spatfilt: Evoked
        spatial filters as topographies."""

    # evoked object to contain spatial filters
    spatfilt = copy.deepcopy(topos)   

    # make sure covariance matrix and topos agree in channels
    # assumes noise_cov has at least the channels of topos
    ch_names = topos.ch_names

    # channel indices for different channel types
    ch_idx = mSSP_get_channel_types(ch_names)

    print "Preparing covariance matrix"
    cov = prepare_noise_cov(cov, info=topos.info, ch_names=ch_names, rank=None)

    # noise covariance matrix as np.array
    cov_mat = cov.data

    print "Getting sub-matrices for channel types"
    cov_dict = {}
    if ch_idx['mag']:
        cov_dict['mag'] = cov_mat[np.ix_(ch_idx['mag'],ch_idx['mag'])]
    if ch_idx['grad']:
        cov_dict['grad'] = cov_mat[np.ix_(ch_idx['grad'],ch_idx['grad'])]
    if ch_idx['eeg']:
        cov_dict['eeg'] = cov_mat[np.ix_(ch_idx['eeg'],ch_idx['eeg'])]

        ### if using whitener-approach (not happy with it):
    # cov_inv_dict = {}    
    # if ch_idx['mag']:
    #     chs = [ch_names[idx] for idx in ch_idx['mag']]
    #     cov_now = pick_channels_cov(cov, include=chs)
    #     [cov_inv_now, _, _] = mSSP_make_whitener(cov_now, topos.info, rank=None)
    #     cov_inv_dict['mag'] = cov_inv_now
    # if ch_idx['grad']:
    #     chs = [ch_names[idx] for idx in ch_idx['grad']]
    #     cov_now = pick_channels_cov(cov, include=chs)
    #     [cov_inv_now, _, _] = mSSP_make_whitener(cov_now, topos.info, rank=None)
    #     cov_inv_dict['grad'] = cov_inv_now
    # if ch_idx['eeg']:
    #     chs = [ch_names[idx] for idx in ch_idx['eeg']]
    #     cov_now = pick_channels_cov(cov, include=chs)
    #     [cov_inv_now, _, _] = mSSP_make_whitener(cov_now, topos.info, rank=None)
    #     cov_inv_dict['eeg'] = cov_inv_now 
   
    print "Computing spatial filters"
    sf_dict = {} # spatial filters per channel type

    for ch_type in cov_dict:
        print ch_type

        topo_now = topos.data[ch_idx[ch_type],:]
        [n_c, n_t] = topo_now.shape
        sf_dict[ch_type] = np.zeros([n_c,n_t])

        # # take square of whitener for beamformer
        # cov_dict_inv = cov_inv_dict[ch_type].dot(cov_inv_dict[ch_type])

        # (pseudo)invert covariance matrix
        # cov_dict_inv = pinv(cov_dict[ch_type])
        # using eigenvalue truncation
        # cov_dict_inv = mSSP_invert_SVD_trunc(cov_dict[ch_type], n_comp=30)
        # using Tikhonov regularisation
        if type(regpars)==dict: # if reg params specified per channel type
            regpar = regpars[ch_type]
        else: # if only one reg param specified: use for all channel types
            regpar = regpars

        cov_dict_inv = mSSP_invert_square_Tikhonov(cov_dict[ch_type], regpar=regpar)

        for tt in range(n_t):
            targ = topo_now[:,tt]
            denom = targ.T.dot(cov_dict_inv).dot(targ)
            sf_dict[ch_type][:,tt] = cov_dict_inv.dot(targ) / denom

        spatfilt.data[ch_idx[ch_type],:] = sf_dict[ch_type]

    return spatfilt

    ### Maybe useful when combining sensor types:

    # # make sure covariance matrix and topos agree in channels
    # # assumes noise_cov has at least the channels of topos
    # ch_names = topos.ch_names
    # noise_cov = pick_channels_cov(cov, ch_names)

    

    # # (pseudo)invert covariance matrix
    # cov_mat_inv = pinv(cov_mat)

    # # columns are topographies
    # topo_mat = topos.data

    # [n_c, n_t] = topo_mat.shape

    # # initiatlise
    # sf_mat = np.zeros([n_c,n_t])

    # # compute beamformer for every topography
    # for tt in range(n_t):
    #     denom = topo_mat[:,tt].T.dot(cov_mat_inv).dot(topo_mat[:,tt])
    #     sf_mat[:,tt] = cov_mat_inv.dot(topo_mat[:,tt])
        
    # # insert matrix into evoked object
    # spatfilt.data = sf_mat

    # return spatfilt


def mSSP_plot_topo_PDF(pdf_fname, data_in, info, title='', step=1):
    """plot topographies to PDF, for channel types separately.

    Parameters
    ----------
    pdf_fname: string
        Full PDF file name.
    data_in: np.array (2D or 3D)
        Topographies to be plotted.
        If 3D, first dimension is for epochs.
        If 2D, second dimension is for epochs.
    info: info from evoked object
        info associated with data_in.
    title: string
        title for PDF pages
    step : int
        Steps between topographies to be plotted.
        Only makes sense if data is 3D.
        

    Returns
    -------
    Void."""

    if data_in.ndim == 2:
        n_epo = data_in.shape[1]
    elif data_in.ndim == 3:
        n_epo = data_in.shape[0]
    else:
        print "Don't know what to do with {0} dimensions".format(str(data_in.ndim))

# Plot topographies per subject and condition for some epochs into PDFs
    with PdfPages(pdf_fname) as pdf:
        for ee in range(0,n_epo,step):

            # if input only 2D, create third dimension for one "epoch"
            if data_in.ndim == 2:
                data = np.zeros([data_in.shape[0],1])
                data[:,0] = data_in[:,ee]
                times = 0.
            else:
                data = data_in[ee,:,:]
                times = "auto"
            
            # get data for plotting into evoked object
            evoked_now = EvokedArray(data, info, tmin=0.0)                                
            evoked_now.comment = str(ee)

            pdftitle = title + str(ee)
            pdf.attach_note(pdftitle)
            evoked_now.plot_topomap(ch_type='mag', times=times, show=False)
            pdf.savefig()
            plt.close()
 
            evoked_now.plot_topomap(ch_type='grad', times=times, show=False)
            pdf.savefig()
            plt.close()

            evoked_now.plot_topomap(ch_type='eeg', times=times, show=False)
            pdf.savefig()
            plt.close()


def mSSP_plot_avg_timecourses(avgs, t, titles, legend, fname_stem, myaxis=[]):
    """ Plot average projected time courses for channel types (mag, grad, eeg)

    Parameters
    ----------
    avgs: dict of 2D numpy arrays
        average time courses for different channel types (dict)
    t: 1d numpy array
        latencies for samples in avg
    titles: list of text strings
        general figure titles for elements of avg_list
    legend: list of text strings
        labels for different lines in plots.    
    fname_stem: text string
        figure filename stem (not saved if empty).
        Will be saved as PDF and SVG file.
    myaxis: list
        4 elements, used in plt.axis().
        If empty default is used.

    Returns
    -------
    fig: pyplot figure handle
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    if type(titles) != list:
        titles = [titles]

    if legend: # plot legend in extra subplot if present
        n_subs = 4
    else:
        n_subs = 3

    n_avgs = len(avgs)

    # mpl.rcParams['xtick.labelsize'] = 24

    with PdfPages(fname_stem+'.pdf') as pdf:
            fig = plt.figure()            

            plot = fig.add_subplot(n_subs,1,1)
            plot.set_title('MAG')
            plot.tick_params(axis='both', which='major', labelsize=12)
            plot.plot(t, avgs['mag'].T, linewidth=2)
            if myaxis: plot.axis(myaxis)
            plot.set_yticks(range(int(np.ceil(avgs['mag'].max()+1))))

            plot = fig.add_subplot(n_subs,1,2)
            plot.set_title('GRAD')
            plot.tick_params(axis='both', which='major', labelsize=12)
            plot.plot(t, avgs['grad'].T, linewidth=2)
            if myaxis: plot.axis(myaxis)
            plot.set_yticks(range(int(np.ceil(avgs['grad'].max()+1))))

            plot = fig.add_subplot(n_subs,1,3)
            plot.set_title('EEG')
            plot.tick_params(axis='both', which='major', labelsize=12)
            plot.plot(t, avgs['eeg'].T, linewidth=2)
            if myaxis: plot.axis(myaxis)
            plot.set_yticks(range(0,int(np.ceil(avgs['eeg'].max()+1)),2))

            # fig.tight_layout() # not ideal
            fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
            
            if legend:
                plot = fig.add_subplot(n_subs,1,4)
                plot.plot(t, avgs['eeg'].T)
                plot.legend(legend, loc='upper left')
            if fname_stem:
                pdf.savefig()
            plt.close()

    fig.savefig(fname_stem + '.svg')

    return fig

# All over