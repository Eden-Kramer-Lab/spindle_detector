import numpy as np

from hmmlearn import hmm
from spindle_detector.lfp_likelihood import (_DEFAULT_MULTITAPER_PARAMS,
                                             estimate_spindle_band_power)


def atleast_2d(x):
    return np.atleast_2d(x).T if x.ndim < 2 else x


def detect_spindle(lfps, sampling_frequency, start_time=0.0,
                   multitaper_params=_DEFAULT_MULTITAPER_PARAMS):
    '''Finds spindle times using spectral power between 10-16 Hz and an HMM.

    Parameters
    ----------
    lfps : ndarray, shape (n_time, n_signals)
    sampling_frequency : float
    start_time : float

    Returns
    -------
    time : ndarray, shape (n_time_windows,)
    spindle_probability : ndarray, shape (n_time_windows, 2)
    is_spindle : ndarray, shape (n_time_windows,)
    model : hmmlearn.GaussianHMM instance

    '''
    time, spindle_band_power = estimate_spindle_band_power(
        atleast_2d(lfps), sampling_frequency, start_time=start_time,
        multitaper_params=multitaper_params)
    startprob_prior = np.log(np.array([np.spacing(1), 1.0 - np.spacing(1)]))
    model = hmm.GaussianHMM(n_components=2, covariance_type='full',
                            startprob_prior=startprob_prior, n_iter=100)
    model = model.fit(np.log(spindle_band_power))

    state_index = model.predict(np.log(spindle_band_power))

    if (spindle_band_power[state_index == 0].mean() >
            spindle_band_power[state_index == 1].mean()):
        spindle_ind = 0
    else:
        spindle_ind = 1

    is_spindle = np.zeros_like(state_index, dtype=np.bool)
    is_spindle[state_index == spindle_ind] = True

    spindle_probability = model.predict_proba(np.log(spindle_band_power))
    spindle_probability = spindle_probability[:, spindle_ind]

    return time, spindle_probability, is_spindle, model
