import numpy as np
import pandas as pd

from hmmlearn import hmm
from spindle_detector.lfp_likelihood import (_DEFAULT_MULTITAPER_PARAMS,
                                             estimate_spindle_band_power)

_startprob_prior = np.log(np.array([np.spacing(1), 1.0 - np.spacing(1)]))
_DEFAULT_HMM_PARAMS = dict(n_components=2, covariance_type='full',
                           startprob_prior=_startprob_prior, n_iter=100,
                           tol=1E-6)


def atleast_2d(x):
    return np.atleast_2d(x).T if x.ndim < 2 else x


def detect_spindle(time, lfps, sampling_frequency,
                   multitaper_params=_DEFAULT_MULTITAPER_PARAMS,
                   hmm_params=_DEFAULT_HMM_PARAMS,
                   spindle_band=(10, 16)):
    '''Finds spindle times using spectral power and an HMM.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    lfps : ndarray, shape (n_time, n_signals)
    sampling_frequency : float
    multitaper_params : dict, optional
    hmm_params : dict, optional
    spindle_band : tuple, optional

    Returns
    -------
    spindle_dataframe : pandas.DataFrame, shape (n_time, 3)
    model : hmmlearn.GaussianHMM instance

    '''
    power_time, spindle_band_power = estimate_spindle_band_power(
        atleast_2d(lfps), sampling_frequency, start_time=time[0],
        multitaper_params=multitaper_params, spindle_band=spindle_band)
    spindle_band_power = spindle_band_power.reshape((power_time.shape[0], -1))

    model = hmm.GaussianHMM(**hmm_params).fit(np.log(spindle_band_power))

    state_ind = model.predict(np.log(spindle_band_power))
    if (spindle_band_power[state_ind == 0].mean() >
            spindle_band_power[state_ind == 1].mean()):
        spindle_ind = 0
    else:
        spindle_ind = 1

    power_time = pd.Index(power_time, name='time')
    time = pd.Index(time, name='time')

    is_spindle = np.zeros_like(state_ind, dtype=np.bool)
    is_spindle[state_ind == spindle_ind] = True
    is_spindle = (pd.DataFrame(dict(is_spindle=is_spindle),
                               index=power_time)
                  .reindex(index=time, method='pad')
                  .reset_index(drop=True))

    spindle_probability = model.predict_proba(np.log(spindle_band_power))
    spindle_df = (pd.DataFrame(
        dict(spindle_probability=spindle_probability[:, spindle_ind]),
        index=power_time)
        .reindex(index=time)
        .reset_index(drop=True)
        .interpolate())

    spindle_df = pd.concat((spindle_df, is_spindle), axis=1).set_index(time)

    return spindle_df, model
