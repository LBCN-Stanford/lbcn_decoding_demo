###Purpose: This function converts ecog data from a fieldtrip structure to mne epochs.
###Project: General
###Author: Pinheiro-Chagas
###Date: 14 January 2016

#Load libraries
import numpy as np
import scipy.io as sio
import pandas as pd

from mne.io.meas_info import create_info
from mne.epochs import EpochsArray


def ieegmat2mne(filename_data, fname_trialinfo):
    "This function converts data from a fieldtrip structure to mne epochs"

    #Load Matlab/fieldtrip data
    mat = sio.loadmat(filename_data, squeeze_me = True, struct_as_record = False)
    ft_data = mat['data_all']
    
    #Identify basic parameters
    n_trial = np.size(ft_data.wave, 0)
    data = ft_data.wave

    #Add all necessary info
    sfreq = float(ft_data.fsample) #sampling frequency
    time = ft_data.time;
    chan_names = ft_data.labels.tolist()
    chan_types = 'ecog'

    #Create info and epochs
    info = create_info(chan_names, sfreq, chan_types)

    if ft_data.project_name == 'Context' or ft_data.project_name == 'Calculia' or ft_data.project_name == 'Calculia_China' :
        n_events = 5
    elif ft_data.project_name == 'UCLA' or ft_data.project_name == 'MMR':
        n_events = 1
    elif ft_data.project_name == 'Number_comparison':
        n_events = 2

    events = np.c_[np.cumsum(np.ones(n_trial)) * n_events * sfreq,
                   np.zeros(n_trial), np.zeros(n_trial)]

    epochs = EpochsArray(data, info, events = np.array(events, int),
                         tmin = np.min(time), verbose = False)

    # Import trialinfo
    trialinfo = pd.read_csv(fname_trialinfo)

    return epochs, trialinfo