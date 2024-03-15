#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import_ssvep_data.py

A python module to support analysis of SSVEP data. Contains functions to load 
SSVEP data, plot raw SSVEP data, extract epochs of SSVEP data, calculate 
frequency spectra through performing a Fourier transform on SSVEP data, and plot
the mean frequency spectra for channels Oz and Fz. An additional function to 
plot the harmonics for this SSVEP data across all channels is also included.

Created on Thu Feb 22 17:07:14 2024

@author: Ron Bryant and Alaina Birney
"""
import numpy as np
from matplotlib import pyplot as plt
import math
import scipy 

#%%  Cell 1: Load the Data
def load_ssvep_data(subject, relative_data_path='./SsvepData/'):
    '''
    Function to load in the .npz data file containing EEG data for a given 
    subject for SSVEP analysis.
    
    Parameters
    ----------
    subject : Int, required.
        The number of the subject for whom data will be loaded.
    relative_data_path : Str, optional
        The relative path to the data files. The default is './SsvepData/'.

    Returns
    -------
    data_dict : numpy.lib.npyio.NpzFile of size f where f represents the number
    of arrays within this object. 
        This object behaves similarly to a dictionary and can be accessed using 
        keys. Contains raw, unfiltered information about the dataset. Each 
        array within the dictionary holds information corresponding to a 
        different field. In our case, f=6 and corresponds to: eeg data in volts 
        ("EEG"), channels ("channels"), sampling frequency in Hz ("fs"), the 
        sample when each event occured ("event_samples"), the event durations 
        ("event_durations"), and the event types ("event_types").
        

    '''
    # Load dictionary
    data_dict = np.load(f'{relative_data_path}SSVEP_S{subject}.npz',
                        allow_pickle=True)
    return data_dict


#%% Cell 2: Plot Raw EEG Data
def plot_raw_data(data_dict, subject, channels_to_plot):
    '''
    A function to plot raw EEG data for the given channels, for a given subject.
    Results in a figure with two subplots. The top subplot shows events as
    horizontal lines with dots at the start and end times for each event. The
    bottom subplot shows EEG voltage for each specified channel as a function
    of time. Additionally, an image of the plot is saved to 
    SSVEP_S{subject}_rawdata.png within the current directory.

    Parameters
    ----------
    data_dict : numpy.lib.npyio.NpzFile of size F where F represents the number of 
    arrays within this object. Required.
        This object behaves similarly to a dictionary and can be accessed using 
        keys. Contains raw, unfiltered information about the dataset. Each 
        array within the dictionary holds information corresponding to a 
        different field. In our case, f=6 and corresponds to: eeg data in volts 
        ("EEG"), channels ("channels"), sampling frequency in Hz ("fs"), the 
        sample when each event occured ("event_samples"), the event durations 
        ("event_durations"), and the event types ("event_types").
    subject : Int, required.
        The number of the subject for whom data will be plotted.
    channels_to_plot : List of str. Size C where C represents the number of 
    channels for which data will be plotted   Required.
        The channels that EEG data will be plotted for. (Recommended that this 
        be limited to 2 channels to avoid crowding of figure.)

    Returns
    -------
    None.

    '''
    #unpack data_dict
    eeg_data = data_dict['eeg']/1e-6   # convert to microvolts
    channels = data_dict['channels']   # channel names
    fs = data_dict['fs']                # sampling frequency
    event_times = data_dict['event_samples']/fs          # convert to seconds
    event_types = data_dict['event_types']        # frequency of event
    event_durations = data_dict['event_durations']/fs    # convert to seconds
    T = eeg_data.shape[1]/fs     # Total time
    t = np.arange(0,T,1/fs)      # time axis
    
    # initialize figure
    plt.figure(1, clear=True)    
    
    # top subplot: events as horizontal lines with dots at start and end time 
    # of each event
    ax1 = plt.subplot(211)
    for event_index in range(0,len(event_times)):
        event_time = [event_times[event_index],  
                      event_times[event_index]+event_durations[event_index]]
        event_frequency = [event_types[event_index],event_types[event_index]]
        plt.plot(event_time, event_frequency, 'b.-')
    plt.grid(True)
    plt.ylabel('Flash Frequency')
    plt.xlabel('time (sec.)')
    
    # bottom subplot: voltage in channels Fz and Oz as a function of time
    plt.subplot(212, sharex=ax1) # share x axis
    for channel_index in range(0, len(channels_to_plot)):
        channel_name = channels_to_plot[channel_index]
        plt.plot(t,eeg_data[np.where(channels == channel_name)[0][0]], 
                 label=channel_name)
    plt.grid(True)
    plt.ylabel('Voltage (\u03BCV)')
    plt.xlabel('time (sec.)')
    plt.legend(loc='upper right')
   
    plt.suptitle(f'SSVEP Subject {subject} Raw Data')
    plt.tight_layout()
    # save to file
    plt.savefig(f"SSVEP_S{subject}_rawdata.png")
    plt.show()

#%% Cell 3: Extract the epochs
def epoch_ssvep_data(data_dict, epoch_start_time=0, epoch_end_time=20):
    '''
    A function to extract epochs around each event and produce a variable to 
    represent the time of each sample in each epoch, relative to the event 
    onset. Epochs begin when event samples start and end 20 seconds later.

    Parameters
    ----------
    data_dict : numpy.lib.npyio.NpzFile of size F where F represents the number of 
    arrays within this object. Required.
        This object behaves similarly to a dictionary and can be accessed using 
        keys. Contains raw, unfiltered information about the dataset. Each 
        array within the dictionary holds information corresponding to a 
        different field. In our case, F=6 and corresponds to: eeg data in volts 
        ("EEG"), channels ("channels"), sampling frequency in Hz ("fs"), the 
        sample when each event occured ("event_samples"), the event durations 
        ("event_durations"), and the event types ("event_types").
    epoch_start_time : Int, optional
        The time that each epoch begins in seconds, relative to the event 
        sample. The default is 0.
    epoch_end_time : Int, optional
        The time that each epoch ends in seconds, relative to the event sample.
        The default is 20.

    Returns
    -------
    eeg_epochs : Array of float. Size (E,C,T) where E is the number of epochs,
    C is the number of EEG channels, and T is time points.
        EEG data in uV.
    epoch_times : Array of float. Size (T,) where T is time points.
        The time in seconds of each time point in eeg_epochs, relative to the 
        event.
    is_trial_15Hz : Array of bool. Size (E,) where E is the number of epochs.
        An indication of whether the light was flashing at 15 Hz during each
        epoch. True if the light was flashing, false otherwise.

    '''
    #unpack data_dict
    eeg_data = data_dict['eeg']/1e-6   # convert to microvolts
    fs = data_dict['fs']                # sampling frequency
    event_samples = data_dict['event_samples']    #index to start of events
    event_types = data_dict['event_types']    # image frequency of during event
    
    # calculate epoch parameters
    epoch_start_indexes = event_samples + int(epoch_start_time * fs)
    epoch_durations = int((epoch_end_time - epoch_start_time) * fs)
    epoch_times = np.arange(epoch_start_time, epoch_end_time, 1/fs)  #seconds
    epoch_count = len(event_samples)
    
    #initaialize variables to store epochs and indication of whether 15 Hz trial
    eeg_epochs = np.zeros( ( epoch_count, 
                             eeg_data.shape[0], 
                             len(epoch_times) )
                         ) 
    is_trial_15Hz = np.zeros(epoch_count, dtype=bool)
    
    #populate
    for event_index in range(0, len(event_samples)):
        start_eeg_index = epoch_start_indexes[event_index]
        stop_eeg_index = start_eeg_index + epoch_durations
        eeg_epochs[event_index,:,:]  \
                = eeg_data[:,start_eeg_index:stop_eeg_index]
        is_trial_15Hz[event_index] =   event_types[event_index] == '15hz'
        
    return eeg_epochs, epoch_times, is_trial_15Hz

#%% Cell 4: Take Fourier transform & get frequency spectra
def get_frequency_spectrum(eeg_epochs, fs, remove_DC=False):
    '''
    A function to calculate the Fourier transform on each channel in each epoch.
    An optional parameter, remove_DC has been added to allow users to indicate 
    whether they would like to remove the DC offset from EEG signal. This can 
    be useful because as the DC offset is anmartifacts and may be large which 
    will unnecessarily decrease the normalized power of the signal. Setting 
    this variable to False retains the DC offset.
    
    Parameters
    ----------
    eeg_epochs : Array of float. Size (E,C,T) where E is the number of epochs,
    C is the number of EEG channels, and T is time points. Required.
        EEG data in uV.
    fs : Int, required.
        The sampling frequency in Hz.
    remove_DC: Bool, optional.
        An indication of whether or not to remove the DC offset. The default is 
        False.

    Returns
    -------
    eeg_epochs_fft : Array of complex float. Size (E,C,F) where E is the number 
    of epochs, C is the number of EEG channels, and F is the number of frequencies.
    The number of frequencies is equal to (number of time points/2)+1 when the 
    number of time points is even (as it is in our data) and is equal to 
    (number of time points +1)/2 if the number of time points is odd.
        Fourier transformed EEG epochs. Values represent the frequency spectra.
    fft_frequencies : Array of float. Size (F,) where F is the number of 
    frequencies. 
        Frequencies corresponding to columns in eeg_epochs_fft such that a 
        frequency spectrum value within eeg_epochs in column i corresponds to
        the frequency in fft_frequencies at row i. Frequencies range from 0 to
        fs/2 where fs is the sampling frequency, in accordance with the Nyquist 
        Criterion.

    '''
    # remove DC offset if specified by parameter
    if remove_DC:
        means = np.mean(eeg_epochs, axis = -1)
        for time_index in range(0,eeg_epochs.shape[-1]):
            eeg_epochs[:,:,time_index] -= means
            
    # perform Fourier transform on each channel in each epoch
    eeg_epochs_fft = np.fft.rfft(eeg_epochs, axis=-1)
    
    # get corresponding frequencies
    # d represents sample spacing (inverse of sample rate)
    fft_frequencies = np.fft.rfftfreq(eeg_epochs.shape[2], d=1/fs)
    
    return eeg_epochs_fft, fft_frequencies   

#%%
def plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_trial_15Hz, 
                        channels, channels_to_plot, subject):
    '''
    Plots the mean frequency spectrum across trials and saves an image of the 
    plot(s) to SSVEP_S{subject}_power_spectrum{figure_index).png within the 
    current directory. 

    Parameters
    ----------
    eeg_epochs_fft : Array of complex float. Size (E,C,F) where E is the number 
    of epochs, C is the number of EEG channels, and F is the number of frequencies.
    The number of frequencies is equal to (number of time points/2)+1 when the 
    number of time points is even (as it is in our data) and is equal to 
    (number of time points +1)/2 if the number of time points is odd. Required.
        Fourier transformed EEG epochs. Values represent the frequency spectra.
    fft_frequencies : Array of float. Size (F,) where F is the number of 
    frequencies. Required.
        Frequencies corresponding to columns in eeg_epochs_fft such that a 
        frequency spectrum value within eeg_epochs in column i corresponds to
        the frequency in fft_frequencies at row i. 
    is_trial_15Hz : Array of bool. Size (E,) where E is the number of epochs.
    Required.
        An indication of whether the light was flashing at 15 Hz during each
        epoch. True if the light was flashing, false otherwise.
    channels : Array of str. Size (CO,) where CO is the number of channels 
    included in the original dataset. Required.
        The name of each channel included in the original dataset ex. Fz.
    channels_to_plot : List of str.  Required.
        EEG data is plotted for these channels in subplots at 2 channels/figure.
    subject : Int, required.
        The number of the subject for whom data will be plotted.

    Returns
    -------
    spectrum_dB_12Hz : Array of float. Size (C,F) where E is the number of epochs
    and F is the number of frequencies.
        The mean power spectrum of 12 Hz trials in dB. 
    spectrum_dB_15Hz : Array of float. Size (C,F) where E is the number of epochs
    and F is the number of frequencies.
        The mean power spectrum of 15 Hz trials in dB.
        

    '''
    # convert spectra to units of power
    # we do not take the absolute value of the spectrum for each channel and
    # trial prior to this step becuase we are multiplying the value returned
    # by the fft by its complex conjugate to get the power. The calculation
    # of the absolute value is implicit in this statement and thus does not
    # need to be performed separately. 
    eeg_power = eeg_epochs_fft * np.conj(eeg_epochs_fft)
    
    # save event frequencies to variables (str and int)
    frequencies = ['12Hz', '15Hz']
    freqs = [12,15]
    
    # calculations for 12 Hz trials
    eeg_power_12Hz = eeg_power[~is_trial_15Hz,:,:]
    mean_power_12Hz = np.abs(np.mean(eeg_power_12Hz, axis=0))  # n_channels x n_time
    
    # storing variables for normalization
    max_power_12Hz = np.max(mean_power_12Hz, axis=1)   # in each channel
    channel_count = mean_power_12Hz.shape[0] 
    
    # initialize variable to hold normalized powers
    normalized_power_12Hz = np.zeros(mean_power_12Hz.shape)
    
    #normalize each channel
    for channel_index in range(0,channel_count):
        normalized_power_12Hz[channel_index,:]       \
                 = mean_power_12Hz[channel_index,:]   \
                       /max_power_12Hz[channel_index]
                       
    # convert to decibel units
    spectrum_dB_12Hz = 10 * np.log10(normalized_power_12Hz)
    
    # calculations for 15 Hz trials
    eeg_power_15Hz = eeg_power[is_trial_15Hz,:,:]
    mean_power_15Hz = np.abs(np.mean(eeg_power_15Hz, axis=0))
    
    # storing variables for normalization
    max_power_15Hz = np.max(mean_power_15Hz, axis=1)
    channel_count = mean_power_15Hz.shape[0]
    
    # initialize variable to hold normalized powers
    normalized_power_15Hz = np.zeros(mean_power_15Hz.shape)
    
    #normalize each channel
    for channel_index in range(0,channel_count) :
        normalized_power_15Hz[channel_index,:]      \
                 = mean_power_15Hz[channel_index,:]  \
                        /max_power_15Hz[channel_index]
    
    # convert to decibel units
    spectrum_dB_15Hz = 10 * np.log10(normalized_power_15Hz)
   
    #determine number of figures needed to plot the requested 
                                            # channels at 2 channels/figure
    figure_count = math.ceil(len(channels_to_plot)/2)
    for figure_index in range(0, figure_count):
        channel_pair = channels_to_plot[figure_index*2:figure_index*2 + 2]
        
        # initialize figure for a pair of channels
        plt.figure(clear=True)        
        ax1 = plt.subplot(211)
        
        for pair_index in range(0, len(channel_pair)):
            channel_name = channels_to_plot[figure_index*2 + pair_index]
            power_dB = spectrum_dB_12Hz[np.where(channels == channel_name)][0,:]
            for frequency_index in range(0, 2):
                plt.plot(fft_frequencies, power_dB,
                         label=frequencies[frequency_index])
                power_dB = spectrum_dB_15Hz[np.where(channels == channel_name)][0,:]
            
            #annotate graph
            plt.grid(True)
            plt.xlabel('Freqeuncy (Hz)')
            plt.ylabel('Power (dB)')
            plt.xlim(0, 80)
            plt.ylim(-80,0)
            plt.legend(loc='upper right')
            #mark 12Hz and 15Hz
            plt.plot([freqs[0], freqs[0]],[-80, -50],'k--')
            plt.plot([freqs[1], freqs[1]],[-80, -50],'k--')
            plt.title(f'Channel {channels_to_plot[figure_index*2+pair_index]} Frequency Content for SSVEP S{subject}')
            
            #second subplot
            if pair_index == 0:
                plt.subplot(212, sharex=ax1)
        
        plt.tight_layout()
        plt.savefig(f"SSVEP_S{subject}_power_spectrum{figure_index}.png")
    
    
    return spectrum_dB_12Hz, spectrum_dB_15Hz   
        
    
#%%
def plot_harmonics(spectrum_dB_12Hz, spectrum_dB_15Hz, channels, \
                   fft_frequencies, subject ):
    '''
    This calculates the mean of the 20 frequency bin above and below each 
    harmonic and returns the number of dB by which the harmonic exceeds its
    neighbors.  It does this for each electrode and then graphs the results. 
    Figures are saved to disk

    Parameters
    ----------
    spectrum_dB_12Hz : Array of float. Size (CO,F) where CO is the number of 
    channels in the original dataset and F is the number of frequencies. Required.
        The mean power spectrum of 12 Hz trials in dB.
    spectrum_dB_15Hz : Array of float. Size (CO,F) where CO is the number of 
    channels in the original dataset and F is the number of frequencies. Required.
        The mean power spectrum of 15 Hz trials in dB.
    channels : Array of str. Size (CO,) where CO is the number of channels 
    included in the original dataset. Required.
        The name of each channel included in the original dataset ex. Fz.
    fft_frequencies : Array of float. Size (F,) where F is the number of 
    frequencies. Required.
        Frequencies corresponding to columns in eeg_epochs_fft such that a 
        frequency spectrum value within eeg_epochs in column i corresponds to
        the frequency in fft_frequencies at row i. 
    subject : Int, required
        The number of the subject for whom data will be plotted.

    Returns
    -------
    None.

    '''

    channel_count = len(channels)
    df = fft_frequencies[1]-fft_frequencies[0]
    # harmonics to of 12 and 15 Hz to assess.
    harmonics = np.int32(np.array([[12,24,36,48],[15,30,45,60]])/df)
    harmonic_labels = ['Fundamental', '2nd Harmonic',
                       '3rd Harmonic', '4th Harmonic']
    
    power_dB = np.vstack((spectrum_dB_12Hz, spectrum_dB_15Hz))
    delta_dB = np.zeros((channel_count,2,4))
    near_by = 20  #compare harmonic magnitude to harmonic +/- near_by indicies   
 
    for channel_index in range(0,channel_count):
        for freq_index in range(0,2):    # 12 AND 15 Hz
            for harmonic_index in range(0,4):   #assess up to 4th hamonic
                fft_index = harmonics[freq_index, harmonic_index]
                chan_ind = channel_index + (freq_index * channel_count)
                mean_of_neighbors = ( np.sum(power_dB[chan_ind,               \
                                             fft_index-near_by:fft_index]     \
                                            ) +                               \
                                      np.sum(power_dB[chan_ind,               \
                                             fft_index+1:fft_index+1+near_by] \
                                            )                                 \
                                     ) / (2 * near_by)
                # dB of harmonic above neighbors    
                delta_dB[channel_index,freq_index,harmonic_index] =           \
                        power_dB[chan_ind, fft_index] - mean_of_neighbors
    
    # For 12 Hz harmonics                    
    plt.figure(figsize=(12,4), clear=True)          
    for harmonic_index in range(0,4):                 
        plt.plot(channels[np.array(range(0,len(channels)))],     \
                 delta_dB[:,0,harmonic_index], '.-',              \
                 label=harmonic_labels[harmonic_index])
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.xlabel('Channel Name')
    plt.xticks(rotation = 45)
    plt.ylabel('db Above Background')
    plt.title(f'dB Above Background, 12Hz Harmonics by Channel (S{subject})')
    plt.tight_layout()
    plt.savefig(f'harmmonics12S{subject}.png')

    
    # For 15 Hz harmonics          
    plt.figure(figsize=(12,4), clear=True)
    for harmonic_index in range(0,4):                 
        plt.plot(channels[np.array(range(0,len(channels)))],     \
                 delta_dB[:,1,harmonic_index], '.-',              \
                 label=harmonic_labels[harmonic_index])
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.xlabel('Channel Name')
    plt.xticks(rotation = 45)
    plt.ylabel('db Above Background')
    plt.title(f'dB Above Background, 15Hz Harmonics by Channel (S{subject})')
    plt.tight_layout()
    plt.savefig(f'harmmonics15S{subject}.png')
    plt.show()


                         
                        