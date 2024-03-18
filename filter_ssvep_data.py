#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 8 08:36:51 2024

@author: Claire Leahy and Ron Bryant
"""

# import packages
from matplotlib import pylab as plt
from scipy.signal import firwin, filtfilt, freqz, hilbert
import numpy as np

#%% Part 2: Design a Filter

"""
    TODO:
        - Is filter_type optional? Listed as having a default but also listed prior to required(?) arguments
    
"""

def make_bandpass_filter(low_cutoff, high_cutoff, filter_type, filter_order, fs):
    '''
    Description
    -----------
    Function to create a finite impulse response bandpass filter of Hanning type. Plots the impulse and frequency responses using the filters.

    Parameters
    ----------
    low_cutoff : float
        The lower frequency to be used in the bandpass filter in Hz.
    high_cutoff : float 
        The higher frequency to be used in the bandpass filter in Hz.
    filter_type : str
        The finite impulse response filter of choice to use in the firwin() function.
    filter_order : int
        The order of the filter.
    fs : int
        The sampling freqeuncy in Hz.

    Returns
    -------
    filter_coefficients: array of floats, size (O+1)x1, where O is the filter order
        Numerator coefficients of the finite impulse response filter.

    '''
    
    # get filter coefficients
    nyquist_frequency = fs/2 # get Nyquist frequency to use in filter
    filter_coefficients = firwin(filter_order+1, [low_cutoff/nyquist_frequency, high_cutoff/nyquist_frequency], window='hann', pass_zero='bandpass')

    # get frequency response parameters
    filter_frequencies, frequency_responses = freqz(filter_coefficients, fs=fs)
    frequency_responses_dB=10*(np.log10(frequency_responses*np.conj(frequency_responses))) # use conjugate due to complex numbers

    # create figure
    plt.figure(figsize=(8,6), clear=True) 
    
    # impulse response (subplot 1)
    plt.subplot(2,1,1)
    plt.plot(np.arange(0,len(filter_coefficients))/fs, filter_coefficients)
    # subplot format
    plt.title ('Impulse Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Gain')
    plt.grid()
    
    # frequency response (subplot 2)
    plt.subplot(2,1,2)
    plt.plot(filter_frequencies, frequency_responses_dB)
    # subplot format
    plt.xlim(0,40)
    plt.title('Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude Gain (dB)')
    plt.grid()
    
    # general figure formatting
    plt.suptitle(f'Bandpass Hann Filter with fc=[{low_cutoff}, {high_cutoff}], order={filter_order+1}')
    plt.tight_layout()
    
    # save figure
    plt.savefig(f'hann_filter_{low_cutoff}-{high_cutoff}Hz_order{filter_order}')
    
    return filter_coefficients

#%% Part 3: Filter the EEG Signals

"""
    TODO:
        - Make filtering more efficient with array operations rather than loop?

"""

def filter_data(data, b):
    '''
    Description
    -----------
    Function that applies the scipy filtfilt() function to the data using the coefficients of the FIR filter.

    Parameters
    ----------
    data : dict, size F, where F is the number of fields (6)
        Data from Python's MNE SSVEP dataset as a dictionary object, where the fields are relevant features of the data.
    b : Array of floats, size (O+1)x1, where O is the filter order
        Numerator coefficients of the finite impulse response filter.

    Returns
    -------
    filtered_data : 2D array of floats, size CxS, where C is the number of channels and S is the number of samples
        The EEG data, in microvolts, filtered twice.

    '''
    
    # extract data from the dictionary
    eeg = data['eeg']*(10**6) # convert to microvolts
    
    # variables for sizing
    channel_count = len(eeg) # 1st dimension of EEG is number of channels
    sample_count = len(eeg.T) # 2nd dimension of EEG is number of samples
    
    # preallocate array
    filtered_data = np.zeros([channel_count, sample_count])
    
    # apply filter to EEG data for each channel
    for channel_index in range(channel_count):
        
        filtered_data[channel_index,:] = filtfilt(b=b, a=1, x=eeg[channel_index,:])
    
    return filtered_data

#%% Part 4: Calculate the Envelope
"""
    TODO:
        - Is ssvep_frequency optional? Listed after optional argument but does not specify a default --> What should the default be?
        - Docstrings
        - Scale of voltage?
"""

def get_envelope(data, filtered_data, channel_to_plot=None, ssvep_frequency='12'):
    '''
    Description
    -----------
    Given a bandpass filtered group of eeg signals it returns the enclosing envelope of each. If a channel is selected the data for that channel is graphed.

    Parameters
    ----------
    data : dict, size F, where F is the number of fields (6)
        Data from Python's MNE SSVEP dataset as a dictionary object, where the fields are relevant features of the data.
    filtered_data : 2D array of floats  N_channels x T_time_points
        Returns each channel, bandpass filtered and converted
    channel_to_plot : optional two character string
        The default is None.  If specified the result of filtering
                    the eeg_data (in data_dict) is plotted
    ssvep_frequency : optional integer
        The default is '12'.  It represent the simulus frequency of the SSVEP  -- either 12 or 15Hz

    Returns
    -------
    envelope : 2D array of floats  C_channels x T_time_points
        The evevelope of each bandpass filtered signal in microvolts.
    '''
    
    # extract necessary data from the dictionary
    channels = list(data['channels'])
    fs = data['fs']
    
    # variables for sizing
    channel_count = len(filtered_data) # 1st dimension is number of channels
    sample_count = len(filtered_data.T) # 2nd dimension is number of samples
    
    # preallocate the array
    envelope = np.zeros([channel_count, sample_count])
    
    # get the envelope for each channel
    for channel_index in range(channel_count):
        
        envelope[channel_index]=np.abs(hilbert(x=filtered_data[channel_index]))
    
    # plot the filtered data and envelope if given a channel to plot 
    if channel_to_plot != None:
        
        # time parameters
        T = filtered_data.shape[1]/fs # total time
        t = np.arange(0,T,1/fs) # time axis to plot
        
        # extract the index of the channel to plot
        channel_index = channels.index(channel_to_plot)
        
        # create figure
        plt.figure(figsize=(8,6), clear=True)
        
        # plotting
        plt.plot(t, filtered_data[channel_index], label='filtered signal')
        plt.plot(t, envelope[channel_index], label='envelope')
        
        # format figure
        plt.title(f'{ssvep_frequency}Hz BPF Data (Channel {channel_to_plot})') 
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (µV)') 
        plt.legend()
        plt.grid()
        
        # save figure (not given a specified title in lab handout)
        plt.savefig(f'{ssvep_frequency}Hz_BPF_data_channel_{channel_to_plot}')

    return envelope  

#%% Part 5: Plot the Amplitudes

"""

    TODO:
        - Docstrings

"""

def plot_ssvep_amplitudes(data, envelope_a, envelope_b, channel_to_plot, ssvep_freq_a, ssvep_freq_b, subject):
    '''
    Description
    -----------
    Plots the envelope amplitude sof the two filtered (12 or 15 Hz) signals together and adjacent to a graph indicating the period of times when each flash frequency occured.

    Parameters
    ----------
    data : dict, size F, where F is the number of fields (6)
        Data from Python's MNE SSVEP dataset as a dictionary object, where the fields are relevant features of the data.
    envelope_a : 2D array of floats  C_channels x T_time_points
        The envelope of each bandpass (first frequency) filtered signal in microvolts.
    envelope_b : 2D array of floats  C_channels x T_time_points
        The envelope of each bandpass (second frequency) filtered signal in microvolts.
    channel_to_plot : two character string
        The channel of the EEG envelope to plot
    ssvep_freq_a : int
        Corresponds to the frequency of bandpass filter in envelope_a.
    ssvep_freq_b : int
        Corresponds to the frequency of bandpass filter in envelope_b.
    subject : int
        Number of the subject of origin for the eeg data.

    Returns
    -------
    None.

    '''
    
    # extract data from the dictionary
    channels = list(data['channels']) # convert to list
    fs = data['fs']
    event_durations = data['event_durations']
    event_samples = data['event_samples']
    event_types = data['event_types']
    
    
    # time parameters
    T = envelope_a.shape[1]/fs # total time, same for envelope_b
    t = np.arange(0,T,1/fs) # time axis

    # extract the index of the channel to plot
    channel_index = channels.index(channel_to_plot)
    
    # find event samples and times
    event_intervals = np.zeros([len(event_samples),2]) # array to contain interval times
    event_ends = event_samples + event_durations # event_samples contains the start samples
    event_intervals[:,0] = event_samples/fs # convert start samples to times
    event_intervals[:,1] = event_ends/fs # convert end samples to times
    
    # initialize figure
    figure, sub_figure = plt.subplots(2, sharex=True)
    
    # top subplot containing flash frequency over span of event
    for event_number, interval in enumerate(event_intervals):
    
        # determine the event frequency to plot (y axis)
        if event_types[event_number] == "12hz":
            event_frequency = 12
    
        else: 
            event_frequency = 15
        
        # plottting the event frequency
        sub_figure[0].hlines(xmin=interval[0], xmax=interval[1], y=event_frequency, color='b') # line
        sub_figure[0].plot([interval[0], interval[1]], [event_frequency,event_frequency], 'bo') # start and end markers
    
    # format top subplot
    sub_figure[0].set_xlabel('Time (s)')
    sub_figure[0].set_ylabel('Flash Frequency')
    sub_figure[0].set_yticks([12,15])
    sub_figure[0].set_yticklabels(['12Hz','15Hz'])
    sub_figure[0].grid()

    # bottom subplot containing envelopes of the filtered signals
    sub_figure[1].plot(t,envelope_a[channel_index,:], label=f'{ssvep_freq_a}Hz Envelope')
    sub_figure[1].plot(t,envelope_b[channel_index,:], label=f'{ssvep_freq_b}Hz Envelope')
    
    # format bottom subplot
    sub_figure[1].set_title('Envelope Comparison')
    sub_figure[1].set_xlabel('Time (s)')
    sub_figure[1].set_ylabel('Voltage (µV)')
    sub_figure[1].grid()
    sub_figure[1].legend()
    
    # format figure
    plt.suptitle(f'Subject {subject} SSVEP Amplitudes')
    plt.tight_layout()
    
    # save figure (not given a specified title in lab handout)
    figure.savefig(f'subject_{subject}_SSVEP_amplitudes_channel_{channel_to_plot}')

#%% Part 6: Examine the Spectra

def get_frequency_spectrum(eeg_epochs, fs, is_trial_15Hz, 
                           max_power_12Hz=None, max_power_15Hz=None, 
                           remove_DC=False, is_raw_data=False):
    '''
    Calculated the frequency spectrum across each time averaged 12Hz and 
    15Hz epoch.  Results are in normalized dB.

    Parameters
    ----------
    eeg_epochs : 3D array of floats  E_epochs x C_channels x T_time_points
        DESCRIPTION. EEG voltages (microvolts) divided into epochs for each
                    channel.
    fs :    integer
        DESCRIPTION.   Sampling frequency
    is_trial_15Hz : 1D array of Booleans   1 x E_epochs
        DESCRIPTION.  Indicates if epoch stimulus is at 
                        15Hz (True) or 12Hz (False)
    max_power_12Hz: optional 1D array of floats    1 x C_channels
        DESCRIPTION. The default is None.  If not supplied, the data is 
            assumed to be raw eeg data, and this value is calculated as the 
            maximum power in the mean of the 12 Hz stimulated epochs for 
            each channel and the value is returned for use in normalizing
            the filtered and envelope data. 
    max_power_15Hz: optional 1D array of floats    1 x C_channels
        DESCRIPTION. The default is None.  If not supplied, the data is 
            assumed to be raw eeg data, and this value is calculated as the 
            maximum power in the mean of the 15 Hz stimulated epochs for 
            each channel and the value is returned for use in normalizing
            the filtered and envelope data. 
    remove_DC : optional  Boolean
        DESCRIPTION. The default is False.  Removes mean component of signal
                before taking FFT if True.
    is_data_raw : optional Boolean
        DESCRIPTION. The default is False.  If True then the two max_power_xxHz
                parameters must also be specified and the returned spectra
                are normalized by the correwsponding max_power_xxHz parameter. 

    Returns
    -------
    spectrum_dB_12Hz : 2D array of floats   C_channels x F_frequencies
        DESCRIPTION.  Frequency spectrum of signals of averaged 12Hz epochs 
                of each channel in dB normalized for highest frequency 
                amplitude.
    spectrum_dB_15Hz : 2D array of floats   C_channels x F_frequencies
        DESCRIPTION.  Frequency spectrum of signals of averaged 15Hz epochs 
                of each channel in dB normalized for highest frequency 
                amplitude.
    CONDITIONAL Returns
            If is_data_raw=True  the following are also returned
    --------------------
    fft_frequencies : 1D array of floats    1 x F_freqeuncies
        DESCRIPTION.  Frequencies corresponding to the two above spectra.
    max_power_12Hz : 1D array of floats    1 x C_channels
        DESCRIPTION.  The maximum power in the mean of the
            12 Hz stimulate epochs for each channel of the raw data. 
    max_power_15Hz : 1D array of floats    1 x C_channels
        DESCRIPTION.  The maximum power in the mean of the
            15 Hz stimulate epochs for each channel of the raw data. 
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
    
    # convert spectra to units of power
    eeg_power = eeg_epochs_fft * np.conj(eeg_epochs_fft)
        
    # calculations for 12 Hz Stimulus trials
    eeg_power_12Hz = eeg_power[~is_trial_15Hz,:,:]
    mean_power_12Hz = np.abs(np.mean(eeg_power_12Hz, axis=0))# n_channels x n_time
    if is_raw_data:    # storing variables for normalization
        max_power_12Hz = np.max(mean_power_12Hz, axis=1) # in each channel
    
    # calculations for 15 Hz Stimulus trials
    eeg_power_15Hz = eeg_power[is_trial_15Hz,:,:]
    mean_power_15Hz = np.abs(np.mean(eeg_power_15Hz, axis=0))
    if is_raw_data:     # storing variables for normalization
        max_power_15Hz = np.max(mean_power_15Hz, axis=1)
    
    # initialize variable to hold normalized powers
    normalized_power_12Hz = np.zeros(mean_power_12Hz.shape)
    
    #normalize each channel
    for channel_index in range(mean_power_12Hz.shape[0]):
        normalized_power_12Hz[channel_index,:]       \
                 = mean_power_12Hz[channel_index,:]   \
                       /max_power_12Hz[channel_index]
    # convert to decibel units
    spectrum_dB_12Hz = 10 * np.log10(normalized_power_12Hz)
    
    # initialize variable to hold normalized powers
    normalized_power_15Hz = np.zeros(mean_power_15Hz.shape)
    
    #normalize each channel
    for channel_index in range(mean_power_15Hz.shape[0]) :
        normalized_power_15Hz[channel_index,:]      \
                 = mean_power_15Hz[channel_index,:]  \
                        /max_power_15Hz[channel_index]
    # convert to decibel units
    spectrum_dB_15Hz = 10 * np.log10(normalized_power_15Hz)
   
    if is_raw_data:
        return spectrum_dB_12Hz, spectrum_dB_15Hz, fft_frequencies, \
                max_power_12Hz, max_power_15Hz
    else:
        return spectrum_dB_12Hz, spectrum_dB_15Hz







def plot_filtered_spectra(data_dict, filtered_data, envelope_data,
                          filtered_at, subject=1, 
                          channels_to_plot=['Oz', 'Fz'],
                          epoch_start_time=0, epoch_end_time=20):
    '''
    Two stimulus frequencies are compared in a single figure with a 2x3
    series of graphs (subplots.)  For two selected channels, data of the  
    raw, filtered (at a single frequency), and the filtered envelope are 
    graphically compared.

    Parameters
    ----------
    data_dict : A numpy.lib.npyio.NpzFile data dictionary. 
        DESCRIPTION.   Includes eeg_data and necessary parameters as described
                    in Lab 3 protocol.  eeg_data is assumed to be in Volts
    filtered_data : 2D array of floats  C_channels x T_time_points
        DESCRIPTION.  Returns each channel, bandpass filtered and converted
                    to microvolts.
    envelope_data : 2D array of floats  C_channels x T_time_points
        DESCRIPTION.  The evevelope of each bandpass filtered signal in 
                    microvolts
    filtered_at : integer
        DESCRIPTION.  Frequency of filter in Hz
    subject : Interger
        DESCRIPTION.  Number of the subject of origin for the eeg data.
    channels_to_plot : optional list of two strings
        DESCRIPTION. The default is ['Oz', 'Fz']. Correspoiond to the channels 
                in data_dict
    epoch_start_time : optional integer 
        DESCRIPTION. The default is 0. Time from begining of stimulus to  
                begin epoch.  Stimuli last 20 seconds.
    epoch_end_time : optional integer 
        DESCRIPTION. The default is 20. Time from begining of stimulus to  
                end epoch.  Stimuli last 20 secondss.
        
    Returns
    -------
    None.

    '''
    
# save event frequencies to variables (str and int)
#    frequencies = ['12Hz', '15Hz']
#    freqs = [12,15]

# unpack data_dict
    eeg_data = data_dict['eeg']/1e-6            # convert to microvolts
    fs = data_dict['fs']                        # sampling frequency
    event_samples = data_dict['event_samples']  # index to start of events
    event_types = data_dict['event_types']      # image frequency during event
    
# calculate epoch parameters
    epoch_start_indexes = event_samples + int(epoch_start_time * fs)
    epoch_durations = int((epoch_end_time - epoch_start_time) * fs)
    epoch_times = np.arange(epoch_start_time, epoch_end_time, 1/fs) #seconds
    epoch_count = len(event_samples)
    
# Determine indicies of the channels_to_plot 
    channels = data_dict['channels']
    channel_count = len(channels_to_plot)
    channel_indicies = np.zeros(channel_count).astype(int)
    channel_indicies[0] = np.where(channels == channels_to_plot[0])[0][0]
    channel_indicies[1] = np.where(channels == channels_to_plot[1])[0][0]

# initialize epoch variables to store epochs from channels_to_plot 
                # and indication of whether each trial is 15 Hz (True)
    raw_channel_epochs = np.zeros( ( epoch_count, 
                                     channel_count, 
                                     len(epoch_times) )
                                 ) 
    filtered_channel_epochs = np.zeros_like(raw_channel_epochs)
    envelope_channel_epochs = np.zeros_like(raw_channel_epochs)
    is_trial_15Hz = np.zeros(epoch_count, dtype=bool)
    
# Epoch the data
    for event_index in range(0, len(event_samples)):
        #get start and end indicies of the event
        start_eeg_index = epoch_start_indexes[event_index]
        stop_eeg_index = start_eeg_index + epoch_durations
        
        # add epoch to respective epoch variable and indicate stimulus frequency
        raw_channel_epochs[event_index,:,:]  \
                = eeg_data[channel_indicies,start_eeg_index:stop_eeg_index]
        
        filtered_channel_epochs[event_index,:,:]  \
                = filtered_data[channel_indicies,start_eeg_index:stop_eeg_index]
        
        envelope_channel_epochs[event_index,:,:]  \
                = envelope_data[channel_indicies,start_eeg_index:stop_eeg_index]
                
        is_trial_15Hz[event_index] = event_types[event_index] == '15hz'


# Calculate power spectrum (normalized after subtracting DC)
    ## Raw, filtered, and envelope are all normalized by the same factor
    ## derived from the raw data
    
    # Spectrum of raw and return freqeuncies and normalization factors
    fft_raw_data_12Hz, fft_raw_data_15Hz, fft_frequencies,       \
               max_power_12Hz, max_power_15Hz                     \
            =  get_frequency_spectrum(raw_channel_epochs,fs,       \
                                      is_trial_15Hz, remove_DC=True,\
                                      is_raw_data=True)
    
    # Specturm of filtered data
    fft_filtered_data_12Hz, fft_filtered_data_15Hz                    \
            =  get_frequency_spectrum(filtered_channel_epochs,fs,      \
                                      is_trial_15Hz,                    \
                                      max_power_12Hz=max_power_12Hz,     \
                                      max_power_15Hz=max_power_15Hz,      \
                                      remove_DC=True)

    # Spectrum of envelope
    fft_envelope_data_12Hz, fft_envelope_data_15Hz                    \
            =  get_frequency_spectrum(envelope_channel_epochs,fs,      \
                                      is_trial_15Hz,                    \
                                      max_power_12Hz=max_power_12Hz,     \
                                      max_power_15Hz=max_power_15Hz,      \
                                      remove_DC=True)


# Plot the data
    plt.figure(figsize=(10,8), clear=True )
    
    # Raw data, 1st Channel (default Oz)
    ax1=plt.subplot(231)            
    plt.plot(fft_frequencies, fft_raw_data_12Hz[0], label='12Hz Stimulus')
    plt.plot(fft_frequencies, fft_raw_data_15Hz[0], label='15Hz Stimulus')
    plt.xlim(0,60)
    plt.ylim(-70,0)
    plt.ylabel(f'Power (dB) in Channel {channels[channel_indicies[0]]}')
    plt.xlabel('Frequency (Hz.)')
    plt.title('Unfiltered')
    plt.grid()
    
    # Filtered data, 1st Channel (default Oz)
    plt.subplot(232, sharex=ax1, sharey=ax1)    
    plt.plot(fft_frequencies, fft_filtered_data_12Hz[0])
    plt.plot(fft_frequencies, fft_filtered_data_15Hz[0])
    plt.xlabel('Frequency (Hz.)')
    plt.title(f'Bandpass Filter at {filtered_at} Hz')
    plt.grid()

    # Envelope data, 1st Channel (default Oz)
    plt.subplot(233, sharex=ax1, sharey=ax1)    
    plt.plot(fft_frequencies, fft_envelope_data_12Hz[0], label='12Hz Stimulus')
    plt.plot(fft_frequencies, fft_envelope_data_15Hz[0], label='15Hz Stimulus')
    plt.xlabel('Frequency (Hz.)')
    plt.title('Envelope of Filtered Signal')
    plt.legend()
    plt.grid()

    # Raw data, 2nd Channel (default Fz)
    plt.subplot(234, sharex=ax1, sharey=ax1)    
    plt.plot(fft_frequencies, fft_raw_data_12Hz[1])
    plt.plot(fft_frequencies, fft_raw_data_15Hz[1])
    plt.ylabel(f'Power (dB) in Channel {channels[channel_indicies[1]]}')
    plt.xlabel('Frequency (Hz.)')
    plt.title('Unfiltered')
    plt.grid()

    # Filtered data, 2nd Channel (default Fz)
    plt.subplot(235, sharex=ax1, sharey=ax1)    
    plt.plot(fft_frequencies, fft_filtered_data_12Hz[1])
    plt.plot(fft_frequencies, fft_filtered_data_15Hz[1])
    plt.xlabel('Frequency (Hz.)')
    plt.title(f'Bandpass Filter at {filtered_at} Hz')
    plt.grid()

    # Envelope data, 2nd Channel (default Fz)
    plt.subplot(236, sharex=ax1, sharey=ax1)    
    plt.plot(fft_frequencies, fft_envelope_data_12Hz[1])
    plt.plot(fft_frequencies, fft_envelope_data_15Hz[1])
    plt.xlabel('Frequency (Hz.)')
    plt.title('Envelope of Filtered Signal')
    plt.grid()
    plt.tight_layout()
    
    plt.savefig(f'filt_spec_{channels[channel_indicies[0]]}_{channels[channel_indicies[1]]}',)    
    plt.show()
    return 
#%% Part 6: Examine the Spectra

def plot_filtered_spectra(data, filtered_data, envelope):
    '''
    Description
    -----------

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    filtered_data : TYPE
        DESCRIPTION.
    envelope : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    plt.subplot(2,3,1) # Fz, raw
    plt.subplot(2,3,2) # Fz, filtered
    plt.subplot(2,3,3) # Fz, envelope
    plt.subplot(2,3,4) # Oz, raw
    plt.subplot(2,3,5) # Oz, filtered
    plt.subplot(2,3,6) # Oz, envelope
    