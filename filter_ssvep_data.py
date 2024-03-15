#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:36:51 2024

@author: Claire Leahy and Ron Bryant
"""
from matplotlib import pylab as plt
from scipy.signal import firwin, filtfilt, freqz, hilbert
import numpy as np

def make_bandpass_filter(low_cutoff, high_cutoff, filter_order, 
                         fs, filter_type='hann', print_info=False):
    '''
    Creates a finite impulse respoinse bandpass filter of Hanning type

    Parameters
    ----------
    low_cutoff : float
        DESCRIPTION.  Low cutoff in Hz
    high_cutoff :  float 
        DESCRIPTION.  High cutoff in Hz
    filter_order : Int
        DESCRIPTION.  Order of filter
    fs : TYPE   int
        DESCRIPTION.   Sampling freqeuncy
    filter_type :  optional string
        DESCRIPTION. The default is 'hann'.  Other choices should be valid 
                    choices for scipy function firwin().
    print_info : optional Boolean
        DESCRIPTION. The default is False.  If True prints information 
                    to console that is helpful for answering Part 2 questions

    Returns
    -------
    filter_coefficinets: 1D array of floats
        DESCRIPTION.   Coefficients suitable for filtering with scipy 
                        filter functions.

    '''

    
    
    #Todo  check Y-labels,  check order and the +1
    
    # make filter
    Nyq = fs/2
    filter_coefficients = firwin(filter_order + 1, 
                                 [low_cutoff/Nyq, high_cutoff/Nyq], 
                                 window=filter_type, pass_zero = 'bandpass')

    # get frequency response parameters
    filter_frequency, filter_response = freqz(filter_coefficients, fs=fs)
    response_dB = 10*np.log10( filter_response * np.conj(filter_response))

    plt.figure(figsize=(8,6), clear=True) 
    
    plt.subplot(211)
    plt.title ('Impulse Response')
    plt.plot(np.arange(0,len(filter_coefficients))/fs, filter_coefficients)
    plt.grid()
    plt.xlabel('Time (sec.)')
    plt.ylabel('Gain')
        
    plt.subplot(212)
    plt.title('Frequency Response')
    plt.plot(filter_frequency, response_dB)
    plt.xlim(0,40)
    plt.ylim(-150,0)
    plt.xlabel('Frequency (Hz.)')
    plt.ylabel('Amplitude Gain (dB)')
    
    plt.suptitle(f'Bandpass Hamming Filter with fc=[{low_cutoff}, {high_cutoff}], order={filter_order+1}')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'han_filter_{low_cutoff}-{high_cutoff}Hz_order{filter_order}')
    plt.show()
    
    # print 12 Hz and 15 Hz gains to console
    if print_info:
        def interpolate(xs,ys, newx):
            slope = (ys[1] - ys[0])/(xs[1] - xs[0])
            return ys[0] + slope * (newx-xs[0])
        mask = filter_frequency > 12
        index12 = np.where(mask)[0][0]
        dB12 = interpolate([filter_frequency[index12-1],filter_frequency[index12]],
                           [response_dB[index12-1],response_dB[index12]],
                           12),
        mask = filter_frequency > 15
        index15 = np.where(mask)[0][0]
        dB15 = interpolate([filter_frequency[index15-1],filter_frequency[index15]],
                           [response_dB[index15-1],response_dB[index15]],
                           15),
        mask = (filter_frequency >= 11) & (filter_frequency <= 16)
        indicies = np.where(mask)[0]
        print(f'\nBandpass {round(low_cutoff/2+high_cutoff/2)}HZ, order={filter_order}')
        print(f'filter freqs = {filter_frequency[indicies]}')
        print(f'filter response = {response_dB[indicies]}')
        print(f' at 12Hz {dB12} dB   at 15Hz {dB15} dB')
    return filter_coefficients


def filter_data (data_dict, filter_coefficients, 
                 channel_to_plot=None, time_range=[148,164] ):
    '''
    Applies a scipy filtfilt function to the data in data_dict using the 
    supplied coefficients

    Parameters
    ----------
    data_dict : A numpy.lib.npyio.NpzFile data dictionary. 
        DESCRIPTION.   Includes eeg_data and necessary parameters as described
                    in Lab 3 protocol.  eeg_data is assumed to be in Volts
    filter_coefficients : 1D array of floats
        DESCRIPTION.  A valid set of coefficients for the scipy filter 
                    functions
    channel_to_plot : optional two character string
        DESCRIPTION. The default is None.  If specified the result of filtering
                    the eeg_data (in data_dict) is plotted
    time_range : optional list of  two integers  (seconds)
        DESCRIPTION. The default is [148,164]. A low and high time within
                the length of the EEG data.   The interval is plotted.

    Returns
    -------
    filtered_data : 2D array of floats  C_channels x T_time_points
        DESCRIPTION.  Returns each channel, bandpass filtered and converted
                    to microvolts.

    '''
    #initialize output
    eeg_data = data_dict['eeg']/1e-6   # convert to microvolts
    filtered_data = np.zeros_like(eeg_data)
    for channel_index in range(len(eeg_data)):
        filtered_data[channel_index,:] = filtfilt(filter_coefficients,1,
                                                  eeg_data[channel_index,:])
    
    if channel_to_plot != None:
        channels = data_dict['channels']
        channel_index = np.where(channels == channel_to_plot)[0][0]
        fs = data_dict['fs']              # sampling frequency
        T = filtered_data.shape[1]/fs     # Total time
        t = np.arange(0,T,1/fs)           # time axis
 
        plt.figure(figsize=(8,6), clear=True)
        
        ax1=plt.subplot(211)
        plt.plot(t, eeg_data[channel_index]-np.mean(eeg_data[channel_index])) 
        plt.xlim(time_range )
        plt.xlabel('Time (sec.)')
        plt.ylabel('Voltage (\u03BCV)') 
        plt.title(f'Unfiltered EEG Data, Channel {channel_to_plot} (mean voltage removed) ') 
        plt.grid()
        
        plt.subplot(212, sharex=ax1)
        plt.plot(t,filtered_data[channel_index])    
        plt.xlim(time_range) 
        plt.xlabel('Time (sec.)')
        plt.ylabel('Voltage (\u03BCV)') 
        plt.title('Filtered EEG Data, Channel {channel_to_plot}') 
        plt.grid()
        plt.tight_layout()
    
    return filtered_data


def get_envelope(data_dict, filtered_data,
                 channel_to_plot=None, ssvep_frequency='12'):
    '''
    Given a bandpass filtered group of eeg signals it returns the enclosing 
    envelope of each. If a channel is selected the data for that channel is
    graphed.

    Parameters
    ----------
    data_dict : A numpy.lib.npyio.NpzFile data dictionary. 
        DESCRIPTION.   Includes eeg_data and necessary parameters as described
                    in Lab 3 protocol.  eeg_data is assumed to be in Volts
    filtered_data : 2D array of floats  N_channels x T_time_points
        DESCRIPTION.  Returns each channel, bandpass filtered and converted
    channel_to_plot : optional two character string
        DESCRIPTION. The default is None.  If specified the result of filtering
                    the eeg_data (in data_dict) is plotted
    ssvep_frequency : optional integer
        DESCRIPTION. The default is '12'.  It represent the simulus frequency
                    of the SSVEP  -- either 12 or 15Hz

    Returns
    -------
    envelope : 2D array of floats  C_channels x T_time_points
        DESCRIPTION.  The evevelope of each bandpass filtered signal in 
                    microvolts

    '''
    envelope = np.zeros_like(filtered_data)
    for channel_index in range(filtered_data.shape[0]):
        envelope[channel_index] = abs(hilbert(filtered_data[channel_index]))
    
    if channel_to_plot != None:
        fs = data_dict['fs']                # sampling frequency
        T = filtered_data.shape[1]/fs     # Total time
        t = np.arange(0,T,1/fs)      # time axis
        channels = data_dict['channels']
        channel_index = np.where(channels == channel_to_plot)[0][0]

        plt.figure(figsize=(8,6), clear=True)
        plt.plot(t, envelope[channel_index])
        plt.plot(t, -envelope[channel_index])
        plt.plot(t, filtered_data[channel_index])
        plt.xlabel('Time (sec.)')
        plt.ylabel('Voltage (\u03BCV)') 
        plt.title('Filtered {ssvep_frequency}Hz EEG Data with Envelope, Channel {channel_to_plot}') 
        plt.grid()
        plt.show()

    return  envelope  



def plot_ssvep_amplitudes(data_dict, envelope_a, envelope_b,
                          channel_to_plot, ssvep_freq_a, ssvep_freq_b,
                          subject, print_amplitudes=False):
    '''
    Plots the envelope amplitude sof the two filtered (12 or 15 Hz) signals
    together and adjacent to a graph indicating the period of times when each 
    flash frequency occured.

    Parameters
    ----------
    data_dict : A numpy.lib.npyio.NpzFile data dictionary. 
        DESCRIPTION.   Includes eeg_data and necessary parameters as described
                    in Lab 3 protocol.  eeg_data is assumed to be in Volts
    envelope_a : 2D array of floats  C_channels x T_time_points
        DESCRIPTION.  The envelope of each bandpass (first frequency)
                    filtered signal in microvolts.
    envelope_b : 2D array of floats  C_channels x T_time_points
        DESCRIPTION.  The envelope of each bandpass (second frequency)
                    filtered signal in microvolts.
    channel_to_plot : two character string
        DESCRIPTION. The channel of the EEG envelope to plot
    ssvep_freq_a : Integer
        DESCRIPTION.  Corresponds to the frequency of bandpass filter in 
                    envelope_a
    ssvep_freq_b : Integer
        DESCRIPTION.  Corresponds to the frequency of bandpass filter in 
                    envelope_b
    subject : Integer
        DESCRIPTION.  Number of the subject of origin for the eeg data.
    print_amplitudes : optional Boolean
        Description.  Default is False.  If True the mean amplitude of each 
                envelope during the period of 12Hz and 15Hz stimuli is 
                printed to the console. 

    Returns
    -------
    None.

    '''
    fs = data_dict['fs']           # sampling frequency
    T = envelope_a.shape[1]/fs     # Total time
    t = np.arange(0,T,1/fs)        # time axis

    event_times = data_dict['event_samples']/fs        # convert to seconds
    event_types = data_dict['event_types']             # frequency of event
    event_durations = data_dict['event_durations']/fs  # convert to seconds

    channels = data_dict['channels']
    channel_index = np.where(channels == channel_to_plot)[0][0]

    ax1 = plt.subplot(211)
    for event_index in range(0,len(event_times)):
        event_time = [event_times[event_index],  
                      event_times[event_index]+event_durations[event_index]]
        event_frequency = [event_types[event_index],event_types[event_index]]
        plt.plot(event_time, event_frequency, 'b.-')
    plt.grid()
    plt.ylabel('Flash Frequency')
    plt.xlabel('Time (Sec.)')
    plt.title('Frequency and Times Stimuli are Active')

    plt.subplot(212, sharex=ax1)
    plt.plot(t,envelope_a[channel_index,:], label=f'{ssvep_freq_a}Hz Envelope')
    plt.plot(t,envelope_b[channel_index,:], label=f'{ssvep_freq_b}Hz Envelope')
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Voltage (\u03BCV)') 
    plt.title('Envelope Compaarison')
    plt.suptitle(f'Subject {subject} SSVEP Amplitudes')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    if print_amplitudes:
        #from inspection of the figure  
        start12Hz_index = 7*int(fs)
        stop12Hz_index = 228*int(fs)   # = start15Hz_index
        stop15Hz_index = 452*int(fs)
        
        print(f'Envelope Amplitudes on Channel {channel_to_plot} (mean+/-sd)')
        print('  During 12Hz stimulation:')
        mean12 = np.round(np.mean(envelope_a[channel_index, start12Hz_index:stop12Hz_index]),2)
        mean15 = np.round(np.mean(envelope_b[channel_index, start12Hz_index:stop12Hz_index]),2)  
        sd12 = np.round(np.std(envelope_a[channel_index, start12Hz_index:stop12Hz_index]),2)
        sd15 = np.round(np.std(envelope_b[channel_index, start12Hz_index:stop12Hz_index]),2)
        print(f'   Mean 12Hz amplitude is {mean12}  (+/- {sd12})')
        print(f'   Mean 15Hz amplitude is {mean15}  (+/- {sd15})')
        
        print('  During 15Hz stimulation:')
        mean12 = np.round(np.mean(envelope_a[channel_index, stop12Hz_index:stop15Hz_index]),2)
        mean15 = np.round(np.mean(envelope_b[channel_index, stop12Hz_index:stop15Hz_index]),2)  
        sd12 = np.round(np.std(envelope_a[channel_index, stop12Hz_index:stop15Hz_index]),2)
        sd15 = np.round(np.std(envelope_b[channel_index, stop12Hz_index:stop15Hz_index]),2)
        print(f'   Mean 12Hz amplitude is {mean12}  (+/- {sd12})')
        print(f'   Mean 15Hz amplitude is {mean15}  (+/- {sd15})')
        
        
    return


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
