#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 8 08:42:42 2024

test_filter_ssvep_data.py

This file serves as the test script for Lab 4 (Filtering). The data from Python's MNE SSVEP dataset are loaded utilizing a function developed in Lab 3. Subsequently, the functions described in filter_ssvep_data.py are called for subject 1. The bandpass filter is generated at both 12Hz and 15Hz frequencies, and the EEG data is then passed through each of these filters. The envelopes, therefore, may also be calculated for the filtered data that had undergone filtering for each of these stimulus frequencies. These envelopes are then compared graphically. Finally, the power spectra for raw, filtered, and envelope data are plotted under the consideration of the 15Hz filter to depict differences between the two stimuli frequencies within the frequency space.

Useful abbreviations:
    EEG: electroencephalography
    SSVEP: steady-state visual evoked potentials
    fs: sampling frequency
    FFT: Fast Fourier Transform

@authors: Claire Leahy and Ron Bryant
"""

# import packages
from import_ssvep_data import load_ssvep_data
from filter_ssvep_data import make_bandpass_filter, filter_data, get_envelope, plot_ssvep_amplitudes, plot_filtered_spectra

#%%  Part 1: Load the Data

# load in the data for subject 1
data_dict = load_ssvep_data(subject=1,data_directory='SsvepData/')

# extract data from the dictionary
eeg = data_dict['eeg'] # in volts, converted in functions (although this does not serve as an input anywhere)
channels = list(data_dict['channels'])
fs = data_dict['fs']
event_durations = data_dict['event_durations']
event_samples = data_dict['event_samples']
event_types = data_dict['event_types']
    
#%%  Part 2: Design a Filter

# filter and plot at 12Hz
filter_coefficients_12Hz = make_bandpass_filter(low_cutoff=11, high_cutoff=13, filter_type='hann', filter_order=1000, fs=fs)

# filter and plot at 15Hz
filter_coefficients_15Hz = make_bandpass_filter(low_cutoff=14, high_cutoff=16, filter_type='hann', filter_order=1000, fs=fs)

# # decreased order
# # filter and plot at 12Hz
# filter_coefficients_12Hz_low_order = make_bandpass_filter(low_cutoff=11, high_cutoff=13, filter_type='hann', filter_order=500, fs=fs,print_info=True)

# # filter and plot at 15Hz
# filter_coefficients_15Hz_low_order = make_bandpass_filter(low_cutoff=14, high_cutoff=16, filter_type='hann', filter_order=500, fs=fs, print_info=True)

# # increased order
# # filter and plot at 12Hz
# filter_coefficients_12Hz_high_order = make_bandpass_filter(low_cutoff=11, high_cutoff=13, filter_type='hann', filter_order=2000, fs=fs,print_info=True)

# # filter and plot at 15Hz
# filter_coefficients_15Hz_high_order = make_bandpass_filter(low_cutoff=14, high_cutoff=16, filter_type='hann', filter_order=2000, fs=fs, print_info=True)

'''
A)
    Gains for 12Hz filter: About -0.8dB for 12Hz, about -68dB for 15Hz
    Gains for 15Hz filter: About -42dB for 12Hz, about -1dB for 15Hz
    (Determined graphically.)
B)
    Higher order yields a narrower/sharper freqeuncy response, more oscillations in the impulse response. Frequency response becomes smoother with increased order. The "center" of the impulse response is also shifted as a consequency of changing the order (for example, doubling the order will cause the center to shift to twice the original time), while the peak of the frequency response always occurs at 12Hz or 15Hz for their respective filters.                                
'''

#%% Part 3: Filter the EEG Signals

# filter the data with the 12Hz filter (subject 1)  
filtered_data_12Hz = filter_data(data=data_dict, b=filter_coefficients_12Hz)    

# filter the data with the 15Hz filter (subject 1)
filtered_data_15Hz = filter_data(data=data_dict, b=filter_coefficients_15Hz)    

#%% Part 4: Calculate the Envelope

# envelope for the 12Hz filter (subject 1)
envelope_12Hz = get_envelope(data=data_dict, filtered_data=filtered_data_12Hz, channel_to_plot='Oz', ssvep_frequency='12')

# envelope for the 15Hz filter (subject 1)
envelope_15Hz = get_envelope(data=data_dict, filtered_data=filtered_data_15Hz, channel_to_plot='Oz', ssvep_frequency='15')

#%% Part 5: Plot the Amplitudes

# plot the SSVEP amplitudes (envelope) for both filters (subject 1)
plot_ssvep_amplitudes(data=data_dict, envelope_a=envelope_12Hz, envelope_b=envelope_15Hz, channel_to_plot='Oz', ssvep_freq_a=12, ssvep_freq_b=15, subject=1)

#%% Part 6: Examine the Spectra

# plot the spectra for channels Fz and Oz (subject 1) considering the 15Hz filter
plot_filtered_spectra(data=data_dict, filtered_data=filtered_data_15Hz, envelope=envelope_15Hz, channels_to_plot=['Fz','Oz'], subject=1)

'''
Describe how the spectra change at each stage and why.
    a. Raw data exhibit many peaks (stimuli, harmonics, artifacts, etc.). Power spectra depict prevelance of a frequency within a signal, so filtering will reduce the variety of frequencies in a signal, highlighted by the smoothed nature of the spectra after the raw data. Peak at 12Hz (centered around 15Hz filter) disappears at higher orders but is prominent at lower orders. Spectra changes to reflect most prominent signal frequencies within the dataset; for the 15Hz filter, 15Hz will demonstrate a peak centered at that point. Peaks about width of bandpass, higher order required for sharper peak. In envelope, higher frequencies get knocked out. Envelope takes frequency of amplitudes, and that is not limited to 15Hz; thus, there is no peak at 15Hz. Changes in amplitude are very small (low frequency), so the higher power occurs at lower frequencies and continues to decrease as frequencies increase. The difference between the 12Hz and 15Hz stimuli power spectra is dependent on normalization: This difference (where the 12Hz stimulus is higher than the 15Hz stimulus) appears when the dataset is normalized to itself (raw to raw, filtered to filtered, envelope to envelope). This phenomenon can also be observed when the DC component of the data is removed. However, if the data are all normalized to the raw data, little to no difference is observed for the filtered power spectra at the two stimuli for channel Oz. 
'''
