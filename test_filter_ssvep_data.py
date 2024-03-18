#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 8 08:42:42 2024

@author: Claire Leahy and Ron Bryant
"""

# import packages
from import_ssvep_data import load_ssvep_data
from filter_ssvep_data import make_bandpass_filter, filter_data, get_envelope, plot_ssvep_amplitudes, plot_filtered_spectra

# global parameters
print_envelope_amplitudes = True  # if True, mean amplitude of each envelope during the period of 12Hz and 15Hz stimuli is printed to the console
subject = 1         # choose patient -->   1 or 2

#%%  Part 1: Load the Data

# load in the data for subject 1
data_dict = load_ssvep_data(subject=1,data_directory='SsvepData/')

# extract data from the dictionary
eeg = data_dict['eeg']
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

bandpass = 12

plot_filtered_spectra(data_dict, filtered_data_12Hz, envelope_12Hz,
                          bandpass, subject, ['Oz', 'Fz'])

