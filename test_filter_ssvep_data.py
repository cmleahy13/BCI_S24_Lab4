#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 8 09:08:22 2024

@author: Ron Bryant and Claire Leahy
"""

# imports
from import_ssvep_data import load_ssvep_data
from filter_ssvep_data import make_bandpass_filter, filter_data, get_envelope, plot_ssvep_amplitudes

#%% Part 1: Load the Data

# load data for subject 1 as a dictionary
data = load_ssvep_data(subject=1, data_directory='SsvepData/')

#%% Part 2: Design a Filter

# generate 2 filters to keep data around 12Hz and 15Hz

'''
A) How much will 12Hz oscillations be attenuated by the 15Hz filter? How much will 15Hz oscillations be attenuated by the 12Hz filter?
B) Experiment with higher and lower order filters. Describe how changing the order changes the frequency and impulse response of the filter.
'''

#%% Part 3: Filter the EEG Signals

# call twice with each of the bandpass filters

#%% Part 4: Calculate the Envelope

# call for the 12Hz envelope at electrode Oz
envelope_12Hz = get_envelope(data, filtered_data, channel_to_plot='Oz', ssvep_frequency='12Hz')

# call for the 15Hz envelope at electrode Oz
envelope_15Hz = get_envelope(data, filtered_data, channel_to_plot='Oz', ssvep_frequency='15Hz')

#%% Part 5: Plot the Amplitudes

# generate the plots for subject 1 at electode Oz
plot_ssvep_amplitudes(data, envelope_a=envelope_12Hz, envelope_b=envelope_15Hz, channel_to_plot='Oz', ssvep_freq_a=12, ssvep_freq_b=15, subject=1)

# describe what is seen in the plots
# investigate and describe for other electrodes

#%% Part 6: Examine the Spectra

# generate the spectra
plot_filtered_spectra(data, filtered_data, envelope)

'''
Describe how spectra change at each stage and why (including but not limited to):
1. Why does the overall shape of the spectrum change after filtering?
2. In the filtered data on Oz, why do 15Hz trials appear to have less power than 12Hz trials at most frequencies? 
3. In the envelope on Oz, why do we no longer see any peaks at 15Hz?
'''
