#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 8 09:07:48 2024

@author: Ron Bryant and Claire Leahy
"""

import numpy as np
from matplotlib import pyplot as plt

#%% Part 2: Design a Filter

def make_bandpass_filter(low_cutoff, high_cutoff, filter_type, filter_order, fs):
    
    return filter_coefficients, b

#%% Part 3: Filter the EEG Signals

def filter_data(data, b):
    
    return filtered_data

#%% Part 4: Calculate the Envelope

def get_envelope(data, filtered_data, channel_to_plot='None', ssvep_frequency='12Hz'):
    
    return envelope
    
#%% Part 5: Plot the Amplitudes

def plot_ssvep_amplitudes(data, envelope_a, envelope_b, channel_to_plot='None', ssvep_freq_a=12, ssvep_freq_b=15, subject=1):
    
#%% Part 6: Examine the Spectra

def plot_filtered_spectra(data, filtered_data, envelope):