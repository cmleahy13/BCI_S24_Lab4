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
    FIR: Finite impulse response
    IIR: Infinite impulse response

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
# filter_coefficients_12Hz_low_order = make_bandpass_filter(low_cutoff=11, high_cutoff=13, filter_type='hann', filter_order=500, fs=fs)
# filter_coefficients_15Hz_low_order = make_bandpass_filter(low_cutoff=14, high_cutoff=16, filter_type='hann', filter_order=500, fs=fs)

# # increased order
# filter_coefficients_12Hz_high_order = make_bandpass_filter(low_cutoff=11, high_cutoff=13, filter_type='hann', filter_order=2000, fs=fs)
# filter_coefficients_15Hz_high_order = make_bandpass_filter(low_cutoff=14, high_cutoff=16, filter_type='hann', filter_order=2000, fs=fs)

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

    a. Power spectra depict the strength of any given frequency within a signal (dependent on the sampling frequency); considering filtering is effectively capable of "removing" certain frequencies from a signal, it follows that there would be distinct change in the shape of the power spectra from the raw to filtered signals. Broadly, the bandpass filter generated provides a filtered spectrum that appears much smoother and presents a distinct peak about the passband permitted by the filter (and tighter as the filter order increases).
    
    In the power spectrum for the raw data, many peaks are observed, including peaks at the stimuli frequencies and their harmonics as well as artifacts. When the (15Hz) filter is applied to the data, a single peak is observed for both the 12Hz and 15Hz stimuli, which occurs at 15Hz. The bandpass filter reduces the prominence (reflected by power) of signals frequencies outside of the passband, highlighting why the peak is observed at 15Hz, as, in the nearby frequencies, 15Hz peaks in the raw data power spectrum as well. At lower filter orders, a notable peak also occurs at 12Hz, though this peak disappears at higher filter orders, indicative of the sharpened cutoff effect that occurs at higher orders (for the cutoff occurring between 14Hz and 16Hz, higher orders will continue to approach those cutoff values more strictly). Within the filtered spectra, no peaks are observed at the known harmonic or artifact frequencies. Overall, this filter reduces the prominence of signals outside of the bandpass frequency window, and as filter order increases, the stronger that effect becomes on "outside" frequencies.
    
    The envelope of the signal comprises the magnitude of each of the points present in this signal. The changes in the amplitude occur at a very low frequency, and the higher frequencies become "knocked out". Thus, this particular feature of the signal is not largely dependent upon frequency, which is why the peak frequency is at 0Hz and continually decreases: The power of the signal at 0Hz will be highest since signal magnitude does not occur with a "set" frequency, and if it happens to, it is not a prevalent occurrence (making the power weaker at frequencies increasingly far from 0Hz). As such, no peak in power is observed at 15Hz.
    
    There are numerous ways in which the filtered spectra for the 12Hz and 15Hz stimuli may be compared to the raw data's spectra. First, there are multiple ways in which normalization may be performed; namely, each aspect of the data may be normalized to itself (i.e. the raw data normalized to the maximum within the raw data power spectrum, filtered data to the maximum within the filtered spectrum, etc.) or to the raw data's spectrum. When the spectra are normalized to themselves, the power for each set is less comparable to the other datasets but highlights important features related to each manipulation. For example, normalizing the power spectra of the filtered data to itself presents distinct power differences between the 12Hz and 15Hz stimuli. At most frequencies, the 12Hz stimulus presents with a higher power than the 15Hz stimulus at most frequencies for the 15Hz filter (except at 15Hz, where there is a very high peak in power). The increased power for the 12Hz stimulus could be reflective of several signals in the data, included not only the 12Hz stimulus but the alpha waves (occur between 8 and 12Hz). Though the frequencies outside of 14-16Hz are largely filtered out (especially at higher filter orders), the signals still possess higher "baseline" power, which can be seen where the 12Hz peak in the unfiltered signal has a higher power than the 15Hz signal.
    
    In contrast, it is also possible to observe the power spectra normalized to the raw data power spectrum. In channel Oz, the difference in the filtered power spectrum between the two signals mostly disappears, but the difference between the spectra for these stimuli in channel Fz increases (and shows the opposite pattern observed before). More importantly, utilizing this normalization factor allows for a more direct comparison across power spectra for the two stimuli. Notably, it can be seen that the peak power in the filtered spectrum at 15Hz (channel Oz) is the same power at 15Hz as in the unfiltered data's spectrum. Similarly, the envelope's spectrum appears to have a peak power at 0 with the amplitude potentially aligning with the highest power observed in the raw spectrum, which is the powerline artifact. Thus, the relationship between the raw spectrum and filtered spectrum at each frequency is easily observed regarding how the filter changed the power directly, while normalizing each spectrum to itself shows the relative effect the filter had (making trends/shapes more signficant, whereas exact values may be more informative with using the raw data normalization factor).
    
'''
