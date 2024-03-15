#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:42:42 2024

@author: Claire Leahy and Ron Bryant
"""
#%%  Part 1   Load data
import import_ssvep_data as imp
import filter_ssvep_data as fil
import numpy as np


# elective parameters.
unpack_data = True    # unpack data_dict for reference during development
print_info_part2 = True # if True, causes make_band_pass() to print frequency 
                        # response values that cna be interpolated to see 
                        # effect of filters on the 12Hz and 15Hz signals
print_envelope_amplitudes = True  # If True the mean amplitude of each 
                                  # envelope during the period of 12Hz and 
                                  # 15Hz stimuli is printed to the console. 

subject = 1         # choose patient -->   1 or 2



data_dict = imp.load_ssvep_data(subject=1)



if unpack_data:    # unpack data_dict to view in workspace
    eeg_data = data_dict['eeg']/1e-6   # convert to microvolts
    channels = data_dict['channels']   # channel names
    fs = data_dict['fs']               # sampling frequency
    event_times = data_dict['event_samples']/fs       # convert to seconds
    event_types = data_dict['event_types']            # frequency of event
    event_durations = data_dict['event_durations']/fs # convert to seconds
    T = eeg_data.shape[1]/fs     # Total time of recording
    t = np.arange(0,T,1/fs)      # time axis for full eeg

    
#%%  Part 2 Make plot bandpass filters

#   Set 12Hz filter parameters
low_cutoff = 11
high_cutoff = 13
filter_type = 'hann'
filter_order = 500
if ~unpack_data:
    fs = data_dict['fs']                # sampling frequency
    

coefficients_12Hz = fil.make_bandpass_filter(low_cutoff, high_cutoff, 
                                             filter_order, fs,
                                             print_info=print_info_part2)

#  Change cutoffs to 15Hz filter parameters 
low_cutoff = 14
high_cutoff = 16


coefficients_15Hz = fil.make_bandpass_filter(low_cutoff, high_cutoff, 
                                             filter_order, fs,
                                             print_info=print_info_part2)

'''  By interpolation of data printed to console with print_info=True
     Part 2, Question A)
         Gains for 12Hz filter:  12Hz is about -1dB,    15Hz is about -65dB
         Gains for 15Hz filter:  12Hz is about -40dB,   15Hz is about -1dB
     Part 2, Question B)
         From images:  Higher order -yields a narrower/sharper freqeuncy respoinse.
                                    -more ripples in the impulse response.
                                    
'''

#%% Part 3   Filter the EEG signal

  
filtered_12Hz = fil.filter_data(data_dict, coefficients_12Hz,
                                channel_to_plot='Oz', )    

filtered_15Hz = fil.filter_data(data_dict, coefficients_15Hz,
                                channel_to_plot='Cz', time_range=[200,300])    


#%%    Part 4   Calculate the envelope

envelope_12Hz = fil.get_envelope(data_dict, filtered_12Hz, 'Oz', '12')

envelope_15Hz = fil.get_envelope(data_dict, filtered_15Hz)
    

#%% Part 5  Plot the amplitudes

fil.plot_ssvep_amplitudes(data_dict, envelope_12Hz, envelope_15Hz,
                          'Oz', 12, 15, 1, 
                          print_amplitudes=print_envelope_amplitudes)


#%% Part 6   Evaluate spectra

bandpass = 12

fil.plot_filtered_spectra(data_dict, filtered_12Hz, envelope_12Hz,
                          bandpass, subject, ['Oz', 'Fz'])

