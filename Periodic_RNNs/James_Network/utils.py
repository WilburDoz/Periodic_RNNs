#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import parameters_will
import torch

def generate_data(params, freqs = [], jitter = []):
    trial_len = params.data.min_length + np.random.randint(params.data.max_length - params.data.min_length)
    number_of_beats = params.data.min_beats + np.random.randint(params.data.max_beats - params.data.min_beats)
    if len(freqs) == 0:
        if params.data.freq_type == 0:
            freqs = params.data.min_freq + np.random.randint(params.data.max_freq-params.data.min_freq, size=[params.data.batch_size])
        else:
            freqs = np.random.choice(params.data.freqs, size = params.data.batch_size)
    elif len(freqs) != params.data.batch_size:
        print("Num freqs does not equal batch size!")
        
    if len(jitter) == 0:
        jitter = np.random.randint(freqs)
    elif len(jitter) != params.data.batch_size:
        print("Num jitters does not equal batch size!")
        
    output_traces = np.zeros([trial_len, params.data.batch_size])
    input_traces = np.zeros([trial_len, params.data.batch_size])
    input_or_rnn = np.zeros([trial_len, params.data.batch_size])
    
    for (freq_ind, freq) in enumerate(freqs):
        number_of_predictions = int(np.floor((trial_len-1-params.data.offset-jitter[freq_ind])/freq+1))
        if params.data.data_type == 0:
            output_traces[freq*np.arange(number_of_predictions) + jitter[freq_ind], freq_ind] = 1
            input_traces[freq*np.arange(min(number_of_beats,number_of_predictions)) + jitter[freq_ind]+params.data.offset, freq_ind] = 1
        elif params.data.data_type == 1:
            output_traces[:, freq_ind] = np.cos(np.arange(trial_len)*2*np.pi/freq + jitter[freq_ind][freq_ind])
            input_traces[:freq*number_of_beats, freq_ind] = np.cos(np.arange(freq*params.data.number_of_beats)*2*np.pi/freq + jitter[freq_ind])
        input_or_rnn[:freq*number_of_beats,freq_ind] = 1
        #input_or_rnn[:,:] = 1
        
    input_dict = parameters_will.DotDict()
    input_dict.observation = torch.from_numpy(input_traces).type(torch.float32)
    input_dict.outputs = torch.from_numpy(output_traces).type(torch.float32)
    input_dict.input_or_rnn = torch.from_numpy(input_or_rnn).type(torch.float32)
    
    params.model.seq_len = trial_len
    
    return input_dict

def generate_data_easy(params, freqs = [], jitter = []):
    trial_len = params.data.min_length + np.random.randint(params.data.max_length - params.data.min_length)
    number_of_beats = params.data.min_beats + np.random.randint(params.data.max_beats - params.data.min_beats)
    if len(freqs) == 0:
        if params.data.freq_type == 0:
            freqs = params.data.min_freq + np.random.randint(params.data.max_freq-params.data.min_freq, size=[params.data.batch_size])
        else:
            freqs = np.random.choice(params.data.freqs, size = params.data.batch_size)
    elif len(freqs) != params.data.batch_size:
        print("Num freqs does not equal batch size!")

    output_traces = np.zeros([trial_len, params.data.batch_size])
    
    for (freq_ind, freq) in enumerate(freqs):
        number_of_predictions = int(np.floor((trial_len-1-params.data.offset)/freq+1))
        output_traces[freq*np.arange(number_of_predictions), freq_ind] = 1
         
    input_dict = parameters_will.DotDict()
    input_dict.outputs = torch.from_numpy(output_traces).type(torch.float32)
    freq_inds = torch.zeros(len(freqs), dtype=int)
    for (freq_ind, freq) in enumerate(freqs):
        freq_inds[freq_ind] = np.where(torch.tensor(params.data.freqs) == freqs[freq_ind])[0][0]
    input_dict.freqs = freq_inds
    
    params.model.seq_len = trial_len
    
    return input_dict

def generate_osc_data(params):
    freqs = np.array(params.data.freqs)
    trial_len = params.data.min_length + np.random.randint(params.data.max_length - params.data.min_length)

    outputs = np.zeros([trial_len, len(freqs)])
    for (freq_counter, freq) in enumerate(freqs):
        outputs[freq-1::freq, freq_counter] = 1

    input_dict = parameters_will.DotDict()
    input_dict.freq = torch.from_numpy(freqs)
    input_dict.outputs = torch.from_numpy(outputs)

    params.model.num_freqs = 2
    
    return input_dict