import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,glob
import seaborn as sns

from pynwb import NWBHDF5IO


 

## set of ephys analysis functions designed to work with openephys data ()


#! general tool and functions for paths and stuff
def find_folder(mouse, date):
    locations = [r'E:\\', r'C:\Users\hickm\Documents']
    for location in locations:
        search_pattern = os.path.join(location, f'*{mouse}*{date}*')
        matching_folders = glob.glob(search_pattern)
        if matching_folders:
            return matching_folders[0]
    return None



def binary_path(recording, probe, band = 'ap'):
    # takes in an open ephys recording object and a probe name and returns the continuous binary object
    if band == 'ap': 
        if str(probe) == 'probeA':
            data = recording.continuous[1]
        elif str(probe) == 'probeB':
            data = recording.continuous[3]
        else:
            data = recording.continuous[5]
        return data

    elif band == 'lfp':
        if str(probe) == 'probeA':
            data = recording.continuous[1]
        elif str(probe) == 'probeB':
            data = recording.continuous[3]
        else:
            data = recording.continuous[5]
        return data
    else: 
        print('You got not bands. Get your paper up.')
        return None

# need to add a polarity parameter checker to NWB file that parses the contacts column
def choose_stim_parameter(trials, amp=-100, pulse_number = 1, pulse_duration=100):
    stim_times = trials.loc[
        (trials['amplitude'] == amp) &
        (trials['pulse_number'] == pulse_number) &
        (trials['pulse_duration'] == pulse_duration)
    ]['start_time']
    return np.array(stim_times)

def stim_dictionary(trials):
    parameters = {}
    for run in trials.run.unique():
        amp = np.array(trials.loc[trials.run == run].amplitude)[0]
        pulse_width = np.array(trials.loc[trials.run == run].pulse_duration)[0]
        contacts = np.array(trials.loc[trials.run == run].contacts)[0]
        parameters[int(run)] = f'amp: {amp} ua, pw: {pulse_width} us, contacts: {contacts}'
    return parameters

def load_nwb(recording_path):
    nwb_path = glob.glob(os.path.join(recording_path,'*.nwb'))[0]
    io = NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()
    trials = nwb.trials.to_dataframe()
    units = nwb.units.to_dataframe()
    return trials, units



## plotting raw data

def get_chunk(data, # can use binary_path output directly
              stim_times,
              pre = 100, # time in ms
              post = 500, # time in ms
              chs = np.arange(0,200,1), # channels
              output = 'response' # 'response, 'pre/post', 'all'
              ):
    """
    Takes in a continuous binary object and a list of stimulation times and returns a chunk of the data
    """
    sample_rate = data.metadata['sample_rate']
    pre_samps = int((pre/1000 * sample_rate))
    post_samps = int((post/1000 * sample_rate))
    total_samps = pre_samps + post_samps

    n_chs = len(chs)
    if output == 'response':
        response = np.zeros((np.shape(stim_times)[0],total_samps, len(chs)))
        stim_indices = np.searchsorted(data.timestamps,stim_times)
        for i, stim in enumerate(stim_indices):
            start_index = int(stim - ((pre/1000)*sample_rate))
            end_index = int(stim + ((post/1000)*sample_rate))   
            chunk = data.get_samples(start_sample_index = start_index, end_sample_index = end_index, 
                                selected_channels = chs)
            chunk = chunk - np.median(chunk, axis = 0 )
            response[i,:,:] = chunk

        return response
    
    elif output == 'pre/post':
        pre_response = np.zeros((np.shape(stim_times)[0],pre_samps, n_chs))
        post_response = np.zeros((np.shape(stim_times)[0],post_samps, n_chs))
        stim_indices = np.searchsorted(data.timestamps,stim_times)
        for i, stim in enumerate(stim_indices):
            start_index = int(stim - ((pre/1000)*sample_rate))
            end_index = int(stim + ((post/1000)*sample_rate))   

            pre_chunk = data.get_samples(start_sample_index = start_index, end_sample_index = stim, 
                                selected_channels = np.arange(0,n_chs,1))
            post_chunk = data.get_samples(start_sample_index = stim, end_sample_index = end_index, 
                                selected_channels = np.arange(0,n_chs,1))
            pre_chunk = pre_chunk - np.median(pre_chunk, axis = 0 )
            post_chunk = post_chunk - np.median(post_chunk, axis = 0 )
            pre_response[i,:,:] = pre_chunk
            post_response[i,:,:] = post_chunk
        return pre_response, post_response
    
    elif output == 'all':
        response = np.zeros((np.shape(stim_times)[0],total_samps, len(chs)))
        pre_response = np.zeros((np.shape(stim_times)[0],pre_samps, n_chs))
        post_response = np.zeros((np.shape(stim_times)[0],post_samps, n_chs))
        stim_indices = np.searchsorted(data.timestamps,stim_times)
        
        for i, stim in enumerate(stim_indices):
            
            start_index = int(stim - ((pre/1000)*sample_rate))
            end_index = int(stim + ((post/1000)*sample_rate))   

            pre_chunk = data.get_samples(start_sample_index = start_index, end_sample_index = stim, 
                                selected_channels = np.arange(0,n_chs,1))
            post_chunk = data.get_samples(start_sample_index = stim, end_sample_index = end_index, 
                                selected_channels = np.arange(0,n_chs,1))
            
            chunk = data.get_samples(start_sample_index = start_index, end_sample_index = end_index, 
                                selected_channels = chs)
            pre_chunk = pre_chunk - np.median(pre_chunk, axis = 0 )
            post_chunk = post_chunk - np.median(post_chunk, axis = 0 )
            chunk = chunk - np.median(chunk, axis = 0 ) 
            pre_response[i,:,:] = pre_chunk
            post_response[i,:,:] = post_chunk
            response[i,:,:] = chunk
        return pre_response, post_response, response
        
## PSTH and Raster like stuff
def psth_arr(spiketimes, stimtimes, pre=0.5, post=2.5,binsize=0.05,variance=True):
    '''
    Generates avg psth, psth for each trial, and variance for a list of spike times (usually a single unit)
    '''
    numbins = int((post+pre)/binsize)
    x = np.arange(-pre,post,binsize)

    bytrial = np.zeros((len(stimtimes),numbins-1))
    for j, trial in enumerate(stimtimes):
        start = trial-pre
        end = trial+post
        bins_ = np.arange(start,end,binsize)
        trial_spikes = spiketimes[np.logical_and(spiketimes>=start, spiketimes<=end)]
        hist,edges = np.histogram(trial_spikes,bins=bins_)
        if len(hist)==numbins-1:
            bytrial[j]=hist
        elif len(hist)==numbins:
            bytrial[j]=hist[:-1]
        if variance == True:
            var = np.std(bytrial,axis=0)/binsize/np.sqrt((len(stimtimes)))
            
    psth = np.nanmean(bytrial,axis=0)/binsize
    return(psth,bytrial,var)                            


## created in mine specifically for plotting 2D and 3D spatial plotting
def lfp_snap(pre_response, post_response): # trials x chs x samples
    pre_mean = np.mean(pre_response, axis = 0)
    pre_mean = np.mean(pre_mean, axis = 0)
    post_mean = np.mean(post_response, axis = 0)
    post_mean = np.mean(post_mean, axis = 0)

    post_over_pre = post_mean/pre_mean
    return post_over_pre

def fr_snap(units, stim_times, 
            pre, post, binsize = 0.05,
            measure = 'mean', # mean, min, max, var 
            per = 'unit'): 
    # takes the spike times from unitsDF and assigns them to a channel to return firing rate by channel around spike times
    # hopefully useful for 2d and 3d plots... 
    #outputs post/pre 
    #Todo: prolly should make this a Z score rather than a mean firing rate

    pre_mean = []
    post_mean = []
    pre_std = []
    post_std = []
    post_min = []
    post_max = []

    for i,times in enumerate(units.spike_times):
        pre_psth, pre_bytrial, pre_var = psth_arr(times, stim_times, 
                                          pre, 0, binsize = binsize)
        post_psth, post_bytrial, post_var = psth_arr(times, stim_times, 
                                          0, post, binsize = binsize)
        pre_mean.append(np.mean(pre_psth))
        post_mean.append(np.mean(post_psth))
        pre_std.append(np.std(pre_psth))
        post_std.append(np.std(post_psth))
        post_min.append(np.min(post_psth))
        post_max.append(np.max(post_psth))

    pre_mean = np.array(pre_mean)
    post_mean = np.array(post_mean) 
    pre_std = np.array(pre_std)
    post_std = np.array(post_std)
    post_min = np.array(post_min)
    post_max = np.array(post_max)
    if per == 'unit':
        unit_dict = {'mean': (post_mean/pre_mean), 
                     'std': (post_std/pre_std), 
                     'min': (post_min/pre_mean),
                     'max': (post_max/pre_mean)}
        return unit_dict
    if per == 'ch':
        units['mean'] = post_mean/pre_mean
        units['std'] = post_std/pre_std
        units['min'] = post_min/pre_mean
        units['max'] = post_max/pre_mean
        units.sort_values(by = ['probe', 'ch'])
        grouped_units = units.groupby(['probe', 'ch']).agg({'mean': 'mean', 'std': 'std', 'min': 'min', 'max': 'max'})
        return grouped_units

        
    

#plotting raw data

def plot_ap(data, stim_times, pre = 4, post = 20, 
            first_ch = 125, last_ch = 175, 
            probe = 'A', title = '', 
            spike_overlay = False, units = None ,
            n_trials = 10, spacing_mult = 350, 
            save = False, savepath = '', format ='png'):
    
    response = get_chunk(data, stim_times, pre = pre, post = post, chs = np.arange(first_ch,last_ch,2))
        
    if spike_overlay == True:
        sample_rate = data.metadata['sample_rate']
        total_samps = int((pre/1000 * sample_rate) + (post/1000 * sample_rate))
        stim_indices = np.searchsorted(data.timestamps,stim_times)
        spikes = np.array(units.loc[units.ch_evens > first_ch].loc[units.ch_evens < last_ch+2].loc[units.probe == probe].loc[units.group == 'good'].spike_times)
        spike_ch = np.array(units.loc[units.ch_evens > first_ch].loc[units.ch_evens < last_ch+2].loc[units.probe == probe].loc[units.group == 'good'].ch_evens)

        spike_dict = {}
        for i, stim in enumerate(stim_indices):
            start_index = int(stim - ((pre/1000)*sample_rate))
            end_index = int(stim + ((post/1000)*sample_rate))  
            window = data.timestamps[start_index:end_index]
            filtered_spikes = [spike_times[(spike_times >= window[0]) & (spike_times <= window[-1])] for spike_times in spikes]  
            spike_dict[i] = filtered_spikes

    ## plotting 
    
    trial_subset = np.linspace(0,len(stim_times)-1, n_trials) #choose random subset of trials to plot 
    trial_subset = trial_subset.astype(int)
   
    #set color maps
    cmap = sns.color_palette("crest",n_colors = n_trials)
    #cmap = sns.cubehelix_palette(n_trials)
    colors = cmap.as_hex()
    if spike_overlay == True:
        cmap2 = sns.color_palette("ch:s=.25,rot=-.25", n_colors = len(spikes))
        colors2 = cmap2.as_hex()
    fig=plt.figure(figsize=(16,24))
    time_window = np.linspace(-pre,post,(total_samps))
    for trial,color in zip(trial_subset,colors):
        for ch in range(0,int(last_ch - first_ch)/2): plt.plot(time_window,response[trial,:,ch]+ch*spacing_mult,color=color)
    
        if spike_overlay == True:
            for i,ch in enumerate(spike_ch): 

                if spike_dict[trial][i].size > 0:
                    for spike in spike_dict[trial][i]:
                        spike = spike - stim_times[trial]
                        plt.scatter(spike*1000, (spike/spike) + ((ch-(first_ch+1))/2)*spacing_mult, alpha = 0.5, color = colors2[i], s = 500)
    plt.gca().axvline(0,ls='--',color='r')       
    plt.xlabel('time from stimulus onset (ms)')
    plt.ylabel('uV')
    plt.title(title)
    
    if save == True:
        plt.gcf().savefig(savepath,format=format,dpi=600)
    
    return fig



