from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject
from dlab.nwbtools import option234_positions,load_unit_data,make_spike_secs
from datetime import datetime
from dateutil.tz import tzlocal
from scipy import stats
from open_ephys.analysis import Session
import re
import os, glob
import numpy as np
import pandas as pd

from herbs_processing import neuropixel_coords, stim_coords, load_herbs
from ephys_tools import find_folder


def sort_runs(paths):  # sort stimulation runs by time to get runs in order 
    new_names = []
    for path in paths:
        session_time_string = os.path.basename(path).split('_')[-1]
        if len(session_time_string.split('_')[-1]) < 2:
            hour = '0'+ session_time_string.split('_')[-1]
        else: hour = session_time_string.split('_')[-1]
        if len(session_time_string.split('-')[0]) < 2:
            minute = '0'+ session_time_string.split('-')[0]
        else: minute = session_time_string.split('-')[0]
        if len(session_time_string.split('-')[1]) < 2:
            second = '0'+ session_time_string.split('-')[1]
        else: second = session_time_string.split('-')[1]
#         os.path.basename(path).split('-')[0]+'-'+
        new_names.extend([hour+'_'+minute+'_'+second])
    return np.array(paths)[np.argsort(new_names).astype(int)]

def make_stim_df(mouse, date):
    directory = find_folder(mouse, date)
    session = Session(directory)
    recording = session.recordnodes[0].recordings[0]
    df = recording.events
        ## get timestamps for each run into a dictionary
    stim_dict = {}
    run = np.array(df.loc[df.line == 4].loc[df.state == 1].timestamp)
    stim = np.array(df.loc[df.line == 5].loc[df.state == 1].timestamp)
    for i in range(len(run) - 1):
        run_start = run[i]
        run_end = run[i+1]
        stim_dict[i] = [p for p in stim if run_start < p < run_end]
    last_run_start = run[-1]
    stim_dict[len(run)-1] = [p for p in stim if last_run_start < p]

    csv_path = os.path.join(r'C:\Users\hickm\Documents\Stim_CSVs',f'{date}_{mouse}')

    extension = 'csv'
    os.chdir(csv_path)
    result = glob.glob('*.{}'.format(extension))
    #print(result)

    sorted_dbs_runs = sort_runs(result)
    stim_df = pd.DataFrame()
    pd.read_csv(sorted_dbs_runs[0])
    for i, run in enumerate(sorted_dbs_runs):
        lil = pd.read_csv(run)
        lil['Run'] = i    
        #lil = pd.concat([lil] * int(lil.TrainQuantity)).sort_index().reset_index(drop=True)
        lil = pd.concat([lil] * int(len(stim_dict[i])))
        stim_df = pd.concat([stim_df,lil],ignore_index=True)

    concat_df = pd.concat([pd.read_csv(runs) for runs in sorted_dbs_runs])
    os.chdir(directory)
    concat_df.to_csv(f'{mouse}.csv')
    trial_list = []
    for run, trials in stim_dict.items():
        for trial in trials:
            trial_list.append(trial)

    if len(trial_list) == len(stim_df):
        stim_df['stim_time'] = trial_list
        stim_df.to_csv(f'{mouse}_bytrial.csv')
        return stim_df
    else:
        return print('Trials not equal to Dataframe Length, dumbass')


def get_contacts(df):
    try:
        strings = df['comment']
    except:
        strings = df.contacts

    contact_negative = []
    contact_positive = []
    polarity = []
    
    for string in strings:
        r_number = re.search(r'(\d+)r', string)
        r_value = int(r_number.group(1)) if r_number else None
        contact_negative.append(r_value)

        b_number = re.search(r'(\d+)b', string)
        b_value = int(b_number.group(1)) if b_number else 0
        contact_positive.append(b_value)

        if b_value == 0:
            polarity.append("monopolar")
        else:
            polarity.append("bipolar")
    
    try:
        df['contact_negative'] = contact_negative
        df['contact_positive'] = contact_positive
        df['polarity'] = polarity
        
    except: 
        df['contact_negative'] = contact_negative
        df['contact_positive'] = contact_positive
        df['polarity'] = polarity
    return df
 


def make_nwb(mouse,
             date, 
             experimenter = 'jlh', 
             experiment_description = 'Electrical Brain Stimulation'):
    
    subject = Subject(**{'subject_id': mouse,
                    'age': '8-12 weeks',
                    'strain': 'C57/B6',
                        'description': 'craniotomy',
                        'sex': 'Male',
                        'species': 'Mus musculus'})
    directory = find_folder(mouse, date)
    try: 
       stim_df =  pd.read_csv(os.path.join(directory,f'{mouse}_bytrial.csv'))
    except FileNotFoundError:
        stim_df = make_stim_df(mouse, date)
    
    stim_df = get_contacts(stim_df) # adds 'contact_negative, 'contact_positive', and 'polarity' to df 

    

    #UNITS_DF stuff
    probe_a = os.path.join(directory,'Record Node 105','Experiment1','recording1',
                        'continuous','Neuropix-PXI-104.ProbeA-AP')
    probe_b = os.path.join(directory,'Record Node 105','Experiment1','recording1',
                        'continuous','Neuropix-PXI-104.ProbeB-AP')
    probe_c = os.path.join(directory,'Record Node 105','Experiment1','recording1',
                        'continuous','Neuropix-PXI-104.ProbeC-AP')
    probes = [probe_a,probe_b,probe_c]

    for probe in probes:
        make_spike_secs(probe)
    a_df = load_unit_data(probe_a,probe_depth = 2000, spikes_filename = 'spike_secs.npy',probe_name = 'A')
    b_df = load_unit_data(probe_b,probe_depth = 2000, spikes_filename = 'spike_secs.npy',probe_name = 'B')
    c_df = load_unit_data(probe_c,probe_depth = 2000, spikes_filename = 'spike_secs.npy',probe_name = 'C')
    units_df = pd.concat([a_df,b_df,c_df])
    
    cluster_info_a = pd.read_csv(os.path.join(probe_a, 'cluster_info.tsv'), sep='\t')
    cluster_info_b = pd.read_csv(os.path.join(probe_b, 'cluster_info.tsv'), sep='\t')
    cluster_info_c = pd.read_csv(os.path.join(probe_c, 'cluster_info.tsv'), sep='\t')                        

    a_ch = np.array(cluster_info_a.ch)
    b_ch = np.array(cluster_info_b.ch)
    c_ch = np.array(cluster_info_c.ch)

    ch = np.concatenate((a_ch,b_ch,c_ch))

    a_depth = np.array(cluster_info_a.depth)
    b_depth = np.array(cluster_info_b.depth)
    c_depth = np.array(cluster_info_c.depth)

    depth_clusterinfo = np.concatenate((a_depth,b_depth,c_depth))

    units_df['ch'] = ch
    units_df['depth_clusterinfo'] = depth_clusterinfo

    # generate unique unit_ids 
    new_unit_ids = []
    for unitid in units_df.index:
        uu = units_df.iloc[[unitid]]
        new_unit_ids.append("{}{}".format(str(list(uu["probe"])[0]), str(list(uu["unit_id"])[0])))
    units_df["unit_id"] = new_unit_ids
    
    try:
        probe_list, stim = load_herbs(mouse)
        probe_IDs = ['A','B','C']
        probes_dist = {}
        stim_coords_ = stim_coords(stim)
        most_used_contact = stim_df['contact_negative'].value_counts().idxmax()
        from metrics import distance
        for ID, probe in zip(probe_IDs, probe_list):
            vox = neuropixel_coords(probe)
            dist = distance(vox, stim_coords_[most_used_contact]) 
            probes_dist[ID] = dist
            #pick middle of the stim probe for this for now
            #todo: use the contacts most used... 

            mask = units_df['probe'] == ID
            units_df.loc[mask, 'distance_from_stim'] = units_df.loc[mask, 'ch'].apply(lambda x: dist[x-1] if x-1 < len(dist) else None)            

    except IndexError:
        units_df['distance_from_stim'] = None

  
    ## assemble NWB
    
    nwbfile = NWBFile(mouse, 
                      directory, 
                      datetime.now(tzlocal()),
                      experimenter = experimenter,
                      lab = 'Denman Lab',
                      keywords = ['neuropixels','mouse','electrical stimulation', 'orthogonal neuropixels'],
                      institution = 'University of Colorado',
                      subject = subject,
                      experiment_description = experiment_description,
                      session_id = os.path.basename(directory))
    stimuli = stim_df
    stimuli['notes'] = ''
    stimuli['start_time'] = stimuli['stim_time']
    stimuli['end_time'] = stimuli['stim_time']+2
    #add epochs

    nwbfile.add_epoch(stimuli.start_time.values[0],
                    stimuli.start_time.values[-1], 'stimulation_epoch')
    nwbfile.add_trial_column('train_duration', 'train duration (s)')
    nwbfile.add_trial_column('train_period', 'train period (s)')
    nwbfile.add_trial_column('train_quantity', 'train quantity')
    nwbfile.add_trial_column('shape', 'monophasic, biphasic or triphasic')
    nwbfile.add_trial_column('run', 'the run number')
    nwbfile.add_trial_column('pulse_duration', 'usecs')
    nwbfile.add_trial_column('pulse_number', 'event quantity')
    nwbfile.add_trial_column('event_period', 'milliseconds')
    nwbfile.add_trial_column('amplitude', 'amplitude in uA')
    nwbfile.add_trial_column('contacts', 'the stimulation contacts and polarities used on the stim electrode')
    nwbfile.add_trial_column('contact_negative', 'the negative (cathodal) contact for a trial')
    nwbfile.add_trial_column('contact_positive', 'the positive (anodal) contact used') 
    nwbfile.add_trial_column('polarity', 'bipolar or monopolar')
    nwbfile.add_trial_column('notes', 'general notes from recording')

    for i in range(len(stimuli)):    
        nwbfile.add_trial(start_time = stimuli.start_time[i],
                 stop_time = stimuli.end_time[i],
                 #parameter = str(stimuli.parameter[i]),
                 amplitude = stimuli.EventAmp1[i],
                 pulse_duration = stimuli.EventDur1[i],
                 shape = stimuli.EventType[i],
                 polarity = str(stimuli.polarity[i]),
                 run = stimuli.Run[i],
                 pulse_number = stimuli.EventQuantity[i],
                 event_period = stimuli.EventPeriod[i]/1e3,
                 train_duration = stimuli.TrainDur[i]/1e6,
                 train_period = stimuli.TrainPeriod[i]/1e6,
                 train_quantity = stimuli.TrainQuantity[i],
                 contacts = stimuli.comment[i],
                 contact_positive = stimuli.contact_positive[i],
                 contact_negative = stimuli.contact_negative[i],
                 notes = stimuli.notes[i])

    probeids = ['A','B', 'C']

    device = nwbfile.create_device(name='DenmanLab_EphysRig2')

    for i, probe in enumerate(probes):
        electrode_name = 'probe'+str(i)
        description = "Neuropixels1.0_"+probeids[i]
        location = "near visual cortex"

        electrode_group = nwbfile.create_electrode_group(electrode_name,
                                                        description=description,
                                                        location=location,
                                                        device=device)
        
        #add channels to each probe
        for ch in range(option234_positions.shape[0]):
            nwbfile.add_electrode(x=option234_positions[ch,0],y=0.,z=option234_positions[0,1],imp=0.0,location='none',filtering='high pass 300Hz',group=electrode_group)
        
    electrode_table_region = nwbfile.create_electrode_table_region([0], 'the second electrode')
    nwbfile.add_unit_column('probe', 'probe ID')
    nwbfile.add_unit_column('unit_id','cluster ID from KS2')
    nwbfile.add_unit_column('group', 'user label of good/mua')
    nwbfile.add_unit_column('depth', 'the depth of this unit from zpos and insertion depth')
    nwbfile.add_unit_column('xpos', 'the x position on probe')
    nwbfile.add_unit_column('zpos', 'the z position on probe')
    nwbfile.add_unit_column('no_spikes', 'total number of spikes across recording')
    nwbfile.add_unit_column('KSlabel', 'Kilosort label')
    nwbfile.add_unit_column('KSamplitude', 'Kilosort amplitude')
    nwbfile.add_unit_column('KScontamination', 'Kilosort ISI contamination')
    nwbfile.add_unit_column('template', 'Kilosort template')
    nwbfile.add_unit_column('ch', 'channel number')
    nwbfile.add_unit_column('depth_clusterinfo', 'depth derived from the clusterinfo.tsv file')
    nwbfile.add_unit_column('distance_from_stim', 'the distance from the middle contact of the stimulating electrode')




    for i,unit_row in units_df[units_df.group != 'noise'].iterrows():
        nwbfile.add_unit(probe=str(unit_row.probe),
                        id = i,
                        unit_id = unit_row.unit_id,
                        spike_times=unit_row.times,
                        electrodes = np.where(unit_row.waveform_weights > 0)[0],
                        depth = unit_row.depth,
                        xpos= unit_row.xpos,
                        zpos= unit_row.zpos,
                        template= unit_row.template,
                        no_spikes = unit_row.no_spikes,
                        group= str(unit_row.group),
                        KSlabel= str(unit_row.KSlabel),
                        KSamplitude= unit_row.KSamplitude,
                        KScontamination= unit_row.KScontamination,
                        ch = unit_row.ch,
                        depth_clusterinfo = unit_row.depth_clusterinfo,
                        distance_from_stim = unit_row.distance_from_stim)        

    with NWBHDF5IO(os.path.join(directory,f'{mouse}.nwb'), 'w') as io:
        io.write(nwbfile)
