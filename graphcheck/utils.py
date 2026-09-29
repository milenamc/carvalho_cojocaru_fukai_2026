import os
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from typing import List
import datetime

from scipy.signal import find_peaks

from .io import load_prof_dict_npz


def load_obj(name):
    """
    Loads a pickle object from a specified file
    """
    with open(name, 'rb') as f:
        return pickle.load(f)
    

def read_csv(path):
    """
    Reads a csv file and returns a dataframe
    """
    return pd.read_csv(path, index_col=0)


def add_prof_times(prof_dict, times_ms: list):
    for item in prof_dict.values():
        item['times'] = times_ms[item['sequences']]


def open_dict(path: str, filename: str):
    """
    Opens dictionary created by profile analysis
    """
    prof_dict = load_prof_dict_npz(os.path.join(path, filename))
    return prof_dict


def open_file(path, filename, start_time_ms, end_time_ms, frequency_Hz, 
              filetype = np.int16, shift: bool = False):
    with open(os.path.join(path, filename), 'rb') as fid:
        values = np.fromfile(fid, filetype)
        nlines = len(values)
        timestamps = np.arange(nlines) / frequency_Hz * 1000.0 # ms

    subsetter = np.where((timestamps >= start_time_ms) & (timestamps < end_time_ms))[0]
    if shift:
        timestamps -= start_time_ms
    return np.vstack((timestamps[subsetter], values[subsetter]))


class Config:
    def __init__(self, path_data_root, path_prof_root, session, alpha):
        self.path_data = path_data_root + session

        self.path_profiles_dict = {
            "Early": path_prof_root + (
                f"dataset{session}_start100_dur20_fr3_win100_"
                f"slide100_bin1_alpha{alpha}_matchpen0_minneu3"
            ),
            "Late": path_prof_root + (
                f"dataset{session}_start3600_dur20_fr3_win100_"
                f"slide100_bin1_alpha{alpha}_matchpen0_minneu3"
            )
        }
        
        self.filename_dict = {
            "Early": (
                f"final_prof_dict_dataset{session}_start100_dur20_"
                f"fr3_win100_slide100_bin1_alpha{alpha}_matchpen0_"
                "minneu3_rand12345_damp0.9_maxiter2000_minclu10_"
                "relev0.7_sim1_align1_recall1_minhits5_overlap2.npz"
            ),
            "Late": (
                f"final_prof_dict_dataset{session}_start3600_dur20_"
                f"fr3_win100_slide100_bin1_alpha{alpha}_matchpen0_"
                "minneu3_rand12345_damp0.9_maxiter2000_minclu10_"
                "relev0.7_sim1_align1_recall1_minhits5_overlap2.npz"
            ),
        }


class Log:
    def __init__(self, path_out, script_name: str = None):
        current_datetime = datetime.datetime.now()
        filename_log = f"log_{current_datetime.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        self.file = open(path_out + filename_log, "w")

        self.prepare_logfile(current_datetime, script_name)
    
    def prepare_logfile(self, current_datetime: datetime, script_name: str):
        if script_name:
            self.file.write("graphcheck | Log file | "+ script_name +"\n")
        else:
            self.file.write("graphcheck | Log file\n")
        self.file.write("Date: " + current_datetime.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        self.file.write("--------------------------------------------------------\n\n")

    def lwrite(self, message):
        self.file.write(str(message) + "\n")

    def bar(self):
        self.file.write("\n--------------------------------------------------------\n\n")

    def write_persistent_assemblies(self, persistent_assemblies: List[dict]):
        self.lwrite("Number of persistent assemblies: " + str(len(persistent_assemblies)))
        for pers_assemb in persistent_assemblies:
            self.lwrite(pers_assemb['pers_assemb'])


def import_spikes_etos4(file_path: str, frequency_Hz: float):
    """
    Import spikes sorted using ETOS4 using files *.clu and *.res
    """
    prefix = 'All'
    file_type = ['clu','res']
    index_list = [1, 2]

    df_spikes = pd.DataFrame()
    id_shift = 0

    for idx in index_list:
        clu_file = f"{prefix}.{file_type[0]}.{idx}"
        res_file = f"{prefix}.{file_type[1]}.{idx}"
        
        classid = np.loadtxt(os.path.join(file_path, clu_file), dtype=np.int32)
        spiketime = np.loadtxt(os.path.join(file_path, res_file), dtype=np.float64)
        
        # ignore cluster count (first line of the clu file)
        classid = classid[1:]

        # convert to ms using frequency
        spiketime = spiketime[classid > 0] / frequency_Hz * 1000.0

        # remove cluster 0 ('outliers') from data
        classid = classid[classid > 0]
        
        df_seq = pd.DataFrame({
            'neuronid': classid + id_shift,
            'spiketime': spiketime,
            'electrode': [idx] * len(spiketime)
            }).astype({'neuronid': np.int32, 'spiketime': np.float64, 'electrode': np.uint8})

        # compute ID shift after current electrode
        unique_clu = np.unique(classid).astype(int)
        id_shift = id_shift + unique_clu.shape[0]

        df_spikes = pd.concat([df_spikes, df_seq], ignore_index=True, sort=False)

    assert df_spikes.groupby("neuronid")["electrode"].nunique().max() <= 1, "Duplicate neuron IDs detected across electrodes"

    ids = df_spikes["neuronid"].dropna().sort_values().unique()
    assert (ids == range(ids.min(), ids.max() + 1)).all(), "Neuron ID sequence has gaps"

    # shift first neuron id to 0
    df_spikes["neuronid"] = df_spikes["neuronid"] - df_spikes["neuronid"].min()

    df_spikes = df_spikes.reset_index(drop=True)

    return df_spikes


def clip_spike_train(df_spikes: pd.DataFrame, start_time_ms: float, duration_ms: float):
    end_time_ms = start_time_ms + duration_ms
    return df_spikes.loc[(df_spikes.spiketime >= start_time_ms) & (df_spikes.spiketime < end_time_ms)]


def extract_neurons_fr(df_spikes: pd.DataFrame, max_rate_Hz: float, duration_ms: float = None):
    rates = []
    remove_neu = []
    remove_flag = []

    if not duration_ms:
        duration_ms = df_spikes["spiketime"].max()

    neu_list = df_spikes["neuronid"].unique()
    
    remove_flag = np.zeros_like(neu_list, dtype=bool)

    for i, neu in enumerate(neu_list):
        df_temp = df_spikes.loc[df_spikes.neuronid == neu]

        rate_neuron = df_temp.shape[0] / (duration_ms) * 1000
        rates.append(rate_neuron)

        if (rate_neuron >= max_rate_Hz):
            remove_flag[i] = True
            remove_neu.append(neu)
            
    df_fr = pd.DataFrame({'neuronid': neu_list, 'rate': rates, 'remove': remove_flag})
            
    return df_fr, remove_neu


def remove_neurons(df_spikes: pd.DataFrame, remove_neu: list):
    index_list = list(df_spikes.loc[df_spikes.neuronid.isin(remove_neu)].index)
    df_spikes = df_spikes.drop(index=index_list)

    return df_spikes


def get_end_elect_1(df: pd.DataFrame):
    """
    Returns last neuron of first electrode for layer mapping
    """
    return df.loc[df.electrode == 1, "neuronid"].max()


def df2binarray_csc(df_spikes: pd.DataFrame, start_time_ms: float, duration_ms: float, bin_ms: float):
    """
    Transforms a spike dataframe into a sparse matrix of binned spike counts.

    - df_spikes must be pre-clipped to the half-open interval [start_time_ms, start_time_ms + duration_ms)
    - spike times, start_time_ms, duration_ms and bin_ms may be floating-point
    - bins are left-closed/right-open intervals of width bin_ms starting at start_time_ms
    """

    assert df_spikes.spiketime.min() >= start_time_ms, "First spike time happens before specified stage"
    assert df_spikes.spiketime.max() < start_time_ms + duration_ms, "Last spike time happens after specified stage"

    unique_ids = np.sort(df_spikes.neuronid.unique())
    neuron_list = [int(x) for x in unique_ids]

    neuronids = df_spikes.neuronid
    spikes_ms = df_spikes.spiketime # spike time already in ms
    
    nrow = len(unique_ids)
    ncol = int(np.ceil(duration_ms / bin_ms))
    bins = start_time_ms + np.arange(ncol) * bin_ms

    binarray_lil = sparse.lil_matrix((nrow, ncol), dtype=np.uint8)

    for i, n_id in enumerate(unique_ids):
        spk_train = spikes_ms[neuronids == n_id]
        digitized_spk_train = np.digitize(spk_train, bins) - 1
        binned_spk_train = np.bincount(digitized_spk_train)
        binarray_lil[i, digitized_spk_train] = binned_spk_train[digitized_spk_train]
    
    return binarray_lil.tocsc(), neuron_list


def prepare_data(path_data: str, start_time_sec: float, duration_min: float, max_rate_Hz: float, window_ms: float, slide_ms: float, bin_ms: float, full: bool = True):
    """
    Loads ETOS4 spiking data using auxiliary functions from graph-check. 
    
    If "full," calculates the average firing rate of each neuron for the entire 
    session (default); if the firing rate calculation should be clipped to the 
    duration, set this variable as false
    """
    frequency_Hz = 20000
    start_time_ms = start_time_sec * 1000
    duration_ms = duration_min * 60 * 1000

    if full:
        df_spikes_temp = import_spikes_etos4(path_data, frequency_Hz)
        _, remove_neu = extract_neurons_fr(df_spikes_temp, max_rate_Hz)
        df_spikes_lowFR = remove_neurons(df_spikes_temp, remove_neu)
        df_spikes = clip_spike_train(df_spikes_lowFR, start_time_ms, duration_ms)
    else:
        df_spikes_temp = import_spikes_etos4(path_data, frequency_Hz)
        df_spikes_clip = clip_spike_train(df_spikes_temp, start_time_ms, duration_ms)
        _, remove_neu = extract_neurons_fr(df_spikes_clip, max_rate_Hz, duration_ms)
        df_spikes = remove_neurons(df_spikes_clip, remove_neu)
    
    end_elect_1 = get_end_elect_1(df_spikes)

    binmat, neuron_list = df2binarray_csc(df_spikes, start_time_ms, duration_ms, bin_ms)

    duration_bins = binmat.shape[1]
    window_len_bins = int(np.ceil(window_ms / bin_ms))
    slide_len_bins = int(np.ceil(slide_ms / bin_ms))

    start_bins = np.arange(0, duration_bins - window_len_bins + 1, slide_len_bins, dtype=int)
    windows = [binmat[:, t:(t + window_len_bins)] for t in start_bins]

    times_ms = start_time_ms + start_bins * bin_ms

    return binmat, neuron_list, start_bins, windows, times_ms, window_len_bins, end_elect_1


def calculate_correlation_matrix(df_spikes: pd.DataFrame, start_time_ms: int, 
                                 duration_ms: int, corr_bin_width: int = 1):
    """
    Calculate correlation matrix from spike data
    """
    binmat, neuron_list = df2binarray_csc(df_spikes, start_time_ms=start_time_ms, 
                                          duration_ms=duration_ms, bin_ms=corr_bin_width)
    corrmat = np.corrcoef(binmat.toarray())
    return corrmat, neuron_list


def calculate_success_failure(event: np.ndarray):
    times = event[0]

    # assumes that the event array has been normalized
    success = (event[1] > 0.85).astype(float)
    failure = -(event[1] < -0.85).astype(float)

    switch_success = np.diff(success, prepend=0) >= 1
    index_success = find_peaks(switch_success)[0] # holds indices
    times_success = times[index_success]

    switch_failure = np.diff(failure, prepend=0) <= -1
    index_failure = find_peaks(switch_failure)[0] # holds indices
    times_failure = times[index_failure]

    return times_success, times_failure


def calculate_intertimes(times_success, times_failure):
    times_allevents = np.sort(np.concatenate((times_success, times_failure)))

    # calculates inter-reward interval
    iri = np.diff(times_success)
    avg_iri = np.average(iri)

    # calculates inter-trial interval
    iti = np.diff(times_allevents)
    avg_iti = np.average(iti)

    return avg_iri, avg_iti


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_all_nearest_sequences(profile_times: List, reward_time: float, interval_sec: float = 1.5):
    # New method: find all nearest sequences within the desired time interval
    
    nearest_sequences = [] # [0]: sequence time, [1]: time distance to reward

    for sequence_time in profile_times:
        if (abs(sequence_time - reward_time) < interval_sec * 1000): # convert to ms
            nearest_sequences.append([sequence_time, sequence_time - reward_time])
    return nearest_sequences


def norm_max_abs(data):
    return data / np.max(np.abs(data))


def norm_min_max(data):
    dmin, dmax = min(data), max(data)
    return (data - dmin) / (dmax - dmin)