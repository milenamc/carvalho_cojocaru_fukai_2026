import numpy as np
import pandas as pd
from typing import List


from .utils import prepare_data, add_prof_times, import_spikes_etos4, clip_spike_train, extract_neurons_fr, remove_neurons, open_file, norm_min_max, norm_max_abs, Log

# from .graph import Graph, ProfileGraph, ShuffledProfileGraph, CorrelationGraph, find_persistent_assemblies_graph
# from .data_utils import calculate_correlation_matrix, extract_firing_rate_shuffled_profiles
# from .plot import plot_graph, plot_components, plot_degrees, plot_neuron_properties, plot_edge_properties
# from .dataframes import create_df_neuron, create_df_edge, create_df_profile, create_df_reward, create_df_reward_combined, create_df_profile_shuffle


# Data classes
class LoadData:
    def __init__(self, session: str, min_neu: int, stage: str, alpha: str, path_data: str):
        self.session = session # Session identifier
        self.min_neu = min_neu # Minimum number of neurons in profile
        self.stage = stage # Stage identifier
        self.alpha = alpha # Exponential penalty for edit similarity calculation

        self.window_ms = 100
        self.slide_ms = 100
        self.bin_ms = 100

        self.frequency_Hz = 20000 # Acquisition frequency
        self.max_rate_Hz = 3 # Maximum firing rate allowed
        self.duration_min = 20 # Duration of the analyzed period

        if stage == "Early":
            self.start_time_sec = 100
        else:
            self.start_time_sec = 3600
        
        self.start_time_ms = self.start_time_sec * 1000
        self.duration_ms = self.duration_min * 60 * 1000
        self.end_time_ms = self.start_time_ms + self.duration_ms

        self.lever_data = None
        self.event_data = None

        self.times_ms = None
        self.neuron_list = None
        self.df_spikes = None
        self.df_fr = None
        self.profile_dict = None
        self.end_elect1 = None

        self.prepare_data(path_data)


    def prepare_data(self, path_data):
        _, self.neuron_list, _, _, self.times_ms, _, self.end_elect1 = prepare_data(
            path_data, 
            self.start_time_sec, 
            self.duration_min, 
            self.max_rate_Hz, 
            self.window_ms, 
            self.slide_ms, 
            self.bin_ms, 
            full=False # Stage-specific FR cut
        )


    def load_profiles(self, profile_dict: dict):
        """
        Load pre-imported profile dictionary and ID dataframe
        """
        add_prof_times(profile_dict, self.times_ms)
        self.profile_dict = profile_dict


    def load_spikes(self, df_spikes: pd.DataFrame, df_fr: pd.DataFrame):
        """
        Load pre-imported spikes dataframe
        """
        self.df_spikes = df_spikes
        self.df_fr = df_fr


    def load_lever_data(self, lever: np.ndarray):
        self.lever_data = lever
    

    def load_event_data(self, event: np.ndarray):
        self.event_data = event


    def import_spikes(self, path_data: str):
        """
        Import spikes processed by ETOS4. Using stage-specific FR cut
        """
        df_spikes_temp = import_spikes_etos4(path_data, self.frequency_Hz)
        df_spikes_clip = clip_spike_train(df_spikes_temp, self.start_time_ms, self.duration_ms)
        df_fr, remove_neu = extract_neurons_fr(df_spikes_clip, self.max_rate_Hz, self.duration_ms)
        df_spikes = remove_neurons(df_spikes_clip, remove_neu)
        self.load_spikes(df_spikes, df_fr)
    

    def import_lever_data(self, path_lever: str, norm: bool = True):
        """
        Import lever data
        """
        lever = open_file(
            path_lever, "Lever.dat", self.start_time_ms, self.end_time_ms, 
            self.frequency_Hz, np.int16, shift = False)
        if norm:
            lever[1] = norm_min_max(lever[1])
        self.load_lever_data(lever)


    def import_event_data(self, path_event: str, norm: bool = True):
        """
        Import event data
        """
        event = open_file(
            path_event, "Event.dat", self.start_time_ms, self.end_time_ms, 
            self.frequency_Hz, np.int16, shift = False)
        if norm:
            event[1] = norm_max_abs(event[1])
        self.load_event_data(event)


    def get_profile_dict(self):
        return self.profile_dict.copy()
    

    def get_df_spikes(self):
        return self.df_spikes.copy()


    def get_df_fr(self):
        return self.df_fr.copy()
    

    def get_lever_data(self):
        return self.lever_data.copy()
    

    def get_event_data(self):
        return self.event_data.copy()
    

    def get_end_elect1(self):
        return self.end_elect1
    

    def get_num_neurons_below_fr_max(self):
        if isinstance(self.df_fr, pd.DataFrame):
            num_neurons_below_fr_max = len(self.df_fr[self.df_fr['rate'] < self.max_rate_Hz])
        else:
            num_neurons_below_fr_max = None
        return num_neurons_below_fr_max
    

    def write_report_terminal(self):
        """
        Write report to terminal
        """
        print(f"Session {self.session} ({self.stage}) | Min. Neurons: {self.min_neu}")
        print("Loaded profile dictionary" if isinstance(self.profile_dict, dict) else "No profile dictionary loaded")
        print("Loaded spike dataframe" if isinstance(self.df_spikes, pd.DataFrame) else "No spike dataframe loaded")
        print("Loaded ID dataframe" if isinstance(self.df_id, pd.DataFrame) else "No ID dataframe loaded")
        print("End neuron of electrode 1: " + str(self.end_elect1))


    def write_report_log(self, log: Log):
        """
        Write report to log file
        """
        log.lwrite(f"Session {self.session} ({self.stage}) | Min. Neurons: {self.min_neu}")
        log.lwrite("Loaded profile dictionary" if isinstance(self.profile_dict, dict) else "No profile dictionary loaded")
        log.lwrite("Loaded spike dataframe" if isinstance(self.df_spikes, pd.DataFrame) else "No spike dataframe loaded")
        log.lwrite("Loaded ID dataframe" if isinstance(self.df_id, pd.DataFrame) else "No ID dataframe loaded")
        log.lwrite("End neuron of electrode 1: " + str(self.end_elect1))