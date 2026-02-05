import h5py
import numpy as np
import yaml
import os
import sys
import time

# structure that stores a dictionary of arrays {var_name: data_array} and offsets
class CollJagged:
    def __init__(self, data_dict, offsets):
        self.data = data_dict
        self.offsets = offsets

    def n_events(self):
        return len(self.offsets) - 1

# structure that stores a dictionary of data for a single event {var_name: slice}
class SingleEvent:
    def __init__(self, data_dict):
        self.data = data_dict

    def size(self):
        # Return length of first variable found (assuming all same length)
        key = next(iter(self.data))
        return len(self.data[key])

# round like C++ rounding
def round_cpp(x):
    if x >= 0:
        return int(x + 0.5)
    else:
        return int(x - 0.5)

# function to digitize the variable
def digitize(value, bits, min_val, max_val):
    if value < min_val: value = min_val
    if value > max_val: value = max_val
    scale = (2**bits - 1) / (max_val - min_val)
    return round_cpp((value - min_val) * scale)

# reads a group fron the hdf5 file and returns as a collection dictionary CollJagged
def read_collection(h5_file, group_name, variable_config):
    # check if the group is indeed in the file
    if group_name not in h5_file:
        raise ValueError(f"Group {group_name} not found in file.")
    
    data_dict = {}
    
    # read offsets which are stored in Et
    offsets_path = f"{group_name}/Et/offsets"
    
    # check if this path indeed exists
    if offsets_path in h5_file:
        offsets = h5_file[offsets_path][:]
    else:
        raise ValueError(f"Offsets not found at {offsets_path}")

    # read the data for each variable specified in config file
    for var_name, cfg in variable_config.items():
        key = cfg['hdf5_key']
        path = f"{group_name}/{key}/data"
        if path in h5_file:
            data_dict[var_name] = h5_file[path][:]
        else:
            print(f"Warning: {path} missing in HDF5.")
            # Create empty array of zeroes as fallback?
            data_dict[var_name] = np.zeros(len(offsets)-1) # simplistic fallback

    return CollJagged(data_dict, offsets)

# get the data for a single event and return the SingleEvent structure
def make_event_coll(coll, i_evt):
    # use the offsets to note where the event starts and ends
    start = coll.offsets[i_evt]
    stop = coll.offsets[i_evt+1]
    
    evt_data = {}
    # loop through the variables
    for var_name, full_array in coll.data.items():
        evt_data[var_name] = full_array[start:stop]
        
    return SingleEvent(evt_data)

# write the test vectors for one event with the specified selections
def write_test_vectors(f_out, i_evt, event, var_config, order):
    idx = 0
    
    # Iterate over particles in the event (looping over the arrays)
    n_particles = event.size()
    
    for i in range(n_particles):
        
        # check if 'et' exists and is < 0
        if 'et' in event.data and event.data['et'][i] < 0:
            continue

        bin_strings = []
        packed_word = 0
        current_shift = 0
        
        # digitize the variables with the specified selections in config file
        # store the digitized variables here
        values = {}

        # loop through each variable
        for var_name in order:
            cfg = var_config[var_name]
            raw_val = event.data[var_name][i]
            digi_val = digitize(raw_val, cfg['bits'], cfg['min'], cfg['max'])
            values[var_name] = digi_val
            
            # Format binary string part
            bin_strings.append(f"{digi_val:0{cfg['bits']}b}")

        # pack the bits, and iterate list in reverse order so they get packed in the forward order
        reversed_order = list(reversed(order))
        
        for var_name in reversed_order:
            val = values[var_name]
            bits = var_config[var_name]['bits']
            
            # Add to packed word
            packed_word |= (val << current_shift)
            
            # Increment shift for next variable
            current_shift += bits

        # Join binary strings with |
        full_bin_str = "|".join(bin_strings)

        if idx == 0:
            f_out.write(f"Event : {i_evt}\n")
            
        f_out.write(f"0x{idx:02x} {full_bin_str} 0x{packed_word:08x}\n")
        
        idx += 1

# load the data from the configuration file
def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def HDF5toTestVector(conig_data: dict):
    """
    take a config file as input and convert the specified HDF5 file to test vectors
    """
    # start timer
    t_evt_start = time.perf_counter()

    # get info from config file
    h5_path = config_data["io"]["input_file"]
    out_dir = config_data["io"]["output_base_dir"]

    # make the base directory
    os.makedirs(out_dir, exist_ok = True)

    # get the variable info
    var_config = config_data["variables"]
    pack_order = config_data["packing_order"]

    # open the hdf5 file
    with h5py.File(h5_path, 'r') as h5:

        # for each collection, get the information
        for col_cfg in config_data['collections']:
            group_name = col_cfg['name']
            out_name = col_cfg['out_filedir']
            full_out_path = os.path.join(out_dir, out_name)
            
            # read the collection data
            col_data = read_collection(h5, group_name, var_config)

            print(f"printing {group_name} test vectors")
            
            # print the test vectors to the output path
            with open(full_out_path, 'w') as f_out:
                for i_evt in range(col_data.n_events()):
                    evt = make_event_coll(col_data, i_evt)
                    write_test_vectors(f_out, i_evt, evt, var_config, pack_order)

config_data = load_config("/home/crystalwang/AODtestVectorPipeline/HDF5toTestVector/config.yaml")
HDF5toTestVector(config_data)


        