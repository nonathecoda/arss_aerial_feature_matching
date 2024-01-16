from icecream import ic
import numpy as np

file_path = "/Users/antonia/dev/UNITN/remote_sensing_systems/arss_aerial_feature_matching/groundtruth_gui/pairs.txt"
with open(file_path, 'r') as file:
    # Iterate over each line in the file
    for line in file:
        # Process the line
        string = line.strip()
        arr = string.split(", ")
        ic(arr)
        