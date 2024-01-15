from icecream import ic
import numpy as np

# Create a sample NumPy array
array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Define the file path where you want to save the array
file_path = '/Users/antonia/dev/UNITN/remote_sensing_systems/arss_aerial_feature_matching/groundtruth_gui/pairs.txt'

# Open the file in write mode
#with open(file_path, 'w') as file:
file = open(file_path, 'w')
file.write("hello")
# Iterate over the rows of the array
for row in array:
    # Convert the row to a string and write it to the file
    row_string = ', '.join(map(str, row))  # This creates a comma-separated string of the row's values
    ic(type(row_string))
    file.write(row_string + '\n')  # Write the string to the file, followed by a newline character

# Now each row of the array is written as a separate line in the text file
