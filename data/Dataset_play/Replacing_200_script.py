import numpy as np
import ast

def read_the_file(file_path):
    # Read the data from the file
    # file_path = "Just the data_v2.txt"
    with open(file_path, "r") as file:
        raw_data = file.read()

    # Parse the string into a list of lists
    data = ast.literal_eval(raw_data)

def Remove_200_data(raw_data):
    # Convert the data into a numpy array for easier manipulation
    array = np.array(raw_data, dtype=np.float32)

    # Replace all occurrences of -200 with NaN to prepare for interpolation
    array[array == -200] = np.nan

    # Function to interpolate missing values
    def interpolate_nan(arr):
        # Get indices of valid (non-NaN) and invalid (NaN) positions
        valid_mask = ~np.isnan(arr)
        invalid_mask = np.isnan(arr)
        valid_indices = np.where(valid_mask)[0]
        invalid_indices = np.where(invalid_mask)[0]

        # Perform linear interpolation
        arr[invalid_mask] = np.interp(
            invalid_indices, valid_indices, arr[valid_mask]
        )
        return arr

    # Interpolate along each row
    for i in range(array.shape[0]):
        array[i] = interpolate_nan(array[i])

    # Convert back to a list for exporting or further processing
    interpolated_data = array.tolist()
    return interpolated_data


# # Print or save the output
# print("Interpolated Data:")
# for row in interpolated_data:
#     print(row)

def save_the_file(interpolated_data, name):
    # Optionally save the output to a file
    output_file_path = name
    with open(output_file_path, "w") as output_file:
        output_file.write(str(interpolated_data))
