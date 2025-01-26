import numpy as np

# Input data: List of NumPy arrays
data = [
    np.array([[32, 31, 43, 28, 33],
              [45, 16, 13, 22, 39],
              [41, 32, 47, 35, 38],
              [27, 47, 38, 46, 35],
              [21, 34, 32, 28, 33]]),
    np.array([[23, 22, 12, 16, 44],
              [44, 10, 14, 30, 35],
              [24, 19, 30, 23, 27],
              [11, 27, 48, 12, 36],
              [32, 37, 37, 21, 25]]),
    np.array([[11, 12, 40, 10, 40],
              [22, 36, 22, 36, 44],
              [29, 44, 34, 10, 44],
              [43, 14, 43, 11, 30],
              [16, 12, 36, 24, 31]]),
    np.array([[44, 19, 41, 22, 36],
              [10, 22, 17, 36, 41],
              [30, 19, 47, 17, 34],
              [39, 10, 29, 42, 13],
              [26, 16, 37, 24, 21]]),
    np.array([[38, 14, 48, 22, 36],
              [39, 30, 17, 46, 33],
              [20, 19, 17, 42, 49],
              [34, 34, 28, 10, 16],
              [30, 19, 43, 24, 14]])
]

# Convert to a dictionary with values from the arrays
def convert_to_dict_with_values(arrays):
    """
    Converts a list of NumPy arrays into a dictionary where:
    - Keys are (i, j) tuples representing positions.
    - Values are the corresponding values from the arrays.
    
    Args:
        arrays (list of np.array): List of 2D NumPy arrays.
        
    Returns:
        dict: Dictionary with keys as positions and values from the arrays.
    """
    result = {}
    for arr in arrays:
        rows, cols = arr.shape
        for i in range(rows):
            for j in range(cols):
                result[(i + 1, j + 1)] = arr[i, j]
    return result

# Convert the data to the desired dictionary
result_dict = convert_to_dict_with_values(data)

# Print result
print(result_dict)