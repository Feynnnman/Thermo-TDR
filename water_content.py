#%%
import numpy as np
import pandas as pd
import io

def theta_data_prep(input_data):
    # Check if input_data is a file path or a DataFrame
    if isinstance(input_data, str):  # If it's a string, treat it as a file path
        data = pd.read_csv(input_data, delim_whitespace=True, header=None)
    elif isinstance(input_data, pd.DataFrame):  # If it's a DataFrame, use it directly
        data = input_data
    else:
        raise ValueError("Input must be either a file path or a pandas DataFrame")

    # Data preprocessing
    data = data.iloc[4:]  # Remove the first four rows
    data = data[0].str.split(",", expand=True)  # Split the first column by comma
    data = data.apply(pd.to_numeric, errors="coerce")  # Convert the data type to numeric
    data = data.iloc[:, 12:]  # Remove the first 12 columns

    data = data.T  # Transpose the data

    return data

def Theta(input_data, parameters=None):

    # Define default parameters
    default_theta_parameters = {
        "Probe length (m)": 0.045
    }

    theta_data = theta_data_prep(input_data)

    # Create a list to store results
    results = []

    # Define the parameters
    L = default_theta_parameters["Probe length (m)"]

    for i in range(0, theta_data.shape[1]):
        x = np.arange(0, 251) / 251
        y = theta_data.iloc[:, i]

        # Find the first 70% of the data
        first_70_percent_index = int(0.7 * len(y))
        y_first_70_percent = y.iloc[:first_70_percent_index]
        
        # Find minimum
        local_min = y_first_70_percent.min()
        local_min_idx = y_first_70_percent.idxmin()

        # Find maximum before the minimum
        local_max = y.iloc[:local_min_idx].max()
        local_max_idx = y.iloc[:local_min_idx].idxmax()

        # Calculate the derivatives
        derivatives = np.diff(y) / np.diff(x)
        derivatives_subset = derivatives[:first_70_percent_index]

        # Average the derivatives
        averaged_derivatives = np.convolve(derivatives_subset, np.ones(5)/5, mode='valid')

        # Find the maximum and minimum derivatives
        max_derivative = np.max(averaged_derivatives)
        min_derivative = np.min(averaged_derivatives)
        max_derivative_idx = int(np.argmax(averaged_derivatives))
        min_derivative_idx = int(np.argmin(averaged_derivatives))

        print(f"Local minimum in the first 70% of column {i}: {local_min}")
        print(f"Local maximum in the first 70% of column {i}: {local_max}")
        print(f"Max derivative in the first 70% of column {i}: {max_derivative}")
        print(f"Min derivative in the first 70% of column {i}: {min_derivative}")

        # Define the tangent lines
        tangent_line_local_max = float(y.iloc[local_max_idx])
        tangent_line_local_min = float(y.iloc[local_min_idx])
        tangent_line_max_derivative = y.iloc[max_derivative_idx] + max_derivative * (x - x[max_derivative_idx])
        tangent_line_min_derivative = y.iloc[min_derivative_idx] + min_derivative * (x - x[min_derivative_idx])

        # Define the inflection points
        first_inflection_point = (local_max - y.iloc[min_derivative_idx]) / min_derivative + x[min_derivative_idx]
        second_inflection_point = (local_min - y.iloc[max_derivative_idx]) / max_derivative + x[max_derivative_idx]

        # Calculate the dielectric constant
        dielectric_constant = ((second_inflection_point - first_inflection_point) / L) ** 2

        # Calculate the water content 
        water_content = -0.053 + 0.0292 * dielectric_constant - 0.00055 * dielectric_constant ** 2 + 0.0000043 * dielectric_constant ** 3

        # Store results
        results.append({
            "column": i,
            "local_max": local_max,
            "local_min": local_min,
            "max_derivative": max_derivative,
            "min_derivative": min_derivative,
            "first_inflection_point": first_inflection_point,
            "second_inflection_point": second_inflection_point,
            "dielectric_constant": dielectric_constant,
            "water_content": water_content
        })
    return pd.DataFrame(results)

# %%
