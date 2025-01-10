#%%
import pandas as pd

def electrical_data_prep(input_data):

    if isinstance(input_data, str):  # If it's a string, treat it as a file path
        data = pd.read_csv(input_data, delim_whitespace=True, header=None)
    elif isinstance(input_data, pd.DataFrame):  # If it's a DataFrame, use it directly
        data = input_data
    else:
        raise ValueError("Input must be either a file path or a pandas DataFrame")

    data = data.iloc[4:]  # Remove the first four rows
    data = data[0].str.split(",", expand=True)  # Split the first column by comma
    data = data.apply(pd.to_numeric, errors="coerce")  # Convert the data type to numeric
    data = data.iloc[:, 12:]

    data = data.T  # Transpose the data  

    return data

def Sigma(input_data, parameters=None):

    # Define default parameters
    default_sigma_parameters = {
        "Characteristic impedance of the cable tester system (ohm)": 75,
        "Resistance of the cable tester system (ohm)": 6.9,
        "Cell constant of the probe (m-1)": 85.33,
        "Temperature coefficient of the sample (°C-1)": 0.0191,
        "Temperature (°C-1)": 25
    }

    # Use default parameters if not provided
    if parameters is None:
        input_sigma_parameters = default_sigma_parameters
    else:
        # If any parameter is updated, use the updated value
        input_sigma_parameters = {key: parameters.get(key, default_sigma_parameters[key]) for key in default_sigma_parameters.keys()}
    
    sigma_data = electrical_data_prep(input_data)

    # Create a list to store results
    results = []

    # Define the parameters
    Zc = input_sigma_parameters["Characteristic impedance of the cable tester system (ohm)"]
    Rcable = input_sigma_parameters["Resistance of the cable tester system (ohm)"]
    Kp = input_sigma_parameters["Cell constant of the probe (m-1)"]
    alpha = input_sigma_parameters["Temperature coefficient of the sample (°C-1)"]
    T = input_sigma_parameters["Temperature (°C-1)"]

    for i in range(0, sigma_data.shape[1]):

        # Define X1 as the mean of the first two cells in the column
        X1 = sigma_data.iloc[:2, i].mean()

        column = sigma_data.iloc[:, i]
        median = []
        std = []
        X2 = None  # Initialize v0
        for j in range(0, len(column) - 100):
            segment = column.iloc[j:j + 10]
            median.append(segment.median())
            std.append(segment.std())

            # Find the minimum standard deviation, and the corresponding median is v0
            if std[-1] == min(std):
                X2 = median[-1]

        # Define X3 as the mean of the last 10 cells in the column
        X3 = column.iloc[-10:].mean()
        
        v0 = X2 - X1
        vf = X3 - X1
        pho_inf = (vf - v0) / v0
        Rt = Zc * (1 + pho_inf) / (1 - pho_inf)
        EC25 = Kp / (Rt - Rcable)
        EC = EC25 * (1 + alpha * (T - 25))

        # Append the results
        results.append({"V0": v0, "Vf": vf, "Pho_inf": pho_inf, "Rt": Rt, "EC": EC})

    return pd.DataFrame(results)

