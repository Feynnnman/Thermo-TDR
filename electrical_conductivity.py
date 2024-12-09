#%%
import pandas as pd

def EC(input_data):
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
    data = data.iloc[:, 12:]

    data = data.T  # Transpose the data

    # Create a list to store results
    results = []

    for i in range(0, data.shape[1]):

        # Define X1 as the mean of the first two cells in the column
        X1 = data.iloc[:2, i].mean()

        column = data.iloc[:, i]
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
        Rt = 75 * (1 + pho_inf) / (1 - pho_inf)
        EC = 85.33 / (Rt - 6.9)

        # Append the results
        results.append({"V0": v0, "Vf": vf, "Pho_inf": pho_inf, "Rt": Rt, "EC": EC})

    return pd.DataFrame(results)

