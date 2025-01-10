import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import expn
import scipy.special as sp
from pathlib import Path

def thermal_data_prep(input_data):

    # Check if input_data is a file path or a DataFrame
    if isinstance(input_data, str):  # If it's a string, treat it as a file path
        data = pd.read_csv(input_data, delim_whitespace=True, header=None)
    elif isinstance(input_data, pd.DataFrame):  # If it's a DataFrame, use it directly
        data = input_data
    else:
        raise ValueError("Input must be either a file path or a pandas DataFrame")

    # Data preprocessing and cleaning
    data = data.iloc[4:]  # Remove the first four rows
    data = data[0].str.split(",", expand=True)  # Split the first column by comma
    data = data.apply(pd.to_numeric, errors="coerce")  # Convert the data type to numeric
    data = data.iloc[:, -4:]
    data.columns = ["Counter", "T1", "T3", "Volt"]

    while data["Counter"].iloc[0] != 0:  # Remove the few rows until the counter is 0
        data = data.iloc[1:]

    valid_rows = []  # Store the indices of valid rows
    expected_counter = 0  # Initialize the expected counter value
    for idx, counter in enumerate(data["Counter"]):  # Remove the rows with unexpected counter values
        if counter == expected_counter:
            valid_rows.append(idx)
            expected_counter = (expected_counter + 1) % 300

    data = data.iloc[valid_rows].reset_index(drop=True)  # Reset the index

    T1_outliers = (data["T1"] > 10 * data["T1"].shift(1).rolling(10).median()) | \
        (data["T1"] > 10 * data["T1"].shift(-1).rolling(10).median()) | \
            (data["T1"].isna())   # Identify outliers for T1
    T3_outliers = (data["T3"] > 10 * data["T3"].shift(1).rolling(10).median()) | \
        (data["T3"] > 10 * data["T3"].shift(-1).rolling(10).median()) | \
            (data["T3"].isna())   # Identify outliers for T3

    data["T1"] = data["T1"].mask(T1_outliers, data["T1"].shift(-1))  # Replace outliers for T1
    data["T3"] = data["T3"].mask(T3_outliers, data["T3"].shift(-1))  # Replace outliers for T3

    return data

def Heat(input_data, parameters=None):

    # Define default parameters
    default_heat_parameters = {
        "Radius of the probe (m)": 6.35e-4,
        "Volumetric heat capacity of the probe (MJ m-3 K-1)": 2.84e6,
        "Probe spacing for T1 (m)": 0.008,
        "Probe spacing for T3 (m)": 0.008,
        "Resistance of the heating element (Ohm)": 887.6,
        "Duration of one measurement (s)": 300
    }
    
    # Use default parameters if not provided
    if parameters is None:
        input_heat_parameters = default_heat_parameters
    else:
        # If any parameter is updated, use the updated value
        input_heat_parameters = {key: parameters.get(key, default_heat_parameters[key]) for key in default_heat_parameters.keys()}

    heat_data = thermal_data_prep(input_data)
        
    results = []

    # Define constants and user_parameters
    a = input_heat_parameters["Radius of the probe (m)"]
    C0 = input_heat_parameters["Volumetric heat capacity of the probe (MJ m-3 K-1)"]
    r1 = input_heat_parameters["Probe spacing for T1 (m)"]
    r3 = input_heat_parameters["Probe spacing for T3 (m)"]
    R = input_heat_parameters["Resistance of the heating element (Ohm)"]
    T = input_heat_parameters["Duration of one measurement (s)"]

    Times = len(heat_data) // T

    # define t0 as the heat pulse width, i.e., number of cells in column volt than values are greater than 50
    t0 = len(heat_data[heat_data["Volt"] > 50]) // Times
    # define t1 as the time when the heat pulse starts, i.e., counter value when the first value in column volt is greater than 50\\
    t1 = int(heat_data[heat_data["Volt"] > 50].iloc[0]["Counter"])

    # define t2 as the time when the heat pulse ends, i.e., counter value when the last value in column volt is greater than 50
    t2 = int(t0 + t1)

    # Implement the ICPC function
    def fasticpcinv(time, L, a0, qprime, beta0, lambda_, kappa):
        MM = len(time)
        omega = np.array([-3.96825396825396825e-4, 2.13373015873016, -551.01666666666667,
                        33500.161111111111, -812665.11111111111111, 10076183.766666667,
                        -73241382.977777778, 339059632.073016, -1052539536.278571,
                        2259013328.5833333, -3399701984.433333333, 3582450461.7,
                        -2591494081.3666667, 1227049828.7666667, -342734555.42857143,
                        42841819.428571428])

        omega_array = np.tile(omega, (MM, 1))
        i_array = np.tile(np.arange(1, 17), (MM, 1))
        time_array = np.tile(time, (16, 1)).T  # Corrected to use time instead of np.arange(1, 17)

        with np.errstate(invalid='ignore'):  # Suppress invalid value warnings
            mu = np.sqrt(np.log(2) * i_array / (kappa * time_array))
        mu = np.nan_to_num(mu)  # Replace NaNs with zero

        arg = mu * a0
        bsslterm = arg * (sp.kv(1, arg) + arg * beta0 / 2 * sp.kv(0, arg))
        seriesterm = omega_array * sp.kv(0, mu * L) / (i_array * bsslterm * bsslterm)

        result = qprime / (2 * np.pi * lambda_) * np.sum(seriesterm, axis=1)
        return result

    # Define the ICPC model functions
    def Vsub21(params, t):
        a1, a2 = params
        return fasticpcinv(t, r1, a, q, C0 / a1 / 1.0e6, a1 * a2 / 10, a1 * 1.0e-7) - \
            fasticpcinv(t - t0, r1, a, q, C0 / a1 / 1.0e6, a1 * a2 / 10, a2 * 1.0e-7)

    def Vsub23(params, t):
        a3, a4 = params
        return fasticpcinv(t, r3, a, q, C0 / a3 / 1.0e6, a3 * a4 / 10, a3 * 1.0e-7) - \
            fasticpcinv(t - t0, r3, a, q, C0 / a3 / 1.0e6, a3 * a4 / 10, a4 * 1.0e-7)

    # Define the PILS model functions
    def PILS_T1(params, t):
        b1, b2 = params
        return -q / (4 * np.pi * b1 * b2 / 10) * (
            expn(1, (r1 ** 2) / (4 * (b2 * 1.0e-7) * (t - t0)))
            - expn(1, (r1 ** 2) / (4 * (b2 * 1.0e-7) * t))
        )

    def PILS_T3(params, t):
        b3, b4 = params
        return -q / (4 * np.pi * b3 * b4 / 10) * (
            expn(1, (r3 ** 2) / (4 * (b4 * 1.0e-7) * (t - t0)))
            - expn(1, (r3 ** 2) / (4 * (b4 * 1.0e-7) * t))
        )

    for i in range(Times):
        Counter = heat_data[["Counter"]][T*i:T*(i+1)].reset_index(drop=True)
        T1 = heat_data["T1"][T*i:T*(i+1)].reset_index(drop=True)
        T3 = heat_data["T3"][T*i:T*(i+1)].reset_index(drop=True)
        Volt = heat_data["Volt"][T*i:T*(i+1)].reset_index(drop=True)

        # Compute baseline temperatures and temperature rises
        BTemp1 = np.mean(T1[:t2])
        BTemp3 = np.mean(T3[:t2])

        deltaT1 = T1[t1:0.8*T].reset_index(drop=True) - BTemp1
        deltaT3 = T3[t1:0.8*T].reset_index(drop=True) - BTemp3
        q = (np.mean(Volt[t1:t2]) / 1000) ** 2 * R

        # Identify the data points for PILS fitting
        M1 = np.where((0.7 * np.max(deltaT1) < deltaT1) & (deltaT1 < 0.9 * np.max(deltaT1)))[0]
        M3 = np.where((0.7 * np.max(deltaT3) < deltaT3) & (deltaT3 < 0.9 * np.max(deltaT3)))[0]
        N1 = np.where(deltaT1 == np.max(deltaT1))[0]
        N3 = np.where(deltaT3 == np.max(deltaT3))[0]

        if N1.size > 0:
            xp1 = M1[M1 > np.max(N1)]
        else:
            xp1 = np.array([])

        if N3.size > 0:
            xp3 = M3[M3 > np.max(N3)]
        else:
            xp3 = np.array([])

        yp1 = deltaT1.iloc[xp1]
        yp3 = deltaT3.iloc[xp3]

        # Perform nonlinear fitting for ICPC models
        if xp1.size > 1:  # Ensure there are enough data points
            C1, _ = curve_fit(lambda t, a1, a2: Vsub21([a1, a2], t), xp1, yp1, p0=[2.0, 4.0], maxfev=2000)
        else:
            C1 = np.array([np.nan, np.nan])

        if xp3.size > 1:  # Ensure there are enough data points
            C3, _ = curve_fit(lambda t, a3, a4: Vsub23([a3, a4], t), xp3, yp3, p0=[2.0, 4.0], maxfev=2000)
        else:
            C3 = np.array([np.nan, np.nan])

        # Perform nonlinear fitting for PILS models
        if xp1.size > 1:  # Ensure there are enough data points
            P1, _ = curve_fit(lambda t, b1, b2: PILS_T1([b1, b2], t), xp1, yp1, p0=[2.0, 4.0], maxfev=2000)
        else:
            P1 = np.array([np.nan, np.nan])

        if xp3.size > 1:  # Ensure there are enough data points
            P3, _ = curve_fit(lambda t, b3, b4: PILS_T3([b3, b4], t), xp3, yp3, p0=[2.0, 4.0], maxfev=2000)
        else:
            P3 = np.array([np.nan, np.nan])

        # Store results for each group
        results.append({
            "Group": i + 1,
            "Heat Capacity needle 1 (ICPC)": C1[0],
            "Heat Capacity needle 3 (ICPC)": C3[0],
            "Thermal diffusity needle 1(ICPC)": C1[1],
            "Thermal diffusity needle 3(ICPC)": C3[1],
            "Thermal conductivity (ICPC)": (C1[0] * C1[1]) / 10,
            "Heat Capacity needle 1 (PILS)": P1[0],
            "Heat Capacity needle 3 (PILS)": P3[0],
            "Thermal diffusity needle 1(PILS)": P1[1],
            "Thermal diffusity needle 3(PILS)": P3[1],
            "Thermal conductivity (PILS)": (P1[0] * P1[1]) / 10,
            "q (Heat input)": q
        })

    return pd.DataFrame(results)
