import streamlit as st
import pandas as pd
from pathlib import Path

# Import functions 
from thermal_properties import Heat
from electrical_conductivity import Sigma
from water_content import Theta

# Streamlit interface
st.title("Thermo-TDR")
st.sidebar.header("Upload your data")
uploaded_file = st.sidebar.file_uploader("Choose a file")

# Initialize results variable
results = None

def read_file(uploaded_file):
    # Read uploaded file and return pandas DataFrame
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    readers = {
        '.xlsx': lambda f: pd.read_excel(f, header=None),
        '.csv': lambda f: pd.read_csv(f, delim_whitespace=True, header=None),
        '.txt': lambda f: pd.read_csv(f, delim_whitespace=True, header=None)
    }
    
    try:
        reader = readers.get(file_extension)
        if not reader:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
        try:
            return reader(uploaded_file)
        except UnicodeDecodeError:
            # Fallback to latin1 encoding if utf-8 fails
            return pd.read_csv(uploaded_file, delim_whitespace=True, header=None, encoding='latin1')
            
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

# Only run the analysis if a file is uploaded
if uploaded_file:
    data = read_file(uploaded_file)
    if data is not None:
        st.write("Uploaded Data:")
        st.write(data.head())
        
    st.sidebar.header("Choose Function Type")
    analysis_type = st.sidebar.selectbox("Select an analysis", ["Thermal Properties", "Electrical Conductivity", "Water Content"])

    # Call the respective function
    if analysis_type == "Thermal Properties":
        st.header("Thermal Properties Computation")
        
        # Define default heat parameters
        default_heat_parameters = {
            "Radius of the probe (m)": 6.35e-4,
            "Volumetric heat capacity of the probe (MJ m-3 K-1)": 2.84e6,
            "Probe spacing for Temperature Needle 1 (m)": 0.008,
            "Probe spacing for Temperature Needle 3 (m)": 0.008,
            "Starting time of heating (s)": 7.0,  # Changed to float
            "Heat pulse width (s)": 10.0,  # Changed to float
            "Resistance of the heating element (Ohm)": 887.6
        }
        
        # Create input fields for each parameter
        input_heat_parameters = {}
        st.subheader("Calibration Parameters")
        for param, default_value in default_heat_parameters.items():
            # Determine step size based on the magnitude of the default value
            if abs(default_value) < 0.01 or abs(default_value) > 1000:  # For very small values, use scientific notation
                step = float(f"{default_value:.1e}"[0]) * 10 ** (int(f"{default_value:.1e}".split('e')[1]))
                input_heat_parameters[param] = st.number_input(
                    f"{param}", 
                    value=float(default_value),  # Ensure float type
                    format="%.1e",
                    step=float(step)  # Ensure float type
                )
            else:  # For normal values, use standard decimal format
                input_heat_parameters[param] = st.number_input(
                    f"{param}",
                    value=float(default_value),  # Ensure float type
                    format="%.1f",
                    step=0.1  # Changed to 0.1 for better precision
                )

        # Add a "Run" button
        st.subheader("Run Analysis")
        if st.button("Run"):
            # Ensure all parameters are provided
            if all(value is not None for value in input_heat_parameters.values()):
                st.write("Processing thermal properties...")
                results = Heat(data, parameters=input_heat_parameters)
    
    elif analysis_type == "Electrical Conductivity":
        st.header("Electrical Conductivity Computation")

        default_sigma_parameters = {
            "Characteristic impedance of the cable tester system (ohm)": 75,
            "Resistance of the cable tester system (ohm)": 6.9,
            "Cell constant of the probe (m-1)": 85.33,
            "Temperature coefficient of the sample (°C-1)": 0.0191,
            "Temperature (°C-1)": 25
        }
        
        # Create input fields for each parameter
        input_sigma_parameters = {}
        st.subheader("Calibration Parameters")
        for param, default_value in default_sigma_parameters.items():
            if param == "Temperature coefficient of the sample (°C-1)":
                input_sigma_parameters[param] = st.number_input(
                    f"{param}",
                    value=float(default_value),
                    format="%.4f",
                    step=0.0001 
                )
            else:
                input_sigma_parameters[param] = st.number_input(
                    f"{param}",
                    value=float(default_value),
                    format="%.2f",
                    step=0.01
                )

        # Add a "Run" button
        st.subheader("Run Analysis")
        if st.button("Run"):
            # Ensure all parameters are provided
            if all(value is not None for value in input_sigma_parameters.values()):
                st.write("Processing electrical conductivity...")
                results = Sigma(data, parameters=input_sigma_parameters)

    # Call the theta function
    elif analysis_type == "Water Content":
        st.header("Water Content Computation")

        # Define default theta parameters
        default_theta_parameters = {
            "Probe length (m)": 0.045
        }

        # Create input fields for each parameter
        input_theta_parameters = {}
        st.subheader("Calibration Parameters")
        for param, default_value in default_theta_parameters.items():
            input_theta_parameters[param] = st.number_input(
                f"{param}",
                value=float(default_value),  # Ensure float type
                format="%.3f",
                step=0.005  # Changed to 0.1 for better precision
            )

        # Add a "Run" button
        st.subheader("Run Analysis")
        if st.button("Run"):
            # Ensure all parameters are provided
            if all(value is not None for value in input_theta_parameters.values()):
                st.write("Processing water content...")
                results = Theta(data, parameters=input_theta_parameters)
    
    # Display and allow download of results
    if results is not None:
        st.write("Results:")
        st.write(results)
        st.download_button(
            label="Download Results",
            data=results.to_csv(index=False),
            file_name=f"{analysis_type.replace(' ', '_')}_results.csv",
            mime="text/csv",
        )
    else:
        st.info("Click 'Run' to compute the results.")
else:
    st.write("Upload a data file to begin.")
