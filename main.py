import streamlit as st
import pandas as pd
from pathlib import Path

# Import the functions
from thermal_properties import Heat
from thermal_properties import thermal_data_prep
from electrical_conductivity import Sigma
from water_content import Theta

# Streamlit interface
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Select a file to upload")

# Initialize results variable
results = None

def read_file(uploaded_file):

    file_extension = Path(uploaded_file.name).suffix.lower()  # Get the file extension
    
    readers = {
        '.xlsx': lambda f: pd.read_excel(f, header=None),
        '.csv': lambda f: pd.read_csv(f, delim_whitespace=True, header=None),
        '.dat': lambda f: pd.read_csv(f, delim_whitespace=True, header=None),    # Default format for thermo-TDR data
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

    st.sidebar.header("Select Analysis Type")
    analysis_type = st.sidebar.selectbox("Choose an analysis type", ["Thermal Properties", "Electrical Conductivity", "Water Content"])

    # Call the respective function
    if analysis_type == "Thermal Properties":


        # Data preview
        st.subheader("Data preview")
        path = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
        heat_data = thermal_data_prep(path)
        
        st.dataframe(heat_data, width=800, height=400)


        # Data visualization
        st.subheader("Data visualization")

        # Select rows to visualize
        row_number = st.slider("Rows to visualize", 1, len(heat_data), step=300)

        # Select columns to visualize
        columns = st.multiselect("Columns to visualize", heat_data.columns, default=heat_data.columns)

        # Plot the selected columns
        # If the user selects the "volt" column, plot it only in the right y-axis
        if "Volt" in columns:
            fig = heat_data.iloc[:row_number][columns].plot(secondary_y=["Volt"])
        else:
            fig = heat_data.iloc[:row_number][columns].plot()

        st.pyplot(fig.figure)


        # Calibration parameters
        st.subheader("Calibration Parameters")
        
        default_heat_parameters = {
            "Radius of the probe (m)": 6.35e-4,
            "Volumetric heat capacity of the probe (MJ m-3 K-1)": 2.84e6,
            "Probe spacing for Temperature Needle 1 (m)": 0.008,
            "Probe spacing for Temperature Needle 3 (m)": 0.008,
            "Resistance of the heating element (Ohm)": 887.6
        }
        
        # Create input fields for each parameter
        input_heat_parameters = {}
        for param, default_value in default_heat_parameters.items():
            # Determine step size based on the magnitude of the default value
            if abs(default_value) < 0.01 or abs(default_value) > 1000:
                step = float(f"{default_value:.1e}"[0]) * 10 ** (int(f"{default_value:.1e}".split('e')[1]))   
                input_heat_parameters[param] = st.number_input(
                    f"{param}", 
                    value=float(default_value),  # Ensure float type
                    format="%.1e",  # Scientific notation
                    step=float(step)  # Ensure float type
                )
            else:  # For normal values, use standard decimal format
                input_heat_parameters[param] = st.number_input(
                    f"{param}",
                    value=float(default_value),  # Ensure float type
                    format="%.1f",  # One decimal place
                    step=0.1  # Changed to 0.1 for better precision
                )


        # Add a "Run" button
        st.subheader("Run Analysis")
        if st.button("Run"):
            # Ensure all parameters are provided
            if all(value is not None for value in input_heat_parameters.values()):
                st.write("Processing calculations...")
                results = Heat(heat_data, parameters=input_heat_parameters)
    
    elif analysis_type == "Electrical Conductivity":
        st.header("Electrical Conductivity Analysis")

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
                st.write("Processing calculations...")
                results = Sigma(data, parameters=input_sigma_parameters)

    # Call the theta function
    elif analysis_type == "Water Content":
        st.header("Water Content Analysis")

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
                st.write("Processing calculations...")
                results = Theta(data, parameters=input_theta_parameters)
    
    # Display and allow download of results
    if results is not None:
        st.write("Analysis Complete!")
        st.write("Results:")
        st.write(results)
        st.download_button(
            label="Download Results as CSV",
            data=results.to_csv(index=False),
            file_name=f"{analysis_type.replace(' ', '_')}_results.csv",
            mime="text/csv",
        )
    else:
        st.info("Click 'Run' to start the analysis.")
else:
    st.write("Please upload a data file to begin.")
