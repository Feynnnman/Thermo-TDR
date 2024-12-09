import streamlit as st
import pandas as pd

# Import functions from the scripts in the same directory
from electrical_conductivity import EC
from thermal_properties import thermal_properties
from water_content import water_content

# Streamlit interface
st.title("Thermo-TDR Tool")
st.sidebar.header("Upload your data")
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file:
    # Load uploaded data
    st.sidebar.header("Choose Function Type")
    analysis_type = st.sidebar.selectbox("Select Function", ["Electrical Conductivity", "Thermal Properties", "Water Content"])

    # Read the uploaded data
    data = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
    st.write("Uploaded Data:")
    st.write(data.head())

    # Call the respective function based on analysis type
    if analysis_type == "Electrical Conductivity":
        st.header("Electrical Conductivity")
        results = EC(data)
    elif analysis_type == "Thermal Properties":
        st.header("Thermal Properties Analysis")
        results = thermal_properties(data)
    elif analysis_type == "Water Content":
        st.header("Water Content")
        results = water_content(data)

    # Display and allow download of results
    st.write("Results:")
    st.write(results)
    st.download_button(
        label="Download Results",
        data=results.to_csv(index=False),
        file_name=f"{analysis_type.replace(' ', '_')}_results.csv",
        mime="text/csv",
    )
else:
    st.write("Upload a data file to begin.")
