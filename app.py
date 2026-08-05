
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Page Configuration
st.set_page_config(
    page_title="Manufacturing Efficiency Dashboard",
    page_icon="🏭",
    layout="wide"
)


st.title("🏭 Manufacturing Efficiency Prediction Dashboard")

st.write(
    "Machine Learning based Manufacturing Efficiency Prediction System"
)


# Load Dataset and Model Files

data = pd.read_csv("Thales_Group_Manufacturing.csv")

model = joblib.load("manufacturing_model.pkl")

scaler = joblib.load("scaler.pkl")

label_encoder = joblib.load("label_encoder.pkl")


# Sidebar Menu

menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Home",
        "Dataset",
        "Visualization",
        "Prediction",
        "Model"
    ]
)


# Home Page

if menu == "Home":

    st.header("Project Overview")

    st.write("""
    This project predicts manufacturing machine efficiency
    using Machine Learning.

    The model predicts:

    - High Efficiency
    - Medium Efficiency
    - Low Efficiency

    The system uses manufacturing sensor data such as:

    - Temperature
    - Vibration
    - Power Consumption
    - Production Speed
    - Error Rate
    - Maintenance Score
    """)



# Dataset Page

elif menu == "Dataset":

    st.header("Manufacturing Dataset")

    st.dataframe(data.head())

    st.write("Dataset Shape:")
    st.write(data.shape)

    st.write("Dataset Columns:")
    st.write(list(data.columns))



# Visualization Page

elif menu == "Visualization":

    st.header("Efficiency Status Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.countplot(
        data=data,
        x="Efficiency_Status",
        ax=ax
    )

    ax.set_title(
        "Machine Efficiency Distribution"
    )

    st.pyplot(fig)



# Prediction Page

elif menu == "Prediction":

    st.header("Predict Machine Efficiency")


    machine_id = st.number_input(
        "Machine ID",
        value=0
    )

    operation_mode = st.number_input(
        "Operation Mode",
        value=0
    )

    temperature = st.number_input(
        "Temperature (°C)"
    )

    vibration = st.number_input(
        "Vibration (Hz)"
    )

    power = st.number_input(
        "Power Consumption (kW)"
    )

    latency = st.number_input(
        "Network Latency (ms)"
    )

    packet_loss = st.number_input(
        "Packet Loss (%)"
    )

    defect_rate = st.number_input(
        "Quality Defect Rate (%)"
    )

    production_speed = st.number_input(
        "Production Speed"
    )

    maintenance_score = st.number_input(
        "Maintenance Score"
    )

    error_rate = st.number_input(
        "Error Rate (%)"
    )


    # Feature Engineering

    energy_efficiency = (
        production_speed / power
        if power != 0 else 0
    )

    error_output_ratio = (
        error_rate / production_speed
        if production_speed != 0 else 0
    )

    network_reliability = (
        100 - (packet_loss + latency/10)
    )

    sensor_stability = (
        temperature + vibration
    ) / 2



    input_data = pd.DataFrame({

        "Machine_ID":[machine_id],

        "Operation_Mode":[operation_mode],

        "Temperature_C":[temperature],

        "Vibration_Hz":[vibration],

        "Power_Consumption_kW":[power],

        "Network_Latency_ms":[latency],

        "Packet_Loss_%":[packet_loss],

        "Quality_Control_Defect_Rate_%":[defect_rate],

        "Production_Speed_units_per_hr":[production_speed],

        "Predictive_Maintenance_Score":[maintenance_score],

        "Error_Rate_%":[error_rate],

        "Energy_Efficiency":[energy_efficiency],

        "Error_Output_Ratio":[error_output_ratio],

        "Network_Reliability":[network_reliability],

        "Sensor_Stability":[sensor_stability]

    })



    if st.button("Predict"):

        scaled_data = scaler.transform(
            input_data
        )


        prediction = model.predict(
            scaled_data
        )


        result = label_encoder.inverse_transform(
            prediction
        )


        st.success(
            "Predicted Efficiency Status : "
            + result[0]
        )



# Model Page

elif menu == "Model":

    st.header("Machine Learning Model")

    st.success(
        type(model).__name__
    )

    st.write(
        "Model loaded successfully."
    )

    st.write(
        """
        Files used:

        - manufacturing_model.pkl
        - scaler.pkl
        - label_encoder.pkl
        """
    )
