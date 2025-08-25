import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import datetime
from datetime import time
import pytz
import time as t

# Set page configuration
st.set_page_config(
    page_title="SHEM",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        color: #00796b; /* Dark Teal */
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sub-header styling */
    .sub-header {
        font-size: 1.8rem;
        color: #43a047; /* Energetic Green */
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    /* General card styling with dynamic shadow */
    .card {
        background-color: #e6f7ff; /* Light Blue */
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem;
        border-left: 8px solid #1f77b4;
    }
    
    /* Metric card styling with vibrant gradient */
    .metric-card {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        padding: 1.2rem;
        border-radius: 1rem;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15);
        margin-bottom: 1rem;
        text-align: center;
        border: 1px solid #00acc1;
    }

    /* Streamlit's default metric color updates */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #00796b;
    }

    [data-testid="stMetricDelta"] {
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    /* Recommendation card styling */
    .recommendation-card {
        background-color: #e3f2fd; /* Light Blue */
        padding: 1.2rem;
        border-radius: 0.7rem;
        border-left: 5px solid #1e88e5; /* Blue */
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Alert card styling */
    .alert-card {
        background-color: #ffcdd2; /* Light Red */
        padding: 1.2rem;
        border-radius: 0.7rem;
        border-left: 5px solid #e53935; /* Red */
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Positive and negative text colors */
    .positive {
        color: #43a047;
        font-weight: bold;
    }
    .negative {
        color: #e53935;
        font-weight: bold;
    }
    
    /* Navigation button styling */
    .nav-button {
        background-color: #ffffff;
        border: 2px solid #b0bec5;
        color: #546e7a;
        padding: 0.75rem 1.5rem;
        border-radius: 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        margin: 0.5rem;
        cursor: pointer;
    }
    .nav-button:hover {
        background-color: #cfd8dc;
        border-color: #78909c;
        color: #263238;
    }
    .nav-button.active {
        background: linear-gradient(45deg, #00acc1, #26a69a);
        color: white;
        border-color: #00796b;
    }
    
    /* Real-time monitoring specific styles */
    .real-time-card {
        background-color: #fff3e0; /* Light Orange */
        padding: 1.2rem;
        border-radius: 0.7rem;
        border-left: 5px solid #ff9800; /* Orange */
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .device-status {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        border-bottom: 1px solid #eee;
    }
    
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.5rem;
    }
    
    .status-on {
        background-color: #4caf50;
    }
    
    .status-off {
        background-color: #f44336;
    }
    
    .status-standby {
        background-color: #ff9800;
    }
</style>
""", unsafe_allow_html=True)

# Load data and model functions (update paths as needed)
@st.cache_resource
def load_data():
    try:
        data_path = "rawdata/smart_home_energy_v3.csv"
        df = pd.read_csv(data_path)

        # Rename columns to a consistent, lowercase format
        if 'Timestamp' in df.columns:
            df.rename(columns={'Timestamp': 'timestamp'}, inplace=True)
        if 'Energy_Consumption(kWh)' in df.columns:
            df.rename(columns={'Energy_Consumption(kWh)': 'energy_consumption'}, inplace=True)
        if 'Peak_Hours' in df.columns:
            df.rename(columns={'Peak_Hours': 'peak_hours'}, inplace=True)
        if 'Appliance_Usage(kWh)' in df.columns:
            df.rename(columns={'Appliance_Usage(kWh)': 'appliance_usage'}, inplace=True)
        
        # Calculate a new energy_cost column based on peak hours and consumption
        base_rate = 0.12
        peak_rate = 0.25
        df['energy_cost'] = df.apply(lambda row: row['energy_consumption'] * (peak_rate if row['peak_hours'] == 1 else base_rate), axis=1)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    except FileNotFoundError:
        st.error(f"Error: The file at {data_path} was not found. Using sample data.")
        return create_sample_data()
    except Exception as e:
        st.error(f"Error loading data from file: {e}. Using sample data for demonstration.")
        return create_sample_data()

@st.cache_resource
def load_model_and_preprocessor():
    try:
        preprocessor_path = "C:/Users/kalur/OneDrive/Documents/Desktop/SmartHomeEnergyManagement/artifacts/preprocessor.pkl"
        model_path = "C:/Users/kalur/OneDrive/Documents/Desktop/SmartHomeEnergyManagement/artifacts/model.pkl" # Corrected path
        
        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(model_path)
        return model, preprocessor
    except Exception as e:
        st.warning(f"Error loading model files: {e}. Prediction functionality will be disabled.")
        return None, None
    
def create_sample_data():
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'energy_consumption': np.random.normal(3.5, 1.2, len(dates)) + 
                              np.sin(np.arange(len(dates)) * 0.05) * 1.5 +
                              (np.arange(len(dates)) % 24 / 24) * 2,
        'temperature': np.random.normal(72, 12, len(dates)) + 
                        np.sin(np.arange(len(dates)) * 0.005) * 20,
        'humidity': np.random.normal(50, 15, len(dates)),
        'occupancy': np.random.poisson(0.8, len(dates)),
        'appliance_usage': np.random.gamma(2, 0.5, len(dates)),
        'solar_generation': np.maximum(0, np.random.normal(2.5, 1.5, len(dates)) * np.sin((np.arange(len(dates)) % 24 - 6) * np.pi/12)),
        'energy_cost': np.random.normal(0.15, 0.03, len(dates)) * (1 + 0.5 * ((np.arange(len(dates)) % 24 >= 16) & (np.arange(len(dates)) % 24 <= 21)))
    })
    
    return data

def get_device_data():
    devices = {
        "HVAC System": {"status": "on", "power": 3.2, "mode": "cooling", "temp": 72},
        "Refrigerator": {"status": "on", "power": 0.15, "temp": 38},
        "Water Heater": {"status": "standby", "power": 0.08, "temp": 120},
        "Lighting": {"status": "on", "power": 0.45, "brightness": 80},
        "Washing Machine": {"status": "off", "power": 0.0, "cycle": "completed"},
        "TV": {"status": "on", "power": 0.12, "source": "Streaming"},
        "Dishwasher": {"status": "off", "power": 0.0, "cycle": "not started"},
        "Computer": {"status": "on", "power": 0.18, "usage": "browsing"}
    }
    
    for device, data in devices.items():
        if data["status"] == "on":
            data["power"] += np.random.normal(0, 0.02)
            data["power"] = max(0.01, data["power"])
    
    return devices

# New function to handle the ML prediction
def predict_energy_consumption(model, preprocessor, input_df):
    try:
        # Preprocess the data using the loaded preprocessor
        transformed_data = preprocessor.transform(input_df)
        
        # Make the prediction
        prediction = model.predict(transformed_data)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# Load data and model
data = load_data()
model, preprocessor = load_model_and_preprocessor()

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

# Header and Navigation
st.markdown('<h1 class="main-header">🏠 SHEM - Smart Homes Energy Management</h1>', unsafe_allow_html=True)
st.markdown("### Data-Driven Insights to Optimize Energy Usage and Reduce Costs")

# Create navigation buttons
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("📊 Dashboard", use_container_width=True):
        st.session_state.current_page = "Dashboard"
with col2:
    if st.button("🔮 Energy Forecast", use_container_width=True):
        st.session_state.current_page = "Energy Forecast"
with col3:
    if st.button("⚡ Peak Demand", use_container_width=True):
        st.session_state.current_page = "Peak Demand"
with col4:
    if st.button("💡 Recommendations", use_container_width=True):
        st.session_state.current_page = "Recommendations"
with col5:
    if st.button("📡 Real-time Monitoring", use_container_width=True):
        st.session_state.current_page = "Real-time Monitoring"

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# ---
# Dashboard Page
if st.session_state.current_page == "Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Current Energy Usage", "3.8 kWh", "-0.6 kWh", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Today's Cost", "$2.05", "-$0.35", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Monthly Savings", "$18.40", "+$4.20")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Carbon Footprint", "42 kg CO₂", "-8%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Energy usage chart
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">📊 Energy Usage Trends</h3>', unsafe_allow_html=True)
    
    if data is not None:
        hourly_data = data.groupby(data['timestamp'].dt.hour).mean(numeric_only=True).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=hourly_data['timestamp'], y=hourly_data['energy_consumption'], 
                        name="Energy Usage", line=dict(color='#00796b', width=3)),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=hourly_data['timestamp'], y=hourly_data['energy_cost']*10, 
                        name="Cost (x10)", line=dict(color='#d84315', width=3, dash='dot')),
            secondary_y=True,
        )
        fig.update_layout(
            title="Hourly Energy Usage and Cost Patterns",
            xaxis_title="Hour of Day",
            hovermode="x unified",
            height=400,
            template="plotly_white",
            colorway=['#00796b', '#d84315']
        )
        fig.update_yaxes(title_text="Energy Usage (kWh)", secondary_y=False)
        fig.update_yaxes(title_text="Cost (¢/kWh)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not load energy data. Please check the data path.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Appliance usage and alerts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">🔌 Appliance Energy Usage</h3>', unsafe_allow_html=True)
        
        appliances = ['HVAC', 'Water Heater', 'Refrigerator', 'Lighting', 'Electronics', 'Other']
        usage = [38, 22, 14, 11, 9, 6]
        
        fig = px.pie(values=usage, names=appliances, title="Energy Usage by Appliance",
                      color_discrete_sequence=px.colors.sequential.Tealgrn)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">⚠ Energy Alerts</h3>', unsafe_allow_html=True)
        
        st.markdown('<div class="alert-card">', unsafe_allow_html=True)
        st.markdown("*High Usage Alert*")
        st.markdown("Peak energy consumption detected between 6-8 PM")
        st.markdown("Consider shifting some appliance usage to off-peak hours")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="alert-card">', unsafe_allow_html=True)
        st.markdown("*HVAC Efficiency Notice*")
        st.markdown("Your HVAC system is using 15% more energy than similar homes")
        st.markdown("Consider scheduling maintenance")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="alert-card">', unsafe_allow_html=True)
        st.markdown("*Peak Rate Period*")
        st.markdown("High electricity rates in effect: 4-9 PM Weekdays")
        st.markdown("Avoid running energy-intensive appliances during this time")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---
# ---
# Energy Forecast Page
elif st.session_state.current_page == "Energy Forecast":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">🔮 Energy Consumption Forecast</h2>', unsafe_allow_html=True)

    if model and preprocessor:
        st.success("ML Model and Preprocessor loaded successfully! Ready to generate forecast.")
        st.markdown("Enter your home and environmental details for a personalized energy consumption forecast.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Home & Environmental Inputs")
            temperature = st.slider("Forecasted Temperature (°C)", 0.0, 40.0, 24.0, 0.1)
            occupancy = st.slider("Number of Occupants", 0, 5, 2)
            appliance_usage = st.number_input("Typical Appliance Usage (kWh)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
            smart_lighting = st.number_input("Smart Lighting Usage (kWh)", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
            thermostat_setting = st.slider("Thermostat Setting (°C)", 18.0, 28.0, 23.0, 0.1)
            weather_options = ['Clear', 'Cloudy', 'Rain', 'Snow']
            selected_weather = st.selectbox("Expected Weather", weather_options)

        with col2:
            st.subheader("Time-Based Inputs")
            forecast_time = st.time_input("Forecast Time", datetime.time(12, 00))
            forecast_date = st.date_input("Forecast Date", datetime.date.today())
            
            # Extract features from datetime
            hour = forecast_time.hour
            dayofweek = forecast_date.weekday()

            # Determine peak hours (assuming peak is 4 PM to 9 PM on weekdays)
            peak_hours = 1 if (dayofweek < 5 and hour >= 16 and hour <= 21) else 0

        if st.button("Generate Forecast", type="primary"):
            # Create a dictionary with raw user inputs
            input_data = {
                'Temperature(°C)': [temperature],
                'Occupancy': [occupancy],
                'Appliance_Usage(kWh)': [appliance_usage],
                'Smart_Lighting(kWh)': [smart_lighting],
                'Thermostat_Setting(°C)': [thermostat_setting],
                'Weather': [selected_weather],
                'Peak_Hours': [peak_hours],
                'Hour': [hour],
                'DayOfWeek': [dayofweek],
            }
            
            # Convert dictionary to a DataFrame
            input_df = pd.DataFrame(input_data)
            
            # --- Feature Engineering to create the missing columns ---
            
            # IsWeekend feature
            input_df['IsWeekend'] = input_df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
            
            # Cyclical features for Hour (using 24 hours)
            input_df['Hour_sin'] = np.sin(2 * np.pi * input_df['Hour'] / 24.0)
            input_df['Hour_cos'] = np.cos(2 * np.pi * input_df['Hour'] / 24.0)
            
            # Cyclical features for DayOfWeek (using 7 days)
            input_df['DayOfWeek_sin'] = np.sin(2 * np.pi * input_df['DayOfWeek'] / 7.0)
            input_df['DayOfWeek_cos'] = np.cos(2 * np.pi * input_df['DayOfWeek'] / 7.0)
            
            # Alert feature (example logic)
            input_df['Alert'] = input_df.apply(lambda row: 1 if row['Temperature(°C)'] > 28 and row['Peak_Hours'] == 1 else 0, axis=1)

            # --- Now, get prediction with the complete DataFrame ---
            predicted_usage = predict_energy_consumption(model, preprocessor, input_df)

            if predicted_usage is not None:
                rate = 0.25 if peak_hours == 1 else 0.12
                predicted_cost = predicted_usage * rate
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Predicted Energy Usage", f"{predicted_usage:.2f} kWh")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Predicted Cost", f"${predicted_cost:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown(f"**Note:** This prediction is for the hour of `{forecast_time.strftime('%I:%M %p')}` on `{forecast_date.strftime('%A, %B %d')}`.")
                st.markdown(f"Factors considered: **Temperature** ({temperature}°C), **Occupants** ({occupancy}), and **Peak Hour Status** ({'Yes' if peak_hours == 1 else 'No'}).")

    else:
        st.warning("Prediction model not available. Please check the file paths and ensure the model has been trained and saved correctly.")
        st.info("You can train the model by running the `train_pipeline.py` script.")

    st.markdown('</div>', unsafe_allow_html=True)

# The rest of the code remains the same.
# ---
# Peak Demand Page
elif st.session_state.current_page == "Peak Demand":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">⚡ Peak Demand Management</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Peak Rate Periods")
        st.info("Higher electricity rates typically apply during these hours:")
        
        peak_hours = {
            "Weekdays": "4:00 PM - 9:00 PM",
            "Weekends": "No peak pricing"
        }
        
        for day_type, hours in peak_hours.items():
            st.write(f"{day_type}: {hours}")
        
        now = datetime.datetime.now()
        current_hour = now.hour
        is_peak = (now.weekday() < 5) and (current_hour >= 16 and current_hour < 21)
        
        if is_peak:
            st.error("⚠ You are currently in a peak rate period!")
        else:
            st.success("You are not currently in a peak rate period")
    
    with col2:
        st.subheader("Potential Savings")
        st.metric("Monthly Peak Usage Cost", "$42.50")
        st.metric("Potential Savings with Shifting", "$12.75", "30%")
        
        st.progress(0.3)
    
    st.subheader("Peak Reduction Strategies")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
        st.markdown("⏰ Shift Appliance Usage**")
        st.markdown("- Run dishwasher after 9 PM")
        st.markdown("- Do laundry on weekends")
        st.markdown("- Use delay start features")
        st.markdown("*Savings: $5-8/month*")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
        st.markdown("🌡 Adjust Thermostat**")
        st.markdown("- Set 2-3° higher during peak hours in summer")
        st.markdown("- Set 2-3° lower during peak hours in winter")
        st.markdown("- Use fans for comfort")
        st.markdown("*Savings: $4-6/month*")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
        st.markdown("🔌 Reduce Phantom Loads**")
        st.markdown("- Use smart power strips")
        st.markdown("- Unplug unused electronics")
        st.markdown("- Enable energy saving modes")
        st.markdown("*Savings: $3-5/month*")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.subheader("Peak Usage Analysis")
    
    peak_data = pd.DataFrame({
        'Hour': list(range(24)),
        'Usage (kWh)': [1.2, 1.0, 0.9, 0.8, 0.9, 1.2, 1.8, 2.1, 1.9, 1.7, 1.6, 1.5, 
                        1.6, 1.5, 1.6, 1.8, 2.5, 3.2, 3.8, 3.5, 2.8, 2.2, 1.8, 1.4],
        'Cost per kWh': [0.12] * 24
    })
    
    peak_data.loc[(peak_data['Hour'] >= 16) & (peak_data['Hour'] < 21), 'Cost per kWh'] = 0.25
    
    peak_data['Hourly Cost'] = peak_data['Usage (kWh)'] * peak_data['Cost per kWh']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=peak_data['Hour'], y=peak_data['Usage (kWh)'], name="Energy Usage", marker_color='#43a047'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=peak_data['Hour'], y=peak_data['Cost per kWh'], name="Energy Cost", line=dict(dash='dash', color='#f9a825')),
        secondary_y=True,
    )
    fig.update_layout(
        title="Energy Usage and Cost by Hour",
        xaxis_title="Hour of Day",
        hovermode="x unified",
        template="plotly_white"
    )
    fig.update_yaxes(title_text="Energy Usage (kWh)", secondary_y=False)
    fig.update_yaxes(title_text="Cost ($/kWh)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---
# Recommendations Page
elif st.session_state.current_page == "Recommendations":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">💡 Personalized Energy Saving Recommendations</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
    st.markdown("🌡 HVAC Optimization**")
    st.markdown("- Set thermostat to 78°F when home and 85°F when away")
    st.markdown("- Use ceiling fans to feel 4-6°F cooler")
    st.markdown("- Schedule annual maintenance for efficiency")
    st.markdown("- Replace air filters monthly during peak usage")
    st.markdown("*Estimated monthly savings: $15-25*")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
    st.markdown("⏰ Appliance Scheduling**")
    st.markdown("- Shift dishwasher and laundry usage to off-peak hours (after 9 PM)")
    st.markdown("- Use delayed start features on appliances")
    st.markdown("- Run full loads only in dishwasher and washing machine")
    st.markdown("*Estimated monthly savings: $8-12*")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
    st.markdown("💡 *Lighting Efficiency*")
    st.markdown("- Replace 5 incandescent bulbs with LEDs (save $5-8/month)")
    st.markdown("- Use natural light during daytime hours")
    st.markdown("- Install motion sensors in low-traffic areas")
    st.markdown("- Utilize dimmers and timers for outdoor lighting")
    st.markdown("*Estimated monthly savings: $5-8*")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
    st.markdown("🔌 *Electronics Management*")
    st.markdown("- Enable power saving modes on computers and TVs")
    st.markdown("- Use smart power strips to eliminate phantom loads")
    st.markdown("- Unplug chargers when not in use")
    st.markdown("- Set game consoles to energy-saving mode")
    st.markdown("*Estimated monthly savings: $3-5*")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.subheader("Potential Savings Summary")
    
    savings_data = {
        'Category': ['HVAC', 'Appliance Scheduling', 'Lighting', 'Electronics'],
        'Savings': [20, 10, 6.5, 4]
    }
    
    fig = px.bar(savings_data, x='Category', y='Savings', 
                      title="Estimated Monthly Savings by Category ($)",
                      color='Savings', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Recommended Implementation Timeline")
    
    timeline_data = {
        'Task': ['Replace light bulbs', 'Adjust thermostat settings', 'Install smart power strips', 
                  'Schedule appliance usage', 'HVAC maintenance'],
        'Start': ['2023-09-01', '2023-09-01', '2023-09-15', '2023-09-15', '2023-10-01'],
        'Finish': ['2023-09-07', '2023-09-01', '2023-09-20', '2023-09-30', '2023-10-15'],
        'Completion': [100, 100, 30, 0, 0]
    }
    
    fig = px.timeline(timeline_data, x_start="Start", x_end="Finish", y="Task", 
                      title="Energy Efficiency Implementation Plan", color_discrete_sequence=['#4CAF50'])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---
# Real-time Monitoring Page
elif st.session_state.current_page == "Real-time Monitoring":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">📡 Real-time Energy Monitoring</h2>', unsafe_allow_html=True)
    
    # Create a placeholder for real-time updates
    real_time_placeholder = st.empty()
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=True)
    refresh_interval = st.slider("Refresh Interval (seconds)", 1, 10, 3)
    
    if auto_refresh:
        st.info(f"Auto-refresh enabled. Updating every {refresh_interval} seconds.")
    
    # Button to manually refresh
    if st.button("Refresh Now", type="primary"):
        # This will trigger a rerun
        pass
    
    # Get current time for display
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get device data
    device_data = get_device_data()
    
    # Calculate total current power
    total_power = sum(data["power"] for data in device_data.values())
    total_cost = total_power * 0.15  # Simplified cost calculation
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Current Power Usage", f"{total_power:.2f} kW", 
                    help="Total power being consumed right now")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Estimated Hourly Cost", f"${total_cost:.2f}", 
                    help="Cost based on current usage rate")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Last Updated", current_time, 
                    help="Time of the last data refresh")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display device status
    st.subheader("Device Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="real-time-card">', unsafe_allow_html=True)
        st.markdown("**Active Devices**")
        
        for device, data in device_data.items():
            if data["status"] == "on":
                status_class = "status-on"
                status_text = "ON"
                
                st.markdown(f"""
                <div class="device-status">
                    <div>
                        <span class="status-indicator {status_class}"></span>
                        <strong>{device}</strong>
                    </div>
                    <div>{data["power"]:.2f} kW</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="real-time-card">', unsafe_allow_html=True)
        st.markdown("**Inactive/Standby Devices**")
        
        for device, data in device_data.items():
            if data["status"] != "on":
                status_class = "status-off" if data["status"] == "off" else "status-standby"
                status_text = "OFF" if data["status"] == "off" else "STANDBY"
                
                st.markdown(f"""
                <div class="device-status">
                    <div>
                        <span class="status-indicator {status_class}"></span>
                        <strong>{device}</strong>
                    </div>
                    <div>{data["power"]:.2f} kW</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display energy usage chart
    st.subheader("Real-time Energy Consumption")
    
    # Generate some sample time series data for the chart
    time_points = 30
    times = [(datetime.datetime.now() - datetime.timedelta(minutes=i)).strftime("%H:%M") 
              for i in range(time_points-1, -1, -1)]
    
    # Create some realistic fluctuations
    base_usage = total_power
    usage_data = [base_usage * (0.9 + 0.2 * np.sin(i/3) + np.random.normal(0, 0.05)) 
                  for i in range(time_points)]
    
    chart_data = pd.DataFrame({
        'Time': times,
        'Power (kW)': usage_data
    })
    
    fig = px.area(chart_data, x='Time', y='Power (kW)', 
                  title="Power Consumption Over Last 30 Minutes",
                  color_discrete_sequence=['#4CAF50'])
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display alerts based on current usage
    st.subheader("Real-time Alerts")
    
    if total_power > 4.0:
        st.markdown('<div class="alert-card">', unsafe_allow_html=True)
        st.markdown("⚠️ **High Power Usage Alert**")
        st.markdown(f"Current usage ({total_power:.2f} kW) is above normal threshold.")
        st.markdown("Consider turning off non-essential devices.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Check for devices in standby that could be turned off
    standby_devices = [device for device, data in device_data.items() 
                       if data["status"] == "standby" and data["power"] > 0.05]
    
    if standby_devices:
        st.markdown('<div class="alert-card">', unsafe_allow_html=True)
        st.markdown("💤 **Standby Power Alert**")
        st.markdown(f"The following devices are drawing standby power: {', '.join(standby_devices)}")
        st.markdown("Consider unplugging them to save energy.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Auto-refresh logic
    if auto_refresh:
        t.sleep(refresh_interval)
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>SHEM - Smart Homes Energy Management | Helping you save energy, reduce costs, and minimize environmental impact</p>
    <p>© 2023 Data-Driven Sustainability Solutions</p>
</div>
""", unsafe_allow_html=True)