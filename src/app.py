import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("power_availability_app")

# -----------------------------
# Load dataset
# -----------------------------
file_path = os.path.join("..", "cleaned_data", "processed_data.csv")
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])
logger.info("Dataset loaded successfully.")

# -----------------------------
# Load trained BiLSTM model
# -----------------------------
@st.cache_resource
def get_model():
    model_path = os.path.join("..", "model_metrics", "BiLSTM Hyperband.keras")
    return load_model(model_path)

model = get_model()
logger.info("Model loaded successfully.")

# -----------------------------
# Prepare encoders and scalers
# -----------------------------
@st.cache_resource
def prepare_preprocessing():
    """Prepare label encoders and scalers using training data"""
    # Label encode categorical features (same as training)
    le_feeder = LabelEncoder()
    le_district = LabelEncoder()
    
    data_copy = df.copy()
    data_copy['feeder_name'] = le_feeder.fit_transform(data_copy['feeder_name'])
    data_copy['district'] = le_district.fit_transform(data_copy['district'])
    
    # Features used during training
    features_for_model = [
        'feeder_name', 'district', 'consumption_mwh', 
        'lag1_avail', 'lag2_avail', 'lag3_avail', 'yesterday_full'
    ]
    
    X = data_copy[features_for_model]
    y = data_copy['availability_hrs']
    
    # Fit scalers on training portion (80% split as in training)
    split_idx = int(len(X) * 0.8)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaler_X.fit(X[:split_idx])
    scaler_y.fit(y[:split_idx].to_numpy().reshape(-1, 1))
    
    return le_feeder, le_district, scaler_X, scaler_y

le_feeder, le_district, scaler_X, scaler_y = prepare_preprocessing()
logger.info("Preprocessing components prepared successfully.")

# -----------------------------
# Precompute avg consumption
# -----------------------------
avg_consumption = df.groupby("feeder_type")["consumption_mwh"].mean()

# Add Healthcare manually
avg_consumption["Healthcare"] =avg_consumption.mean()  # baseline demand proxy


# Streamlit UI
# -----------------------------
st.title("⚡ Power Availability Prediction & Allocation")

supply_mw = st.number_input("Enter total energy supply (MWh)", min_value=10, step=10)
feeder_name = st.selectbox("Select Feeder", sorted(df["feeder_name"].unique()))
selected_date = st.date_input("Select Date for Prediction")

# ---- Prepare features for prediction ----
# Get historical data for the selected feeder
feeder_history = df[df["feeder_name"] == feeder_name].copy()

# We need 7 timesteps of historical data
time_steps = 7
if len(feeder_history) < time_steps:
    st.error(f"Not enough historical data for {feeder_name}. Need at least {time_steps} days.")
    st.stop()

# Get the last 7 rows of historical data for this feeder
last_7_days = feeder_history.tail(time_steps).copy()

# Encode categorical features
feeder_name_encoded = le_feeder.transform(last_7_days['feeder_name'])
district_encoded = le_district.transform(last_7_days['district'])

# Create feature array in the correct order (matching training)
X_input = np.column_stack([
    feeder_name_encoded,
    district_encoded,
    last_7_days['consumption_mwh'].values,
    last_7_days['lag1_avail'].values,
    last_7_days['lag2_avail'].values,
    last_7_days['lag3_avail'].values,
    last_7_days['yesterday_full'].values
])

# Scale the features
X_scaled = scaler_X.transform(X_input)

# Reshape for LSTM: (1 sample, 7 timesteps, 7 features)
X_scaled = X_scaled.reshape(1, time_steps, 7)

# ---- Predict availability ----
pred_scaled = model.predict(X_scaled, verbose=0)
pred_avail_hrs = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
pred_avail_hrs = int(np.round(np.clip(pred_avail_hrs, 1, 24)))

st.success(f"Predicted Availability Hours for {feeder_name}: {pred_avail_hrs} hrs")

# ---- Allocation ----

time_windows = [
    (0, 5, ["Healthcare", "Residential"]),
    (5, 9, ["Healthcare", "Residential", "Commercial"]),
    (9, 12, ["Healthcare", "Industrial", "Commercial"]),
    (12, 15, ["Healthcare", "Industrial"]),
    (15, 18, ["Healthcare", "Industrial", "Commercial"]),
    (18, 23, ["Healthcare", "Residential", "Commercial"]),
    (23, 24, ["Healthcare", "Residential"]),
]


# Collect feeder types
all_feeders = sorted({f for _, _, feeders in time_windows for f in feeders})

# Dataframe for 24 hours
hourly_allocation = pd.DataFrame(0.0, index=range(24), columns=all_feeders)

# Available hours and total supply
available_hours = pred_avail_hrs
total_supply = supply_mw

# Track how many hours we've allocated
allocated_hours = 0

# Go through each time window and allocate for available hours
for start, end, feeders in time_windows:
    for h in range(start, end):
        if allocated_hours >= available_hours:
            break
        
        # For each available hour, allocate the full supply among active feeders
        alloc = {}
        
        # Step 1: Healthcare gets 40% if present
        if "Healthcare" in feeders:
            alloc["Healthcare"] = 0.4 * total_supply
            remaining_supply = 0.6 * total_supply  # 60% remaining
        else:
            remaining_supply = total_supply
        
        # Step 2: Distribute remaining supply among other feeders proportionally
        others = [f for f in feeders if f != "Healthcare"]
        if others and remaining_supply > 0:
            # Get consumption weights for proportional allocation
            others_consumption = avg_consumption[others]
            total_others_consumption = others_consumption.sum()
            
            for feeder in others:
                weight = others_consumption[feeder] / total_others_consumption
                alloc[feeder] = weight * remaining_supply
        
        # Step 3: Fill allocation table for this hour
        for feeder, allocation in alloc.items():
            hourly_allocation.loc[h, feeder] = allocation
        
        allocated_hours += 1
# -----------------------
# Display Results
# -----------------------
st.subheader("Allocation Results by Hour (MW)")

# Convert hour index to time format
hourly_allocation_display = hourly_allocation.copy()
hourly_allocation_display.index = [f"{h:02d}:00" for h in hourly_allocation_display.index]
hourly_allocation_display.index.name = "Time of Day"

st.dataframe(hourly_allocation_display)

#Average per hour allocation (only for hours with allocation)
available_hours_mask = hourly_allocation.sum(axis=1) > 0
avg_per_hour = hourly_allocation[available_hours_mask].mean()

# --- Bar Chart: Average Per-Hour Allocation ---
fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
ax_bar.bar(avg_per_hour.index, avg_per_hour.values, color='skyblue')
ax_bar.set_xlabel("Feeder Type")
ax_bar.set_ylabel("Average Allocated Supply per Hour (MW)")
ax_bar.set_title("Average Energy Allocation per Hour by Feeder")
ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig_bar)

# Pie chart (total distribution)
fig, ax = plt.subplots()
ax.pie(avg_per_hour, labels=avg_per_hour.index, autopct="%1.1f%%")
ax.set_title("Average Energy Allocation Distribution by Feeder")
st.pyplot(fig)

# --- Line Chart: Hourly Allocation Trend ---
fig_line, ax_line = plt.subplots(figsize=(10, 5))
# Use display version with time labels for the chart
hourly_allocation_display.plot(ax=ax_line)
ax_line.set_xlabel("Time of Day")
ax_line.set_ylabel("Allocated Supply (MW)")
ax_line.set_title("Hourly Energy Allocation by Feeder")
ax_line.legend(title="Feeder Type")
ax_line.grid(True, linestyle='--', alpha=0.5)
# Rotate x-axis labels for better readability
ax_line.tick_params(axis='x', rotation=45)
st.pyplot(fig_line)