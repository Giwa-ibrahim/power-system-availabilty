import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.models import load_model

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
file_path = os.path.join('..', 'cleaned_data', 'new_data.csv')
df = pd.read_csv(file_path, index_col=0).reset_index(drop=True)
logger.info("Dataset loaded successfully.")

# -----------------------------
# Load trained BiLSTM model
# -----------------------------
@st.cache_resource
def get_model():
    model_path = os.path.join('..', 'model_metrics', 'BiLSTM_model.keras')
    return load_model(model_path)

model = get_model()
logger.info("Model loaded successfully.")

# -----------------------------
# Precompute avg consumption
# -----------------------------
avg_consumption = df.groupby("feeder_type")["consumption_mwh"].mean()

# Add Healthcare manually
avg_consumption["Healthcare"] = avg_consumption.mean()  # baseline demand proxy


# -----------------------------
# Allocation with time-windows
# -----------------------------
# def allocate_energy_with_timewindows(supply_mw, pred_avail_hrs, avg_consumption):
#     time_windows = {
#         "00:00–05:00": ["Healthcare", "Residential"],
#         "05:00–09:00": ["Healthcare", "Residential", "Commercial"],
#         "09:00–12:00": ["Healthcare", "Industrial", "Commercial"],
#         "12:00–15:00": ["Healthcare", "Industrial"],
#         "15:00–18:00": ["Healthcare", "Industrial", "Commercial"],
#         "18:00–23:00": ["Healthcare", "Residential", "Commercial"],
#         "23:00–24:00": ["Healthcare", "Residential"],
#     }

#     per_hour_supply = supply_mw / pred_avail_hrs
#     hourly_allocation = {h: {} for h in range(24)}

#     for window, feeders in time_windows.items():
#         start, end = window.split("–")
#         start_h = int(start.split(":")[0])
#         end_h = int(end.split(":")[0]) if end != "24:00" else 24

#         for h in range(start_h, end_h):
#             remaining = per_hour_supply
#             alloc = {}

#             # Healthcare always gets 40%
#             if "Healthcare" in feeders:
#                 alloc["Healthcare"] = 0.4 * per_hour_supply
#                 remaining -= alloc["Healthcare"]

#             # Split remainder across other feeders proportionally
#             others = [f for f in feeders if f != "Healthcare"]
#             if others:
#                 weights = avg_consumption[others] / avg_consumption[others].sum()
#                 for feeder in others:
#                     alloc[feeder] = weights[feeder] * remaining

#             hourly_allocation[h] = alloc

#     return hourly_allocation


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("⚡ Power Availability Prediction & Allocation")

supply_mw = st.number_input("Enter total energy supply (MWh)", min_value=10, step=10)
feeder_name = st.selectbox("Select Feeder", df["feeder_name"].unique())
selected_date = st.date_input("Select Date for Prediction")

# ---- Prepare features ----
feeder_history = df[df["feeder_name"] == feeder_name]
last_row = feeder_history.iloc[-1]

day_of_week = selected_date.weekday()
month = selected_date.month
is_weekend = 1 if day_of_week >= 5 else 0
lag1_avail = last_row["availability_hrs"]
avg_feed_consumption = feeder_history["consumption_mwh"].mean()
feeder_id = feeder_history["feeder_name"].astype("category").cat.codes.iloc[-1]
feeder_type = feeder_history["feeder_type"].astype("category").cat.codes.iloc[-1]

X_input = np.array([[feeder_id, avg_feed_consumption, feeder_type,
                     day_of_week, month, is_weekend, lag1_avail]])
X_input = X_input.reshape((1, X_input.shape[0], X_input.shape[1]))

# ---- Predict availability ----
pred_avail_hrs = int(model.predict(X_input).round().clip(1, 24)[0][0])
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