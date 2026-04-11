"""
Power Availability Prediction & Energy Allocation App
"""
import streamlit as st
import matplotlib.pyplot as plt
from utils import (
    load_data, load_prediction_model, setup_preprocessing,
    prepare_prediction_input, predict_availability, allocate_energy,
    format_allocation_display
)

# Page configuration
st.set_page_config(page_title="Power Availability Prediction", page_icon="⚡", layout="wide")


@st.cache_data
def load_dataset():
    """Load dataset with caching"""
    return load_data()


@st.cache_resource
def load_model():
    """Load model with caching"""
    return load_prediction_model()


@st.cache_resource
def get_preprocessing_components(_df):
    """Get preprocessing components with caching"""
    return setup_preprocessing(_df)


@st.cache_data
def get_avg_consumption(_df):
    """Calculate average consumption by feeder type"""
    avg_consumption = _df.groupby("feeder_type")["consumption_mwh"].mean()
    avg_consumption["Healthcare"] = avg_consumption.mean()
    return avg_consumption


# Load resources
df = load_dataset()
model = load_model()
le_feeder, le_district, scaler_X, scaler_y = get_preprocessing_components(df)
avg_consumption = get_avg_consumption(df)

# UI
st.title("⚡ Power Availability Prediction & Allocation")

# Input section
col1, col2, col3 = st.columns(3)

with col1:
    supply_mw = st.number_input("Total Energy Supply (MWh)", min_value=10, max_value=1000, value=60, step=10)

with col2:
    feeder_name = st.selectbox("Select Feeder", sorted(df["feeder_name"].unique()))

with col3:
    selected_date = st.date_input("Prediction Date")

# Prediction
if st.button("Predict & Allocate", type="primary"):
    try:
        # Prepare input and predict
        X_input = prepare_prediction_input(df, feeder_name, le_feeder, le_district)
        pred_hours = predict_availability(model, X_input, scaler_X, scaler_y)
        
        st.success(f"🔮 Predicted Availability: **{pred_hours} hours** for {feeder_name}")
        
        # Energy allocation
        hourly_allocation = allocate_energy(pred_hours, supply_mw, avg_consumption)
        allocation_display = format_allocation_display(hourly_allocation)
        
        # Display results
        st.subheader("📊 Hourly Energy Allocation")
        st.dataframe(allocation_display, use_container_width=True)
        
        # Visualizations
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Average allocation per feeder
            active_hours = hourly_allocation.sum(axis=1) > 0
            avg_per_hour = hourly_allocation[active_hours].mean()
            
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.bar(avg_per_hour.index, avg_per_hour.values, color='skyblue', edgecolor='navy')
            ax1.set_xlabel("Feeder Type", fontsize=11)
            ax1.set_ylabel("Average Supply (MW)", fontsize=11)
            ax1.set_title("Average Allocation per Hour", fontsize=12, fontweight='bold')
            ax1.grid(axis='y', linestyle='--', alpha=0.4)
            plt.xticks(rotation=15)
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col_right:
            # Distribution pie chart
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.pie(avg_per_hour, labels=avg_per_hour.index, autopct="%1.1f%%", startangle=90)
            ax2.set_title("Energy Distribution by Feeder", fontsize=12, fontweight='bold')
            st.pyplot(fig2)
        
        # Hourly trend
        st.subheader("📈 Hourly Allocation Trend")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        allocation_display.plot(ax=ax3, linewidth=2)
        ax3.set_xlabel("Time of Day", fontsize=11)
        ax3.set_ylabel("Allocated Supply (MW)", fontsize=11)
        ax3.set_title("Hourly Energy Allocation by Feeder Type", fontsize=12, fontweight='bold')
        ax3.legend(title="Feeder Type", loc='upper right')
        ax3.grid(True, linestyle='--', alpha=0.4)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig3)
        
    except ValueError as e:
        st.error(f"❌ Error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
else:
    st.info("👆 Click 'Predict & Allocate' to run the prediction")

