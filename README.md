# Power System Availability Prediction & Energy Allocation

A ML system for predicting power system availability and optimizing energy allocation across different feeder types using deep learning models including BiLSTM, LSTM, and GRU networks.

## 🎯 Project Overview

This project addresses the critical challenge of power distribution management by:
- **Predicting availability hours** for 33KV power feeders using historical data
- **Optimizing energy allocation** across different feeder types (Residential, Commercial, Industrial, Healthcare)
- **Providing real-time insights** through an interactive Streamlit web application
- **Supporting decision-making** for power system operators and planners

## 🏗️ Project Structure

```
power-system-availability-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── cleaned_data/           # Processed datasets
│   ├── agbara_data.csv
│   ├── data_stat.csv
│   └── new_data.csv
├── data/                   # Raw datasets (33KV availability & consumption data)
├── model_metrics/          # Model performance and saved models
│   ├── best_model_metrics_sorted.csv
│   ├── best_model_metrics.csv
│   ├── BiLSTM_model.keras
│   └── model_metrics_comparison.csv
├── src/                    # Source code
│   ├── __init__.py
│   ├── app.py             # Streamlit web application
│   ├── data_clean_and_analysis.ipynb  # Data preprocessing & EDA
│   └── model_train.ipynb  # Model training & evaluation
└── tuning_params/         # Hyperparameter tuning results
    ├── bilstm_tuning/
    ├── gru_tuning/
    └── lstm_tuning/
```

## 🚀 Features

### 🔮 Prediction Capabilities
- **Availability Forecasting**: Predict daily availability hours for power feeders
- **Multi-Model Support**: BiLSTM, LSTM, GRU, and traditional ML models
- **Feature Engineering**: Time-based features, lag variables, and consumption patterns

### ⚡ Energy Allocation
- **Time-Window Based Allocation**: Different allocation strategies for different hours
- **Priority-Based Distribution**: Healthcare feeders get priority (40% allocation)
- **Proportional Allocation**: Remaining energy distributed based on consumption patterns

### 📊 Interactive Dashboard
- **Real-time Predictions**: Enter supply values and get instant availability predictions
- **Visual Analytics**: Bar charts, pie charts, and line plots for allocation insights
- **Feeder Selection**: Choose specific feeders for targeted analysis

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/power-system-availability-prediction.git
cd power-system-availability-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify data files**
Ensure your data files are in the `data/` directory:
- 33KV Daily Availability (2019-2021).xlsx
- 33KV Daily Consumption (2019-2021).xlsx

## 🎮 Usage

### Running the Web Application

```bash
cd src
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Using the Application

1. **Enter Energy Supply**: Input total available energy in MWh
2. **Select Feeder**: Choose from available 33KV feeders
3. **Pick Date**: Select date for prediction
4. **View Results**: Get availability predictions and allocation strategies

### Data Processing & Model Training

1. **Data Cleaning & Analysis**
```bash
# Open and run the Jupyter notebook
jupyter notebook src/data_clean_and_analysis.ipynb
```

2. **Model Training**
```bash
# Open and run the model training notebook
jupyter notebook src/model_train.ipynb
```

## 🧠 Machine Learning Models

### Model Architecture

The project implements several deep learning models:

#### BiLSTM (Best Performing)
```python
def build_bilstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)),
        Bidirectional(LSTM(32)),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    return model
```

#### LSTM
- Standard LSTM layers with dropout regularization
- Hyperparameter tuning using Keras Tuner

#### GRU  
- Gated Recurrent Units for sequence modeling
- Optimized using Dragonfly algorithm

### Performance Metrics

Models are evaluated using:
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error) 
- **RMSE** (Root Mean Square Error)
- **R² Score** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)

## 📈 Energy Allocation Strategy

### Time-Based Windows
```python
time_windows = [
    (0, 5, ["Healthcare", "Residential"]),           # Night hours
    (5, 9, ["Healthcare", "Residential", "Commercial"]),    # Morning
    (9, 12, ["Healthcare", "Industrial", "Commercial"]),    # Business hours
    (12, 15, ["Healthcare", "Industrial"]),          # Afternoon peak
    (15, 18, ["Healthcare", "Industrial", "Commercial"]),   # Evening business
    (18, 23, ["Healthcare", "Residential", "Commercial"]),  # Evening residential
    (23, 24, ["Healthcare", "Residential"]),         # Late night
]
```

### Allocation Rules
1. **Healthcare Priority**: Always receives 40% when active
2. **Proportional Distribution**: Remaining 60% distributed based on historical consumption
3. **Time-Sensitive**: Different feeder combinations for different hours

## 📊 Data Features

### Input Features
- `feeder_id`: Unique identifier for each feeder
- `consumption_mwh`: Average consumption in MWh
- `feeder_type`: Type classification (Residential/Commercial/Industrial)
- `day_of_week`: Day of the week (0-6)
- `month`: Month of the year (1-12)
- `is_weekend`: Weekend indicator (0/1)
- `lag1_avail`: Previous day's availability hours

### Target Variable
- `availability_hrs`: Daily availability hours (1-24)

## 🔧 Configuration

### Model Parameters
Key hyperparameters for BiLSTM model:
- **Units**: 128 (first layer), 32 (second layer)
- **Dropout Rate**: 0.3
- **Learning Rate**: 0.0001325
- **Batch Size**: 32
- **Epochs**: 50

### Logging
Application logs are saved to [`app.log`](src/app.py) with INFO level logging for debugging and monitoring.

## 📝 Dependencies

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
tensorflow>=2.13.0
scikit-learn>=1.3.0
keras-tuner>=1.4.0
xgboost>=1.7.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dr. Amole** - Project Supervisor
- **Bells University** - Research Support
- **Power Distribution Companies** - Data Provision
- **TensorFlow/Keras Team** - Deep Learning Framework
- **Streamlit Team** - Web Application Framework

## 📞 Contact

**Project Team**: Bells COLENG Research Group  
**Supervisor**: Dr. Amole  
**Institution**: Bells University of Technology

---

**Note**: This project is part of ongoing research in "Optimization of Power System Availability" at Bells University College of Engineering.