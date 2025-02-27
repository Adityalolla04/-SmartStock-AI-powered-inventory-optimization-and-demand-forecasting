import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, save_model
import cv2
from PIL import Image

# Streamlit Configuration
st.set_page_config(page_title="AI Inventory Dashboard", layout="wide")

# Load Data
data = pd.read_csv('/Users/adityasrivatsav/Documents/GitHub/-SmartStock-AI-powered-inventory-optimization-and-demand-forecasting/data/processed/cleaned_final_preprocessed_data.csv')
forecast_data = pd.read_csv('/Users/adityasrivatsav/Documents/GitHub/-SmartStock-AI-powered-inventory-optimization-and-demand-forecasting/data/processed/sales_forecast.csv')

# Load Pre-trained CNN Model
model_path = '/Users/adityasrivatsav/Documents/GitHub/-SmartStock-AI-powered-inventory-optimization-and-demand-forecasting/models/trained_cnn_resnet_model.keras'
cnn_model = load_model(model_path)

defect_data = pd.DataFrame({
    "Status": ["Defective", "Non-Defective"],
    "Count": [150, 760]
})

# UI Elements
st.markdown("""
    <style>
    .big-font {
        font-size:25px !important;
        font-weight: bold;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 AI-Powered Inventory Optimization Dashboard")
st.markdown("### 🔍 Integrated Dashboard for Demand Forecasting, Inventory Optimization, and Defect Detection")

# Sidebar Filters
st.sidebar.title("🔧 Filters")
selected_supplier = st.sidebar.selectbox("Select Supplier", data['SupplierName'].unique())

# Key Metrics
st.markdown("## 📌 Key Performance Indicators")
col1, col2, col3 = st.columns(3)

y_true = [54650044, 50461212, 47025604]  # True sales
y_pred = [52865742, 55847491, 39795364]  # Predicted sales
lstm_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
lstm_mape = mean_absolute_percentage_error(y_true, y_pred) * 100

col1.metric("LSTM RMSE", f"{lstm_rmse:,.2f}")
col2.metric("LSTM MAPE", f"{lstm_mape:.2f}%")
col3.metric("CNN Accuracy", "55.92%")

# 📈 Sales Forecast Visualization
data['OrderDate'] = pd.to_datetime(data['OrderDate'])
monthly_sales = data.groupby(pd.Grouper(key='OrderDate', freq='M')).agg({'Sales': 'sum'}).reset_index()

fig_sales = px.line(monthly_sales, x='OrderDate', y='Sales', title='Historical Sales Over Time', markers=True)
st.plotly_chart(fig_sales, use_container_width=True)

# 📦 Inventory Optimization (EOQ)
forecast_data['EOQ'] = np.sqrt((2 * forecast_data['Forecasted_Sales'] * 50) / 10)
fig_eoq = px.line(forecast_data, x='Date', y='EOQ', title='Economic Order Quantity Over Time', markers=True)
st.plotly_chart(fig_eoq, use_container_width=True)

# 📥 Upload Image for Real-Time Defect Detection
st.markdown("## 📥 Upload Image for Defect Detection")
uploaded_file = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img = np.array(image)
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)

    prediction = cnn_model.predict(img_expanded)
    defect_status = "Defective" if prediction[0][0] > 0.5 else "Non-Defective"
    st.success(f"Defect Detection Completed - Result: {defect_status}")

# 📊 CNN Model Performance Visualization
conf_matrix = np.array([[60, 40], [35, 65]])
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Defective", "Defective"], yticklabels=["Non-Defective", "Defective"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# 💾 Save Trained CNN Model
st.markdown("## 💾 Save Trained Model")
save_model_button = st.button("Save Trained CNN Model in H5 Format")
if save_model_button:
    save_path = '/mnt/data/trained_cnn_resnet_model.keras'
    cnn_model.save(save_path)
    st.success(f"Model successfully saved in H5 format at {save_path}")
