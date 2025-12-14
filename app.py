import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

# 1. Load Model
model_data = joblib.load('model_breast_cancer.pkl')
model = model_data['model']
feature_names = model_data['feature_names']

# 2. Ambil Data Asli untuk cari Rata-rata (Biar data kosongnya gak 0)
data_raw = load_breast_cancer()
default_values = data_raw.data.mean(axis=0) # Ini nilai rata-rata pasien umum

st.set_page_config(page_title="Prediksi Kanker", layout="wide")

st.title("🩺 Aplikasi Prediksi Breast Cancer")
st.write("Dibuat oleh: **Abdul Jalil Arliansyah**")
st.write("Geser slider di sebelah kiri untuk memasukkan data pasien.")

st.sidebar.header("Input Parameter Pasien")

def user_input_features():
    # Mengambil nilai default rata-rata dulu
    input_data = default_values.copy()
    
    # User cuma ubah 5 fitur utama (biar simpel), sisanya pakai rata-rata
    # Rentang slider disesuaikan dengan nilai min-max asli dataset
    radius = st.sidebar.slider('Mean Radius (Ukuran)', 6.0, 30.0, float(default_values[0]))
    texture = st.sidebar.slider('Mean Texture (Tekstur)', 9.0, 40.0, float(default_values[1]))
    perimeter = st.sidebar.slider('Mean Perimeter (Keliling)', 40.0, 190.0, float(default_values[2]))
    area = st.sidebar.slider('Mean Area (Luas)', 140.0, 2500.0, float(default_values[3]))
    smoothness = st.sidebar.slider('Mean Smoothness (Kehalusan)', 0.05, 0.16, float(default_values[4]))
    
    # Update data array dengan input user
    input_data[0] = radius
    input_data[1] = texture
    input_data[2] = perimeter
    input_data[3] = area
    input_data[4] = smoothness
    
    # Ubah jadi dataframe biar model ngerti
    features = pd.DataFrame([input_data], columns=feature_names)
    return features

input_df = user_input_features()

# Tampilkan input user
st.subheader("Data Pasien:")
st.dataframe(input_df.iloc[:, :5]) # Cuma tampilkan 5 kolom utama biar rapi

st.markdown("---")

# Tombol Prediksi
if st.button('🔍 Prediksi Sekarang', type='primary'):
    prediction = model.predict(input_df)
    probabilitas = model.predict_proba(input_df)
    
    st.subheader('Hasil Analisis:')
    
    if prediction[0] == 0:
        st.error(f'⚠️ HASIL: Kanker GANAS (Malignant)')
        st.write(f"Tingkat Keyakinan Model: {probabilitas[0][0] * 100:.2f}%")
    else:
        st.success(f'✅ HASIL: Kanker JINAK (Benign)')
        st.write(f"Tingkat Keyakinan Model: {probabilitas[0][1] * 100:.2f}%")
