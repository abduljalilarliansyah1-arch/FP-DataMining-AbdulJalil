import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model_data = joblib.load('model_breast_cancer.pkl')
model = model_data['model']
feature_names = model_data['feature_names']

st.title("Aplikasi Prediksi Breast Cancer")
st.write("Dibuat oleh: Abdul Jalil Arliansyah")

st.sidebar.header("Input Parameter")

def user_input_features():
    # Input Slider sederhana
    radius = st.sidebar.slider('Mean Radius', 0.0, 30.0, 15.0)
    texture = st.sidebar.slider('Mean Texture', 0.0, 40.0, 20.0)
    perimeter = st.sidebar.slider('Mean Perimeter', 0.0, 200.0, 90.0)
    area = st.sidebar.slider('Mean Area', 0.0, 2500.0, 500.0)
    smoothness = st.sidebar.slider('Mean Smoothness', 0.0, 0.2, 0.1)
    
    # Kita buat array kosong seukuran fitur asli (30 fitur)
    data = np.zeros((1, 30))
    
    # Isi 5 fitur pertama dengan input user (sisanya biarkan 0/rata-rata)
    data[0, 0] = radius
    data[0, 1] = texture
    data[0, 2] = perimeter
    data[0, 3] = area
    data[0, 4] = smoothness
    
    features = pd.DataFrame(data, columns=feature_names)
    return features

input_df = user_input_features()

st.subheader('Hasil Prediksi')
if st.button('Prediksi Sekarang'):
    prediction = model.predict(input_df)
    
    if prediction[0] == 0:
        st.error('Hasil: Kanker GANAS (Malignant)')
    else:
        st.success('Hasil: Kanker JINAK (Benign)')
