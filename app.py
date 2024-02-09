import numpy as np
import pandas as pd
import pickle
import streamlit as st

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))
st.title('Laptop Price Predictor')
st.image('laptop_image.jpg', width = 500)
company = st.selectbox('Brand', df['Company'].unique())

type = st.selectbox('Type', df['TypeName'].unique())

ram = st.selectbox('Ram(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

weight = st.number_input('Weight')

touchscreen = st.radio('Touchscreen', ['No', 'Yes'], horizontal=True)

ips = st.radio('IPS', ['No', 'Yes'], horizontal= True)

screen_size = st.number_input('Screen Size')

resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900',
                                                '3840x2160', '3840x2160','3200x1800', '2880x1800',
                                                '2560x1600', '2560x1440', '2304x1440'])

cpu = st.selectbox('Cpu', df['cpu_brand'].unique())
hdd = st.selectbox('HDD(in GB)', [0,128,256,512,1024,2048])
ssd = st.selectbox('SSD(in GB)', [0,128,256,512,1024,2048])
gpu = st.selectbox('Gpu', df['gpu_brand'].unique())
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = int(1)
    else:
        touchscreen = int(0)
    if ips == 'Yes':
        ips = int(1)
    else:
        ips = int(0)
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = float((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    query = np.array([company, type, int(ram), float(weight), int(touchscreen), int(ips),int(ppi),
                      cpu, int(hdd), int(ssd), gpu, os])
    query = query.reshape(1, 12)
    predicted = np.exp(pipe.predict(pd.DataFrame([[company, type, ram, weight, touchscreen, ips,ppi,cpu, hdd, ssd, gpu, os]],
                                                        columns = ['Company', 'TypeName', 'Ram', 'Weight','Touchscreen', 'ips', 'ppi',
                                                                     'cpu_brand', 'HDD', 'SSD', 'gpu_brand', 'os'])))
    predicted = np.round(predicted, 2)
    st.title('The Predicted Price --- '+ str(predicted[0]))
