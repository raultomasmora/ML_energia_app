# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from pycaret.regression import *


# entorno streamlit
st.write("""
# **Machine Learning Energy**
Esta aplicación predice el consumo de energía (kWh/m²) a partir de miles de datos de certificados energéticos elaborados con el programa **CE3X**.
""")

st.sidebar.header('Parámetros de entrada')

def user_input_features():
    # selectbox
    NORMATIVA_CONSTRUCCIO = st.sidebar.selectbox('Normativa de construcción', 
                            ("Ant_NBECT79", "NBECT79", "CTE2006"))
    SUPERFICI_HAB = st.sidebar.slider('Superficie habitable (m²)', 6. , 427. , 70.1 )
    COMPACITAT = st.sidebar.slider('Compacidad (m³/m²)', 0.2 , 25. , 3.4 )
    VENTILACIO_USO_RESIDENCIAL = st.sidebar.slider('Ventilación uso residencial (renovaciones/hora)', 0. , 4. , 0.6 )
    VENTILACIO_INFILTRACIONS = st.sidebar.slider('Ventilación por infiltraciones (renovaciones/hora)', 0. , 4. , 0.7 )
    DEMANDA_ACS = st.sidebar.slider('Demanda de ACS (litros/día)', 0. , 1122. , 71.5 )
    OPACOS_Fach_sum = st.sidebar.slider('Suma de superficies en fachada (m²)', 2.5 , 1125.9 , 45.7 )
    OPACOS_Fach_trans = st.sidebar.slider('Transmitancia térmica media en fachadas (W/m² K)', 0.1 , 4.1 , 1.7 )
    OPACOS_Cubi_sum = st.sidebar.slider('Suma de superficies en cubierta (m²)', 0. , 1224.3 , 11.7 )
    OPACOS_Cubi_trans = st.sidebar.slider('Transmitancia térmica media en cubiertas (W/m² K)', 0. , 5.7 , 0.4 )
    HUECOS_sum = st.sidebar.slider('Suma de superficies en huecos (m²)', 1. , 157.9 , 11.8 )
    HUECOS_trans = st.sidebar.slider('Transmitancia térmica media en huecos (W/m² K)', 0.6 , 7. , 4.4 )
    HUECOS_fsol = st.sidebar.slider('Factor solar promedio en huecos (g)', 0. , 1. , 0.5 )
    PUENTE_sum = st.sidebar.slider('Suma de longitudes con puentes térmicos (metros)', 0. , 1100.5 , 70.2 )
    PUENTE_trans = st.sidebar.slider('Transmitancia térmica media lineal en puentes térmicos (W/m K)', 0. , 2.1 , 0.8 )
    # selectbox
    InstCAL_Tipo = st.sidebar.selectbox('Tipo de instalación para calefacción', 
                   ("Sin definir", "Efecto Joule", "Caldera Estándar", 
                    "Bomba de calor", "Caldera Condensación", "Otros sistemas") )
    InstREF_Tipo = st.sidebar.selectbox('Tipo de instalación para refrigeración', 
                   ("Sin definir", "Maquina frigorífica", "Bomba de calor", "Otros sistemas") ) 
    Reducc_EPNoR = st.sidebar.slider('Reducc_EPNoR', 0. , 352.8 , 1.2 )

    data = {'NORMATIVA_CONSTRUCCIO': NORMATIVA_CONSTRUCCIO,
            'SUPERFICI_HAB': SUPERFICI_HAB,
            'COMPACITAT': COMPACITAT,
            'VENTILACIO_USO_RESIDENCIAL': VENTILACIO_USO_RESIDENCIAL,
            'VENTILACIO_INFILTRACIONS': VENTILACIO_INFILTRACIONS,
            'DEMANDA_ACS': DEMANDA_ACS,
            'OPACOS_Fach_sum': OPACOS_Fach_sum,
            'OPACOS_Fach_trans': OPACOS_Fach_trans,
            'OPACOS_Cubi_sum': OPACOS_Cubi_sum,
            'OPACOS_Cubi_trans': OPACOS_Cubi_trans,
            'HUECOS_sum': HUECOS_sum,
            'HUECOS_trans': HUECOS_trans,
            'HUECOS_fsol': HUECOS_fsol,
            'PUENTE_sum': PUENTE_sum,
            'PUENTE_trans': PUENTE_trans,
            'InstCAL_Tipo': InstCAL_Tipo,
            'InstREF_Tipo': InstREF_Tipo,
            'Reducc_EPNoR': Reducc_EPNoR }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Parámetros de entrada')
st.write(df.set_index([[0]]).T)

# load model
import os
folder = os.path.dirname(os.path.abspath(__file__))
name_model = os.path.join(folder, 'model')
load_final_model = load_model(name_model)

# Apply model to make predictions
new_prediction = predict_model(load_final_model, data=df.iloc[[-1]])
predict = (new_prediction['Label'].values[[-1]])
predict = np.round_(np.exp(predict),decimals=4)
#print('Consumo de Energía ESTIMADO: {} kWh/m²'.format(predict))

st.subheader('Predicción del Consumo de energía no renovable (kWh/m²)')
st.write('Consumo de Energía ESTIMADO: {} kWh/m²'.format(predict))
