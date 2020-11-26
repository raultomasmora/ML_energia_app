# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from pycaret.regression import load_model, predict_model
import matplotlib.pyplot as plt


# entorno streamlit
st.write("""
# **Energy Certification** with **Machine Learning**
\n*(en pruebas)*
\nEsta aplicación predice el consumo de energía (kWh/m²) a partir de miles de datos de certificados energéticos elaborados con el programa **[CE3X](https://www.efinova.es/CE3X)**.
\n**Modelo:** Esta herramienta se desarrolla mediante aprendizaje automático supervisado (*supervised machine learning*) con algoritmos de regresión. Se ha diseñado un modelo de conjunto (*enseble learning*) que combina tres algoritmos de aprendizaje distintos basados en *boosting*: CatBoost Regressor [*catboost*](https://catboost.ai/), Light Gradient Boosting Machine [*lightgbm*](https://lightgbm.readthedocs.io/) y Gradient Boosting Regressor [*gbr*](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html).
\n**Datos:** Se utilizan más de 300.000 datos de certificados energéticos de viviendas individuales de la provincia de Barcelona (ubicados en zona climática C2), procedentes del [Instituto Catalán de Energía](http://icaen.gencat.cat/es/inici/).
\n**Precisión:** El modelo de conjunto se ha probado en un set de datos de entrenamiento obteniéndose un R2 de 0.888, y en el set de prueba un R2 de 0.732. Para datos nuevos no utilizados en el modelo se ha obtenido un R2 de 0.790, lo que indica que el modelo generaliza correctamente. 
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
predict = np.round_(np.exp(predict),decimals=1)

st.subheader('Predicción del Consumo de energía no renovable')
st.write('Consumo de Energía ESTIMADO: **{}** kWh/m²'.format(predict))

# Para imprimir el gráfico de las escalas

def plot_escala_letras(results, category_names, line_value):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    line_value = line_value
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn_r')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(10, 1))
    ax.invert_yaxis()
    ax.xaxis.set_visible(True)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, category_names)):
            ax.text(x, y, str(category_names[i]), ha='center', va='center',
                    color=text_color)
    #ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
    #          loc='lower left', fontsize='small')
    ax.axvline(x=line_value, color='black', label='Estimado', linestyle='--', linewidth=2)
    st.pyplot(fig) 
    #return fig, ax

category_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
results = { 'Zona C2': [23.4, 14.6, 20.8, 31.7, 94.0, 24.0, 291.5] }
line_value = predict

plot_escala_letras(results, category_names, line_value)

# fin gráfico escalas
