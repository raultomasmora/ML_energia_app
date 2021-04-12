# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from pycaret.regression import load_model, predict_model
import matplotlib.pyplot as plt
import os

# entorno streamlit

hide_st_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)


try:
    folder = os.path.dirname(os.path.abspath(__file__))
    name_model = os.path.join(folder, 'model')
    final_model = load_model(name_model)
except: 
    print("Se necesita un modelo entrenado")

st.title('Certificación energética con Machine Learning')
st.title('\n\n')
st.error('Entorno web en pruebas... (actualización 2021-04-11)')
with st.beta_expander("Información:", expanded=True):
    st.success('El proyecto ha sido elaborado por el investigador [Raúl Mora-García](https://publons.com/researcher/1717710/raul-tomas-mora-garcia/) [:email:](mailto:rtmg@ua.es) en colaboración con [Grupo Valero](https://www.grupovalero.com/) durante el año 2020. Subvención AEST/2019/005 del Programa para la promoción de la investigación científica, el desarrollo tecnológico y la innovación en la Comunitat Valenciana (Anexo VII) [DOGV nº8355](http://www.dogv.gva.es/datos/2018/08/06/pdf/2018_7758.pdf).')
    st.info('\n\nEsta aplicación predice el consumo de energía (kWh/m²año) a partir de miles de datos de certificados energéticos elaborados con el programa **[CE3X](https://www.efinova.es/CE3X)**. Después se evalúa la posible reducción del consumo de energía al mejorar el aislamiento de la envolvente.'
    '\n\n**Modelo:** Esta herramienta se desarrolla mediante aprendizaje automático supervisado (*supervised machine learning*) con algoritmos de regresión. Se ha diseñado un modelo de conjunto (*enseble learning*) que combina tres algoritmos de aprendizaje distintos basados en *boosting*: CatBoost Regressor [*catboost*](https://catboost.ai/), Light Gradient Boosting Machine [*lightgbm*](https://lightgbm.readthedocs.io/) y Gradient Boosting Regressor [*gbr*](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html).'
    '\n\n**Datos:** Se utilizan más de 10.000 datos de certificados energéticos de viviendas individuales de la provincia de Barcelona (ubicados en zona climática C2), procedentes del [Instituto Catalán de Energía](http://icaen.gencat.cat/es/inici/).'
    '\n\n**Precisión:** El modelo de conjunto se ha probado en un set de datos de entrenamiento obteniéndose un R2 de 0.888, y en el set de prueba un R2 de 0.732. Para datos nuevos no utilizados en el modelo se ha obtenido un R2 de 0.790, lo que indica que el modelo generaliza correctamente. ')

st.sidebar.image('logo.png')
st.sidebar.header('Parámetros de entrada')

def user_input_features():
    NORMATIVA_CONSTRUCCIO = st.sidebar.selectbox('Normativa de construcción', ("Ant_NBECT79", "NBECT79", "CTE2006"), 
                                                help=('Se definen tres periodos normativos'  '\n\n según el año de construcción'))
    SUPERFICI_HAB = st.sidebar.slider('Superficie habitable (m²)', 20. , 250. , 70. , 
                                      help=('Superficie útil de la vivienda'))
    COMPACITAT = st.sidebar.slider('Compacidad (m³/m²)', 0.5 , 18. , 3.4 , 
                                   help=('Compacidad calculada conforme al CTE'  '\n\n (volumen/área de la envolvente)'))
    VENTILACIO_USO_RESIDENCIAL = st.sidebar.slider('Ventilación uso residencial (renovaciones/hora)', 0.3 , 1. , 0.65 ,
                                                  help=('Renovaciones hora de aire exterior para'  '\n\n uso residencial establecido en el CTE'))
    VENTILACIO_INFILTRACIONS = st.sidebar.slider('Ventilación por infiltraciones (renovaciones/hora)', 0.4 , 1.3 , 0.63 ,
                                                help=('Renovaciones hora de aire exterior'  '\n\n debido a infiltraciones'))
    DEMANDA_ACS = st.sidebar.slider('Demanda de ACS (litros/día)', 0. , 1122. , 71.5 ,
                                   help=('Demanda de agua caliente sanitaria'  '\n\n conforme al CTE'))
    OPACOS_Fach_sum = st.sidebar.slider('Suma de superficies en fachada (m²)', 7. , 250. , 45. ,
                                       help=('Total de las superficies de fachada'  '\n\n en contacto con el exterior'))
    OPACOS_Fach_trans = st.sidebar.slider('Transmitancia térmica media en fachadas (W/m²∙K)', 0.2 , 3.2 , 1.7 ,
                                         help=('Valor medio de la transmitancia térmica'  '\n\n de los cerramientos opacos'))
    OPACOS_Cubi_sum = st.sidebar.slider('Suma de superficies en cubierta (m²)', 0. , 1224.3 , 11.7 ,
                                       help=('Total de las superficies de cubierta'  '\n\n en contacto con el exterior'))
    OPACOS_Cubi_trans = st.sidebar.slider('Transmitancia térmica media en cubiertas (W/m²∙K)', 0. , 5.7 , 0.4 ,
                                         help=('Valor medio de la transmitancia térmica'  '\n\n de las superficies de cubierta'))
    HUECOS_sum = st.sidebar.slider('Suma de superficies en huecos (m²)', 1. , 157.9 , 11.8 ,
                                  help=('Total de las superficies de los huecos'  '\n\n en contacto con el exterior '))
    HUECOS_trans = st.sidebar.slider('Transmitancia térmica media en huecos (W/m²∙K)', 0.6 , 7. , 4.4 ,
                                    help=('Valor medio de la transmitancia térmica'  '\n\n de las superficies de los huecos'))
    HUECOS_fsol = st.sidebar.slider('Factor solar promedio en huecos (g)', 0. , 1. , 0.5 ,
                                   help=('Valor medio del factor solar'  '\n\n de las superficies acristaladas'))
    PUENTE_sum = st.sidebar.slider('Suma de longitudes con puentes térmicos (metros)', 0. , 1100.5 , 70.2 ,
                                  help=('Total de las longitudes de los puentes térmicos'  '\n\n en contacto con el exterior'))
    PUENTE_trans = st.sidebar.slider('Transmitancia térmica media lineal en puentes térmicos (W/m∙K)', 0. , 2.1 , 0.8 ,
                                    help=('Valor medio de la transmitancia'  '\n\n de los puentes térmicos'))
    InstCAL_Tipo = st.sidebar.selectbox('Tipo de instalación para calefacción', 
                   ("Sin definir", "Efecto Joule", "Caldera Estándar", 
                    "Bomba de calor", "Caldera Condensación", "Otros sistemas") ,
                    help=('Se definen 6 posibles casos de'  '\n\n instalaciones de calefacción'))
    InstREF_Tipo = st.sidebar.selectbox('Tipo de instalación para refrigeración', 
                   ("Sin definir", "Maquina frigorífica", "Bomba de calor", "Otros sistemas") ,
                    help=('Se definen 4 posibles casos de'  '\n\n instalaciones de refrigeración')) 
    Reducc_EPNoR = st.sidebar.slider('Reducc_EPNoR (kWh/m²∙año)', 0. , 352.8 , 0. ,
                                    help=('Producción de energía renovable'))

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

#st.subheader('Parámetros de entrada')
with st.beta_expander("Parámetros de entrada:", expanded=False):
    if st.checkbox('Visualizar los parámetros de entrada'):
        st.write(df.set_index([[0]]).T)
    st.info('\n\nDescripción de los parámetros utilizados en el modelo:'
            '\n\n- Normativa de construcción. *Se definen tres periodos normativos según el año de construcción: Anterior a la norma NBE-CT79, conforme a la norma NBE-CT79 o conforme a la norma CTE2016.*'
            '\n\n- Superficie habitable (m²). *Superficie útil de la vivienda (en m²).*'
            '\n\n- Compacidad (m³/m²). *Compacidad calculada conforme al CTE (volumen/área de la envolvente).*'
            '\n\n- Ventilación uso residencial (renovaciones/hora). *Renovaciones hora de aire exterior para uso residencial establecido en el CTE (en renovaciones/hora).*'
            '\n\n- Ventilación por infiltraciones (renovaciones/hora). *Renovaciones hora de aire exterior debido a infiltraciones por la carpintería (en renovaciones/hora).*'
            '\n\n- Demanda de ACS (litros/día). *Demanda de agua caliente sanitaria conforme al CTE (en litros/día).*'
            '\n\n- Suma de superficies en fachada (m²). *Total de las superficies de fachada en contacto con el exterior y que forman parte de la envolvente térmica (en m²).*'
            '\n\n- Transmitancia térmica media en fachadas (W/m²K). *Valor medio de la transmitancia térmica de los cerramientos opacos que forman parte de la envolvente térmica (en W/m²K).*'
            '\n\n- Suma de superficies en cubierta (m²). *Total de las superficies de cubierta en contacto con el exterior y que forman parte de la envolvente térmica (en m²).*'
            '\n\n- Transmitancia térmica media en cubiertas (W/m²K). *Valor medio de la transmitancia térmica de las superficies de cubierta que forman parte de la envolvente térmica (en W/m²K).*'
            '\n\n- Suma de superficies en huecos (m²). *Total de las superficies de los huecos en contacto con el exterior y que forman parte de la envolvente térmica (en m²).*'
            '\n\n- Transmitancia térmica media en huecos (W/m²K). *Valor medio de la transmitancia térmica de las superficies de los huecos que forman parte de la envolvente térmica (en W/m²K).*'
            '\n\n- Factor solar promedio en huecos (g). *Valor medio del factor solar de las superficies acristaladas que forman parte de la envolvente térmica (g es un valor adimensional entre 0 y 1).*'
            '\n\n- Suma de longitudes con puentes térmicos (metros). *Total de las longitudes de los puentes térmicos en contacto con el exterior y que forman parte de la envolvente térmica (en metros).*'
            '\n\n- Transmitancia térmica media lineal en puentes térmicos (W/mK). *Valor medio de la transmitancia térmica de los puentes térmicos que forman parte de la envolvente térmica (en W/mK).*'
            '\n\n- Tipo de instalación para calefacción. *Se definen 6 posibles casos de instalaciones de calefacción: Sin definir, Efecto Joule, Caldera Estándar, Bomba de calor, Caldera de condensación, y Otros sistemas.*'
            '\n\n- Tipo de instalación para refrigeración. *Se definen 4 posibles casos de instalaciones de refrigeración: Sin definir, Maquina frigorífica, Bomba de calor, y Otros sistemas.*'
            '\n\n- Reducc_EPNoR (kWh/m²∙año). *Producción de energía renovable (en kWh/m²∙año).*'
            )

new_prediction = predict_model(final_model, data=df.iloc[[-1]])
predict = (new_prediction.iloc[0]['Label'])
predict = np.round_(np.exp(predict),decimals=1)

st.subheader('Predicción del Consumo de energía no renovable')
st.write('Consumo de Energía ESTIMADO:  **{:.1f}**  kWh/m²año'.format(predict))

def plot_escala_letras(results, category_names, line_value):
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
    ax.axvline(x=line_value, color='black', label='Estimado', linestyle='--', linewidth=2)
    st.pyplot(fig) 

category_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
results = { 'Zona C2': [23.4, 14.6, 20.8, 31.7, 94.0, 24.0, 291.5] }

plot_escala_letras(results, category_names, predict)

st.subheader('Mejora de la envolvente')

df_fig = pd.DataFrame(df.iloc[[-1]])

rangos_transm = np.arange(.2,3.3,.02)
for i in range( 1,  len(rangos_transm)  ):
    df_fig.loc[i]  = df_fig.values[0]
df_fig['OPACOS_Fach_trans'] = rangos_transm
df_fig = pd.concat([df_fig, df_fig, df_fig], axis=0).reset_index(drop=True)
df_fig['NORMATIVA_CONSTRUCCIO'] = ["Ant_NBECT79"] * len(rangos_transm) + ["NBECT79"] * len(rangos_transm) + ["CTE2006"] * len(rangos_transm)

predict_transm = predict_model(final_model, data=df_fig)
predict_transm["Label"] = np.round_(np.exp(predict_transm["Label"]),decimals=1)

import plotly.express as px
fig_transm = px.line(predict_transm, x ='OPACOS_Fach_trans', y='Label', color='NORMATIVA_CONSTRUCCIO', #title='Título',
              labels={"OPACOS_Fach_trans": "Transmitancia térmica media en fachadas (W/m²K)", 
                      "Label": "Consumo de energía estimado<br>(kWh/m²año)",
                      "NORMATIVA_CONSTRUCCIO": "Normativa de construcción"  })
fig_transm.update_xaxes(range=[0., 3.4]) # ampliamos el ancho del eje x
fig_transm.add_shape( # add a vertical line
    type="line", line_color="black", line_width=2, opacity=1, line_dash="dash",
    x0=df.iloc[0]['OPACOS_Fach_trans'], x1=df.iloc[0]['OPACOS_Fach_trans'], xref="x", y0=0, y1=1, yref="paper"
    )

if df.iloc[0]['NORMATIVA_CONSTRUCCIO'] == 'CTE2006':
    fig_transm.add_shape( 
        type="rect", x0=0.60, x1=0.86, xref="x", y0=0, y1=1, yref="paper",
        line_width=0, fillcolor=px.colors.qualitative.Plotly[2], opacity=0.2 )
    fig_transm.add_annotation(text="Transmitancias U <br>habituales para <br>CTE2016",
                      xref="x", yref="paper",
                      x=0.73, y=1, showarrow=False)
if df.iloc[0]['NORMATIVA_CONSTRUCCIO'] == 'NBECT79':
    fig_transm.add_shape( 
        type="rect", x0=0.66, x1=1.80, xref="x", y0=0, y1=1, yref="paper",
        line_width=0, fillcolor=px.colors.qualitative.Plotly[1], opacity=0.2 )
    fig_transm.add_annotation(text="Transmitancias U <br>habituales para <br>NBECT79",
                      xref="x", yref="paper",
                      x=1.23, y=1, showarrow=False)
if df.iloc[0]['NORMATIVA_CONSTRUCCIO'] == 'Ant_NBECT79':
    fig_transm.add_shape( 
        type="rect", x0=1.69, x1=2.38, xref="x", y0=0, y1=1, yref="paper",
        line_width=0, fillcolor=px.colors.qualitative.Plotly[0], opacity=0.2 )
    fig_transm.add_annotation(text="Transmitancias U <br>habituales para <br>Ant_NBECT79",
                      xref="x", yref="paper",
                      x=2.04, y=1, showarrow=False)

with st.beta_expander("Información:", expanded=True):
    st.write("La siguiente tabla muestra una comparativa de la relación existente entre la transmitancia de la fachada y la normativa de diseño. En el eje horizontal se representa la transmitancia térmica media en fachadas (en W/m²K), en el eje vertical el consumo de energía estimado (en kWh/m²año), y en colores de detallan las estimaciones según las tres normas de diseño existentes. \n\nEl resto de parámetros utilizados se mantienen constantes (céteris páribus), estando definidos por el usuario en el panel lateral izquierdo (Parámetros de entrada). Independientemente de la norma utilizada como parámetro de entrada, el gráfico muestra el efecto de las tres normas en el consumo de energía. Al cambiar la norma de aplicación en el panel lateral, se mostrará en la figura una franja de valores habituales de transmitancias en fachadas según la norma correspondiente.")
    st.info("Al disminuir la transmitancia térmica del cerramiento también disminuye el consumo de energía, pero este descenso no es lineal. \n\nPuede modificar el parámetro de 'Transmitancia térmica media en fachadas' del panel lateral izquierdo para cuantificar el ahorro de energía en una mejora térmica de la envolvente de fachada.")

st.plotly_chart(fig_transm)

