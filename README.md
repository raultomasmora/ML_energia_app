# ML_energia_app

Esta aplicación predice el consumo de energía (kWh/m²) a partir de miles de datos de certificados energéticos elaborados con el programa CE3X.

En el **[siguiente enlace](https://share.streamlit.io/raultomasmora/ml_energia_app/main/ML_energy_app.py)** se puede ver la implementación web del proyecto en Streamlit.

## Sobre el proyecto
Proyecto elaborado por rtmg@ua.es en colaboración con [Grupo Valero](https://www.grupovalero.com/) durante el año 2020. Subvención AEST/2019/005 del Programa para la promoción de la investigación científica, el desarrollo tecnológico y la innovación en la Comunitat Valenciana (Anexo VII) [DOGV nº8355](http://www.dogv.gva.es/datos/2018/08/06/pdf/2018_7758.pdf).

## Descripción del proyecto
Esta aplicación predice el consumo de energía (kWh/m²año) a partir de miles de datos de certificados energéticos elaborados con el programa CE3X. Después se evalúa la posible reducción del consumo de energía al mejorar el aislamiento de la envolvente.

**Modelo:** Esta herramienta se desarrolla mediante aprendizaje automático supervisado (supervised machine learning) con algoritmos de regresión. Se ha diseñado un modelo de conjunto (enseble learning) que combina tres algoritmos de aprendizaje distintos basados en boosting: CatBoost Regressor catboost, Light Gradient Boosting Machine lightgbm y Gradient Boosting Regressor gbr.

**Datos:** Se utilizan más de 300.000 datos de certificados energéticos de viviendas individuales de la provincia de Barcelona (ubicados en zona climática C2), procedentes del Instituto Catalán de Energía.

**Precisión:** El modelo de conjunto se ha probado en un set de datos de entrenamiento obteniéndose un R2 de 0.888, y en el set de prueba un R2 de 0.732. Para datos nuevos no utilizados en el modelo se ha obtenido un R2 de 0.790, lo que indica que el modelo generaliza correctamente.

## Parámetros necesarios en el modelo
- *Normativa de construcción*. Se definen tres periodos normativos según el año de construcción: Anterior a la norma NBE-CT79, conforme a la norma NBE-CT79 o conforme a la norma CTE2016.
Superficie habitable (m²). Superficie útil de la vivienda (en m²).
- *Compacidad (m³/m²)*. Compacidad conforme a CTE (volumen/área de la envolvente).

## Screenshots
![app](images/app_energia1.jpg)
