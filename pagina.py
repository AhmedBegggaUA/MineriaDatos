# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 11:11:21 2023

@author: Sergio
"""

import streamlit as st
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
# Fix the random seed
np.random.seed(1234)
from urllib.error import URLError
import plotly.graph_objects as go
#sys.path.append('C:/Users/Sergio/Documents/valencia-ia4covid-xprize-master/valencia-ia4covid-xprize-master')

st.set_page_config(layout = 'wide')

@st.cache
def get_UN_data():
    #Read the Macroinvertebrados.xlsx file
    df = pd.read_excel('Macroinvertebrados.xlsx', sheet_name='Prepared_macros')
    df.drop(df[df['dive_date'].isnull()].index, inplace = True)

    return df

try:
    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        st.write("")
        
    #with col2:
    #    st.image(logo)
    
    with col3:
       st.write("")
    
    
    df = get_UN_data()
####################################################
####################################################
    
    
    st.markdown('# Data Visualization')
    st.markdown('Here you can see all the animals')
    
    cols = st.columns((1,5))
    with cols[0]:
        epochs_list  = ['Caliente','Frio']
        epoca = st.selectbox(
            "Choose the epoch ",epochs_list
        )
        refuge_level_list = ['Extractive use','Sanctuary','all']
        
        refugio = st.selectbox(
            "Choose the refuge lvl ",refuge_level_list
        )
    with cols[1]:
        islas_list  = df.Island.unique()
        # Añadimos la opción de todas las islas
        islas_list = np.append(islas_list,'all')
        isla = st.selectbox(
            "Choose the island ",islas_list
        )
        dive_months_list = df.dive_month.unique()
        # Añadimos la opción de todos los meses
        dive_months_list = np.append(dive_months_list,'all')
        dive = st.selectbox(
            "Choose the dive month ",dive_months_list
        )
        if epoca != 'Caliente':
            import plotly.express as px
            df_mod_epoca_calida = df[df['epoca'] != 'Caliente']
            if refugio != 'all':
                df_mod_epoca_calida = df[df['Refuge_Level'] == refugio]
                df_mod_epoca_calida = df_mod_epoca_calida[df_mod_epoca_calida['Island'] == isla]
                df_mod_epoca_calida = df_mod_epoca_calida[df_mod_epoca_calida['dive_month'] == dive]
            fig = px.scatter_mapbox(df_mod_epoca_calida, 
                                    lat="Latitude",
                                    lon="Longitude",
                                    hover_name="Transect.code",
                                    hover_data=["CommonNameSpanish", "Transect.code", "Latitude", "Longitude"],
                                    color="CommonNameSpanish",
                                    zoom=8, 
                                    height=800,
                                    width=800)

            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            # Make the plot BIGGER
            fig.update_layout(
                autosize=False,
                width=1000,
                height=1000,
                )
        else: 
            import plotly.express as px
            df_mod_epoca_calida = df[df['epoca'] == 'Caliente']
            if refugio != 'all':
                df_mod_epoca_calida = df[df['Refuge_Level'] == refugio]
            if isla != 'all':
                df_mod_epoca_calida = df_mod_epoca_calida[df_mod_epoca_calida['Island'] == isla]
            if dive != 'all':
                df_mod_epoca_calida = df_mod_epoca_calida[df_mod_epoca_calida['dive_month'] == dive]
            fig = px.scatter_mapbox(df_mod_epoca_calida, 
                                    lat="Latitude",
                                    lon="Longitude",
                                    hover_name="Transect.code",
                                    hover_data=["CommonNameSpanish", "Transect.code", "Latitude", "Longitude"],
                                    color="CommonNameSpanish",
                                    zoom=8, 
                                    height=800,
                                    width=800)

            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            # Make the plot BIGGER
            fig.update_layout(
                autosize=False,
                width=1000,
                height=1000,
                )
    st.plotly_chart(figure_or_data=fig,use_container_width=True)
    
    
except URLError as e:
    st.error(
        """
            
        **This demo requires internet access.**

        Connection error: %s
        """

        % e.reason
    )