import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from prediction import get_prediction, label_encoder

model = joblib.load(r'Model/model.joblib')

st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚧", layout="wide")


options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_type_of_collision = ['Collision with roadside-parked vehicles',
       'Vehicle with vehicle collision',
       'Collision with roadside objects', 'Collision with animals',
       'Other', 'Rollover', 'Fall from vehicles',
       'Collision with pedestrians', 'With Train']
options_driving_exp = ['1-2yr' ,'Above 10yr', '5-10yr', '2-5yr' ,"nan" ,'No Licence' ,'Below 1yr','unknown']
options_road_surfacce = ['Dry' ,'Wet or damp' ,'Snow' ,'Flood over 3cm. deep'] 
options_types_of_junction = ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other', 'Unknown', 'T Shape',
 'X Shape' ] 
options_light_conditions = ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting',
 'Darkness - lights unlit'] 

features=['Minutes', 'Day_of_week', 'Number_of_vehicles_involved', 'Number_of_casualties', 'Light_conditions', 'Driving_experience', 'Road_surface_conditions', 'Types_of_Junction', 'Hour', 'Type_of_collision']


st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        
        Hour = st.slider("Pickup Hour: ", 0, 23, value=0, format="%d")
        Minutes = st.slider("Pickup Minutes: ", 0, 59, value=0, format="%d")
        Day_of_week = st.selectbox("Select Day of the Week: ", options=options_day)
        Number_of_casualties = st.slider("Number of Casualities: ", 0, 7, value=0, format="%d")
        Number_of_vehicles_involved = st.slider("Number of Vehicles Involved: ", 0, 7, value=0, format="%d")
        Light_conditions = st.selectbox("Select Lighting Conditions: ", options=options_light_conditions)
        Driving_experience = st.selectbox("Select Driving Experience of the Driver: ", options=options_driving_exp)
        Road_surface_conditions = st.selectbox("Select Surface of the Road: ", options=options_road_surfacce)
        Types_of_Junction = st.selectbox("Select the type of Junction: ", options=options_types_of_junction)
        Type_of_collision = st.selectbox("Select the type of Collision: ", options=options_type_of_collision)
        
        
        submit = st.form_submit_button("Predict")


    if submit:
        Day_of_week = label_encoder(Day_of_week, options_day)
        Driving_experience = label_encoder(Driving_experience,options_driving_exp)
        Road_surface_conditions = label_encoder( Road_surface_conditions,options_road_surfacce)
        Types_of_Junction = label_encoder(Types_of_Junction,options_types_of_junction)
        Type_of_collision = label_encoder(Type_of_collision,options_type_of_collision)
        Light_conditions = label_encoder(Light_conditions,options_light_conditions)

        data = np.array([Minutes, Day_of_week, Number_of_vehicles_involved, Number_of_casualties, Light_conditions, Driving_experience, Road_surface_conditions, Types_of_Junction, Hour, Type_of_collision]).reshape(1,-1)

        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted severity is:  {pred[0]}")

if __name__ == '__main__':
    main()
