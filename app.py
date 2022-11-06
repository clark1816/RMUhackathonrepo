import streamlit as st
import pandas
from sklearn import linear_model

#,,,,,,liquid level [m],rotate speed [rpm],,,outlet A wt%,outlet B wt%
option = st.sidebar.selectbox("Which Dashboard?", ('home', 'Linear Regression All Data','Linear Regression Correlation Data', 'Hybrid'),0)
if option == 'home':
    st.write('ROMOs Gold the Gold Standard in Adventuring')

if option == 'Linear Regression All Data':
    value_1 = st.number_input('inlet feed [kg/h]')

    value_2 = st.number_input('inlet polymer wt%')

    value_3 = st.number_input('inlet A wt%')

    value_4 = st.number_input('inlet B wt%')

    value_5 = st.number_input('inlet temp [degC]')

    value_6 = st.number_input('pressure [MPa]')

    value_7 = st.number_input('liquid level [m]')

    value_8 = st.number_input('rotate speed [rpm]')

    value_9 = st.number_input('bottom temp [degC]')

    
    if value_9:
        outletpolym = (-2.5620332e-04*value_1) + (7.8522271e-01*value_2) + (8.2543677e-01*value_3) + (3.5442629e+00*value_4) + (3.2703043e-03*value_5) + (-1.3718221e+02*value_6) + (-7.6586172e-02*value_7) + (2.4416444e-03*value_8) + (2.9435358e-03*value_9)
        st.write(f'Write answer: {outletpolym}')

if option == 'Linear Regression Correlation Data':
    st.write('option 2')
    df = pandas.read_csv("hackdata.csv")
    X = df[['inlet polymer wt%','pressure [MPa]', 'liquid level [m]']]
    #h = df.values.tolist()

    y1 = df['outlet B wt%']
    y2 = df['outlet A wt%']

    regr_b = linear_model.LinearRegression()
    regr_a = linear_model.LinearRegression()
    
    regr_b.fit(X, y1)
    regr_a.fit(X, y2)

    st.write(regr_b.coef_)
    st.write(regr_a.coef_)
    value_1 = st.number_input('inlet polymer wt%')

    value_2 = st.number_input('pressure [MPa]')

    value_3 = st.number_input('liquid level [m]')
    if value_3:
        predictedb = regr_b.predict([[value_1,value_2,value_3]])
        predicteda = regr_a.predict([[value_1,value_2,value_3]])
        st.write(f'the value of outlet B wt% is predicted to be {predictedb}')
        st.write(f'the value of outlet A wt% is predicted to be {predicteda}')
        total_poly = 100 - (predicteda + predictedb)
        st.write(f'total predicted polymer % is {total_poly}')
        












print('code completed')