import streamlit as st
import pandas
from sklearn import linear_model

#,,,,,,liquid level [m],rotate speed [rpm],,,outlet A wt%,outlet B wt%
option = st.sidebar.selectbox("Which Dashboard?", ('home','Linear Regression Correlation Data', 'Linear Regression All Data', 'Hybrid'),0)
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
        Poly_value = (-2.45598028e-04*value_1) + (7.99089074e-01*value_2) + (8.35692048e-01*value_3) + (3.33927655*value_4) + (3.24872392e-03*value_5) + (-1.59723953e+02*value_6) + (-1.03581265e-01*value_7) + (2.27641081e-03
*value_8) + (3.19753168e-03*value_9) +-1.3887262344360352
        
        b_value = (2.5583830e-04*value_1) + (1.9988307e-01*value_2) + (1.3920839e-01*value_3) + (-3.6337156*value_4) + ( -1.5648461e-03*value_5) + (1.4676350e+02*value_6) + (9.1752745e-02*value_7) + (-2.5921480e-03
*value_8) + (-3.2897484e-03*value_9) +2.2742693424224854
        
        a_value = Poly_value - b_value
        
        st.write(f'Total Polymer %: {Poly_value}')
        st.write(f'Polymer A% %: {a_value}')
        st.write(f'Polymer A% %: {b_value}')
        
        

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
        

if option == 'Hybrid':
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











print('code completed')


