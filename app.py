import streamlit as st
import pandas
from sklearn import linear_model

#,,,,,,liquid level [m],rotate speed [rpm],,,outlet A wt%,outlet B wt%
option = st.sidebar.selectbox("Which Dashboard?", ('home','Linear Regression Correlation Data', 'Linear Regression All Data', 'Hybrid'),0)
if option == 'home':
    st.header('Meet the Team')    
    st.image('RomosGold.jpg')
    st.write('ROMOs Gold the Gold Standard in Adventuring')
    st.write('Software Team: Harry and Clark')
    st.write('Biomedical / Chemistry Team: Amanda and Leia')

if option == 'Linear Regression All Data':
    st.header('Linear Regression All Data')
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
        
        a_value = 100 - Poly_value - b_value
        
        st.write(f'Total Polymer %: {Poly_value}')
        st.write(f'Polymer A%: {a_value}')
        st.write(f'Polymer B%: {b_value}')
        
        

if option == 'Linear Regression Correlation Data':
    
    st.header('Linear Regression Correlating Data')
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
    
    #change these values 
    #density = a*C(degrees)+b
    value_10 = -1.2707*value_5+925.11
    value_11 = -1.2707*value_9+925.11
    value_12 = -0.7598*value_5+1020.6
    value_13 = -0.7598*value_9+1020.6
    value_14 = -0.7092*value_5+1248.7
    value_15 = -0.7092*value_9+1248.7

    
    if value_9:
        outletb = (3.0346154e-04*value_1) + (4.9141365e-01*value_2) + (4.4068372e-01*value_3) + (-2.8392928*value_4) + ( -2.3127261e-01*value_5) + (5.1315887e+01*value_6) + (-2.2837190e-02*value_7) + ( -3.3287513e-03*value_8) + ( 4.0363985e-01*value_9) + 0.48465901613235474 + (value_10*-5.6526303e-01)+ (value_11*7.3614556e-01)+(value_12*5.6254750e-01)+(value_13*-1.7261361e-01)+(value_14*8.6331263e-02)+(value_15*-5.6185305e-01)
        
        outletpoly = (-2.9273261e-04*value_1) + (5.8944124e-01*value_2) + (6.1098981e-01*value_3) + (2.3368397*value_4) + ( 1.1619869*value_5) + (-6.5066833e+01*value_6) + (1.0090161e-02*value_7) + ( 3.0155266e-03*value_8) + ( -6.9166225e-01*value_9) + -0.2506895959377289 + (value_10*-1.0483562)+ (value_11*-6.1003977e-01)+(value_12*5.1095623e-01)+(value_13*1.5999396e-01)+(value_14*-7.9202187e-01)+(value_15*-5.6561343e-02)
        
        outleta = 100 - outletpoly - outletb
        
        st.write(f'outlet polymer wt%: {outletpoly}')
        st.write(f'outlet B wt%: {outletb}')
        st.write(f'outlet A wt%: {outleta}')
        
        
        
        
        











print('code completed')


