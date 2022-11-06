import pandas
from sklearn import linear_model

df = pandas.read_csv("hackdata.csv")
X = df[['inlet polymer wt%','pressure [MPa]', 'liquid level [m]']]
h = df.values.tolist()

y = df['outlet B wt%']

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_)
for x in X:
    predictedout = regr.predict([[x[1],x[5], x[6]]])
    print(predictedout)



