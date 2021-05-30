# LinearRegression_Cars_Mileage
#Linear Regression Model for Cars_mileage Dataset

# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# READ AND LOAD THE DATA
cars=pd.read_csv('auto-mpg.csv')
cars.head()
cars.tail()
cars.shape

#Drop car name column(if it is not useful)
cars = cars.drop('car name', axis=1)
cars.head()

# FOR ORIGIN column, REPLACE NUMBERS
# 1: America.2:Europe, 3:Asia
cars['origin']=cars['origin'].replace({1:'America',2:'Europe',3:'Asia'})
cars.head()

# ONE HOT ENCODING
cars=pd.get_dummies(cars,columns=['origin'])
cars.head()

# FIND NULL DATA CELLS if any
cars.isnull()

# CHECK THE MISSING VALUES
a = cars[cars['hp']=='?']
a.shape

# REPLACE ? BY NAN
cars=cars.replace('?',np.nan)
cars

# REPLACE NAN BY MEDIAN
medianFiller= lambda x:x.fillna(x.median())
cars=cars.apply(medianFiller,axis=0)
cars['hp']=cars['hp'].astype('float64')
cars.head()

medianFiller = lambda x: x.fillna(x.median())
cars = cars.apply(medianFiller,axis=0)

cars['hp'] = cars['hp'].astype('float64')

# PAIRPLOT
sns.pairplot(cars,diag_kind='kde')
# To plot limited columns in pairplot(for better visualization)

cars1=cars.iloc[:,0:2]
sns.pairplot(cars1,diag_kind='kde')

# CORRELATION
correlation=cars.corr()
print(correlation)

# SPLIT INTO TRAINING AND TESTING

# X contains all variables other than target variable
X=cars.drop(['mpg'],axis=1)  
# Set the target variable (i.e. y)
y=cars[['mpg']]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=1)

# FIT LINEAR REGRESSION MODEL
reg_model = LinearRegression()
reg_model.fit(X_train,y_train)

# Finding coeff of correlation for all columns of X

for idx,col_name in enumerate(X_train.columns):
    print('The Coefficient for {} is {}'.format(col_name, reg_model.coef_[0][idx]))

#CHECK THE SCORES
reg_model.score(X_test,y_test)
reg_model.score(X_train,y_train)

# TEST_SCORE = 84.33%, TRAIN_SCORE = 81.41%
# So, the score of training model is less than testing, but is closer to it
# Hence our model can be considered as a GOOD MODEL (fitting model)
