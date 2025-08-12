import pandas as pd
import numpy
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load data
df = pd.read_csv("house_prices.csv")

# make x and y for train and test
df = df[['LotArea', 'YearBuilt', 'FullBath', 'BedroomAbvGr', 'GarageCars',
       'SalePrice']]

x  = df.drop('SalePrice',axis=1)

y = df


# apply train and test split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state= 42 )

# model train

model = LinearRegression()
model.fit(x_train,y_train)


# save model
with open("house_price_model.pkl","wb") as f:
    pickle.dump(model,f)

print("Model trained and saved successfully")
