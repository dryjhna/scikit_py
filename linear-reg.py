import numpy as np
import matplotlib.pyplot as plt
import seaborn;
from sklearn.linear_model import LinearRegression

import pylab as pl

seaborn.set()

#Create data
import numpy as np
np.random.seed(444)
X = np.random.random(size=(100,1))
y = 3* X.squeeze() + 2 + np.random.randn(100)

plt.plot(X.squeeze(),y ,'o');

#Fit the model

model = LinearRegression()
model.fit(X,y)
X_fit = np.linspace(0,1,100)[:,np.newaxis]
y_fit = model.predict(X_fit)

plt.plot(X.squeeze(),y,'o')
plt.plot(X_fit.squeeze(),y_fit)
