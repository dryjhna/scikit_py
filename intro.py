%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.linear_model import LinearRegression
from scipy import stats
import pylab as pl

seaborn.set()

from IPython.display import Image
Image("http://scikit-learn.org/dev/_static/ml_map.png", width=800)


from sklearn.datasets import load_iris
iris = load_iris()

n_samples, n_features = iris.data.shape
print(iris.keys())
print((n_samples, n_features))
print(iris.data.shape)
print(iris.target.shape)
print(iris.target_names)
print(iris.feature_names)

#'sepal width (cm)'
x_index = 1

#'petal width (cm)'
y_index = 2

formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

plt.scatter(iris.data[:, x_index], iris.data[:, y_index],
            c=iris.target, cmap=plt.cm.get_cmap('RdYlBu', 3))
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.clim(-0.5, 2.5)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index]);


from sklearn import neighbors,datasets

iris = datasets.load_iris()
X,y = iris.data, iris.target

knn = neighbors.KNeighborsClassifier(n_neighbors=5,weights= 'uniform')

knn.fit(X,y)

X_pred = [3,5,4,2]
result = knn.predict([X_pred,])

print(iris.target_names[result])
print(iris.target_names)
print(knn.predict_proba([X_pred,]))

from fig_code import plot_iris_knn

plot_iris_knn()


