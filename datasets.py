from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes
import pandas as pd

"""
IRIS DATA SET
"""
# save load_iris() sklearn dataset to iris
# if you'd like to check dataset type use: type(load_iris())
# if you'd like to view list of attributes use: dir(load_iris())
# iris = load_iris()
# print(type(iris.data))
# print(type(iris.target))
# print(type(iris.feature_names))
# print(iris.feature_names)
# print(type(iris.target_names))
# print(iris.target_names)
# print(dir(iris))
#
# x = iris.data
# y = iris.target
# x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=0)

# df1_iris = pd.DataFrame(data= np.column_stack((iris.data, iris.target)), columns=iris.feature_names+["target"])
# print(df1_iris)

"""
DIABETES SET
"""
diabetes = load_diabetes()
print(type(diabetes))
print(dir(diabetes))
print(type(diabetes.data))
print(diabetes.feature_names)

df_diabetes = pd.DataFrame(data=np.column_stack((diabetes.data,diabetes.target)), columns=diabetes.feature_names+["target"])
print(df_diabetes)