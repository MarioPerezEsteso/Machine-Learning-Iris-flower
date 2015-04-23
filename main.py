#!/usr/bin/env python

# Dataset: https://archive.ics.uci.edu/ml/datasets/Iris

from pandas import read_csv
from sklearn import cross_validation

# # Columns
# 0. sepal length in cm
# 1. sepal width in cm
# 2. petal length in cm
# 3. petal width in cm
# 4. Name

iris = read_csv('data/iris.data',
                sep = ',',
                names = ['SepalLength',
                         'SepalWidth',
                         'PetalLength',
                         'PetalWidth',
                         'Name'])

iris_data = iris.drop('Name', 1)
iris_target = iris['Name']

data_train, data_test, target_train, target_test = cross_validation.train_test_split(iris_data, iris_target, test_size = 0.4, random_state=0)

