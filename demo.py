import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from k_means import Kmeans

data = pd.read_csv(r"C:\Users\pszpszpsz\Desktop\Iris.csv")
iris_types = ['Iris-setosa','Iris-versicolor','Iris-virginica']

x_axis = 'PetalLengthCm'
y_axis = 'PetalWidthCm'

plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['Species']==iris_type],data[y_axis][data['Species']==iris_type],label = iris_type)
plt.title('lable known')
plt.legend()

plt.subplot(1,2,2)
plt.scatter(data[x_axis][:],data[y_axis][:])
plt.title('lable unknown')

plt.show()

nums_examples = data.shape[0]
x_train = data[[x_axis,y_axis]].values.reshape(nums_examples,2)

#指定好训练所需的参数
nums_clusters = 3
max_iteritions = 50

k_means = Kmeans(x_train,nums_clusters)
centroids, closest_centroids_ids = k_means.train(max_iteritions)

#对比结果
plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['Species']==iris_type],data[y_axis][data['Species']==iris_type],label = iris_type)
plt.title('lable known')
plt.legend()

plt.subplot(1,2,2)
for centroid_id,centroid in enumerate(centroids):
    current_examples_index = (closest_centroids_ids == centroid_id).flatten()
    plt.scatter(data[x_axis][current_examples_index], data[y_axis][current_examples_index], label=centroid_id)

for centroid_id,centroid in enumerate(centroids):
    plt.scatter(centroid[0],centroid[1],c='black',marker = 'x')
plt.legend()
plt.title('lable kmeans')
plt.show()