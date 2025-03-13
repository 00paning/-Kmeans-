import numpy as np

class Kmeans:
    def __init__(self,data,nums_clustres):
        self.data = data
        self.nums_clustres = nums_clustres

    def train(self,max_iterations):
        #先随机选择k个中心点
        centroids = Kmeans.centroids_init(self.data,self.nums_clustres)
        #开始训练
        nums_examples = self.data.shape[0]
        closest_centroids_ids = np.empty((nums_examples,1))
        for _ in range(max_iterations):
            #得到当前每一个样本点到k个中心点的距离，找到最近
            closest_centroids_ids = Kmeans.centroids_find_closest(self.data,centroids)
            #进行中心点位置更新
            centroids = Kmeans.centroids_compute(self.data,closest_centroids_ids,self.nums_clustres)
        return centroids,closest_centroids_ids


    @staticmethod
    def centroids_init(data,nums_clustres):
        nums_examples = data.shape[0]
        random_ids = np.random.permutation(nums_examples)
        centroids = data[random_ids[:nums_clustres],:]
        return centroids
    @staticmethod
    def centroids_find_closest(data,centroids):
        nums_examples = data.shape[0]
        nums_centroids = centroids.shape[0]
        closest_centroids_ids = np.zeros((nums_examples,1))
        for example_index in range(nums_examples):
            distance = np.zeros((nums_centroids,1))
            for centroids_index in range(nums_centroids):
                distance_diff = data[example_index,:] - centroids[centroids_index,:]
                distance[centroids_index] = np.sum(distance_diff**2)
            closest_centroids_ids[example_index] = np.argmin(distance)
        return closest_centroids_ids
    @staticmethod
    def centroids_compute(data,closest_centroids_ids,nums_clustres):
        num_features = data.shape[1]
        centroids = np.zeros((nums_clustres,num_features))
        for centroids_id in range(nums_clustres):
            closest_ids = closest_centroids_ids == centroids_id
            centroids[centroids_id] = np.mean(data[closest_ids.flatten(),:],axis=0)
        return centroids

