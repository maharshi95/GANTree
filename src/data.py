import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.contrib import layers

def create_data(n):
    new_data = []
    num_samples = 10000
    train_ratio = 0.8
    theta = np.random.uniform(0,2*np.pi,num_samples)
    num_samples = theta.shape[0]
    n_train = int (train_ratio*num_samples)
    
    
    filter_theta = [
        [0, 15],
        [345, 360],
        [75, 95],
        [120, 139],
        [200, 270]
    ]
    
    theta = filter(lambda x: any([r[0] <= x * 180 / np.pi <= r[1] for r in filter_theta]), theta)
    
    r=1
    
    if n==1:
        points = np.transpose(np.array([r*np.cos(theta), r*np.sin(theta)]))
        new_data = np.array(points)
        n_train = int (train_ratio*new_data.shape[0])
        training, test = new_data[:n_train,:], new_data[n_train:,:]
        
        return training,test

    elif n==2:
        points = np.transpose(np.array([r*np.cos(theta), r*np.sin(theta)]))
        for point in points:
            if (-0.86<point[0]<0.86 and -0.14<point[1]<0.14):
                new_data.append(point)
        new_data = np.array(new_data)
        training, test = new_data[:n_train,:], new_data[n_train:,:]
        return training,test
    
    elif n==3:
        points = np.transpose(np.array([r*np.cos(theta), r*np.sin(theta)]))
        for point in points:
            if (-0.74<point[0]<0.86 and -0.14<point[1]<0.25):
                new_data.append(point)
        new_data = np.array(new_data)
        training, test = new_data[:n_train,:], new_data[n_train:,:]
        return training,test

points = create_data(1)
print points[0].shape
print(points[0])

train, test = create_data(1)
print test.shape
