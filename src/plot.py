import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.contrib import layers

import run

fig = plt.figure(figsize=(15, 5))
a1 = fig.add_subplot(231)
a2 = fig.add_subplot(232)
a3 = fig.add_subplot(233)
a4= fig.add_subplot(234)
a5 = fig.add_subplot(235)
a1.plot(en_loss_history)
a2.plot(g_acc_history)
a3.plot(d_acc_history)
a4.plot(di_loss_history)
a5.plot(gen_loss_history)