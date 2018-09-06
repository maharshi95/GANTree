import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.contrib import layers

from data import create_data
from autoencoder import * 


train, test = create_data(1)
epochs = 20000
display = 1000

del_logs = True

init = tf.global_variables_initializer()

sess = tf.Session()

if del_logs:
    os.system('rm -r logs/*')

train_writer = tf.summary.FileWriter('logs/train')
test_writer = tf.summary.FileWriter('logs/test')
    
sess.run(init)
en_loss_history = []
de_loss_history = []
di_loss_history = []
d_acc_history = []
g_acc_history = []
gen_loss_history= []
for i in range(1, epochs+1):
    
    z_train = np.random.uniform(-1, 1, [train.shape[0], 1])
    z_test = np.random.uniform(-1, 1, [test.shape[0], 1])
    
    sess.run(encoder_train_op, feed_dict={X: train, Z: z_train})
    
    if (i % 5) == 0:
        
        sess.run(gen_train_op, feed_dict={X: train, Z: z_train})
    else:
        sess.run(disc_train_op, feed_dict={X: train, Z: z_train})
    
    en_loss, de_loss, di_loss, d_acc, g_acc,g_loss, summary_train = sess.run([
        encoder_loss, 
        decoder_loss,
        disc_loss,
        disc_acc,
        gen_acc,
        gen_loss,
        summaries,
    ], feed_dict={X: train, Z: z_train})
    
    en_loss_test, de_loss_test, di_loss_test, d_acc_test, g_acc_test,g_loss_test, summary_test = sess.run([
        encoder_loss, 
        decoder_loss,
        disc_loss,
        disc_acc,
        gen_acc,
        gen_loss,
        summaries,
    ], feed_dict={X: test, Z: z_test})
    
    train_writer.add_summary(summary_train, global_step=i)
    test_writer.add_summary(summary_test, global_step=i)
    
    en_loss_history.append(en_loss)
    de_loss_history.append(de_loss)
    di_loss_history.append(di_loss)
    d_acc_history.append(d_acc)
    g_acc_history.append(g_acc)
    gen_loss_history.append(g_loss)
    
    if i % display == 0:
        print('Step %i: Encoder Loss: %f' % (i, en_loss))
        print('Step %i: Disc Acc: %f' % (i, d_acc))
        print('Step %i: Gen  Acc: %f' % (i, g_acc))
        print('Step %i: Disc Loss: %f' % (i, di_loss))
        print('Step %i: Gen  Loss: %f' % (i, g_loss))
        print
