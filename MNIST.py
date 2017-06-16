import tensorflow as tf
import cv2
import numpy as np
import os,sys
import re

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#Inicio fase de entrenamiento
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#Inicio fase de evaluacion

pattern = r"\w+\.(BMP|bmp|JPG|jpg|BMP|bmp|PNG|png|JPEG|jpeg)"

image_binary = []

path ="data/"

for filename in os.listdir(path):

	result=re.match(pattern, filename)
	
	if result != None:
		imagen = cv2.imread( path + result.group(), 0)

		temp = np.array(imagen)
		image_binary.append(np.reshape(temp, (10,784)))

print image_binary

result = y
print(sess.run(result, feed_dict={x:image_binary}))




