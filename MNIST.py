import tensorflow as tf
import numpy as np
import cv2
import os

def get_images(path):
	# Lista que almacenara los nombres de las imagenes del directorio "data"
	List_Images = []

	# Recorriendo el directorio de imagenes
	for file in os.listdir(path):

		# Almacenando en la lista el nombre del directorio 
		# mas el nombre de la imagen
		List_Images.append(os.path.join(path, file))

	# Lista que almacenara las imagenes en binario
	Images_Binarys = []

	# Recorriendo la lista de etiquetas
	for item in range(len(List_Images)):

		# Leyendo y cargando imagen
		imagen = cv2.imread(List_Images[item], 0)
		
		# Almacenando matriz de la imagen en un arreglo numpy
		temp = np.array(imagen)

		## Cambiando la forma de la matriz de la imagen 
		image_binary = np.reshape(temp, (1,784))

		# Almacenando la imagen en la lista
		Images_Binarys.append(image_binary)

	#return Images_Binarys
	#print(len(Images_Binarys[6][0]))
	#print(len(Images_Binarys[6][0]))
	Img_finales = []
	for item in range(len(Images_Binarys)):
		Img_finales.append(Images_Binarys[item][0])

	return Img_finales



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
imagen = cv2.imread('data/one.jpg', 0)
new = np.array(imagen)
b = np.reshape(new, (1,784))
print(b)
image_binary = get_images("data2")

result = y
print(sess.run(result, feed_dict={x:image_binary}))





