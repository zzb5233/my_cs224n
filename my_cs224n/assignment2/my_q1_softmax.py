import numpy as np
import tensorflow as tf
from utils.general_utils import test_all_close



def softmax(x):
	x_max = tf.reduce_max(x, 1, keep_dims=True)
	x_sub = tf.subtrack(x, x_max)
	x_exp = tf.exp(x_sub)
	sum_exp = tf.reduce_sum(x_exp, 1, keep_dims=True)
	out = tf.div(x_exp, sum_exp)
	return out
	
def cross_entropy_loss(y, yhat):
	l_yhat = tf.log(yhat)
	product = tf.multiply(tf.to_float(y), l_yhat)
	out = tf.negative(tf.reduce_sum(product))
	
	return out
	
	
def test_softmax_basic():
	test1 = softmax(tf.constant(np.array([[1001, 1002],[3, 4]]), dtype=tf.float32))
	with tf.Session as sess:
		test1 = sess.run(test1)
	with tf.Session() as sess:
		test2 = sess.run(test2)
	print('Basic (non-exhaustive) softmax tests pass')
	
def test_cross_entropy_loss_basic():
	y = np.array([[0, 1], [1, 0], [1, 0]])
	yhat = np.array([.5, .5], [.5, .5], [.5, .5])
	
	test1 = cross_entropy_loss(
		tf.constant(y, dtype = tf.int32)
		tf.constant(yhat, dtype = tf.float32))
		
	with tf.Session() as sess:
		test1 = sess.run(test1)
	expected = -3 * np.log(.5)
	
	print('Basic (non-exhaustive) cross-entropy tests pass')
	
if __name__ "__main__":
	test_softmax_basic()
	test_cross_entropy_loss_basic()
	
