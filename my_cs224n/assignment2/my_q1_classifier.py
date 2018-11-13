import time

import numpy as np
import tensorflow tf

from q1_softmax import softmax
from q1_softmax import cross_entropy_loss
from model import Model
from utils.general_utils import get_minibatchs


class Config(object):
	
	n_samples = 1024
	n_features = 100
	n_classes = 5
	batch_size = 64
	n_epochs = 50
	lr = 1e-4
	
	
class SoftmaxModel(Model):
	def add_placeholders(self):
		self.input_placeholder = tf.placeholder(tf.float32, [self.config.batch_size, self.config.n_features])
		self.labels_placeholder = tf.placeholder(tf.int32, [self.config.batch_size, self.config.n_classes])
		
	def create_feed_dict(self, inputs_batch, labels_batch=None):
		feed_dict = {self.input_placeholder: input_batch, self.labels_placeholder: labels_batch}
		return feed_dict
	def add_prediction_op(self):
		with tf.Session() as sess:
			bias = tf.Variable(tf.random_uniform([self.config.n_classes]))
			W = tf.Variable(tf.random_uniform([self.config.n_features, self.config.n_classes]))
			z = tf.matull(self.input_placeholder, W) + bias
		pred = softmax(z)
		return pred
	def add_loss_op(self, pred):
		loss = cross_entropy_loss(self.labels_placeholder, pred)
		return loss
	def add_training_op(self, loss):
		train_op = tf.train.GradientDescentOptimizer(self.config.lr).minimize(loss)
		return train_op
	def run_epoch(self, sess, inputs, labels):
		n_minibatchs, total_loss = 0, 0
		for input_batch, labels_batch in get_minibatchs([inputs, labels], self.config.batch_size):
			n_minibatchs += 1
			total_loss += self.train_on_batch(sess, input_batch, labels)
		return total_loss / n_minibatchs
	def fit(self, sess, inputs, labels):
		losses = []
		for epoch in range(self.config.n_epochs):
			start_time = time.time()
			average_loss = self.run_epoch(sess, inputs, labels)
			duration = time.time() - start_time
			print('Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration))
			losses.append(average_loss)
		return losses
	def __init__(self, config):
		self.config = config
		self.build()
		
def test_softmax_model():
	config = Config()
	
	np.random.seed(1234)
	inputs = np.random.rand(config.n_samples, config.n_features)
	labels = np.zeros((config.n_samples, config.n_classes), dtype=np.int32)
	labels[:, 0] = 1
	
	with tf.Graph().as_default():
		model = SoftmaxModel(config)
		init = tf.global_variable_initializer()
		
		with tf.Session() as sess:
			sess.run(init)
			losses = model.fit(sess, inputs, labels)
			
	print("Basic (non-exhaustive) classifier tests pass")
	
if __name__ == "__main__":
	test_softmax_model()