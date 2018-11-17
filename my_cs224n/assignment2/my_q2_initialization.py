import numpy as np
import tensorflow as tf


def xavier_initializer(shape):
    epsilon = np.sqrt(6/np.sum(shape))
    out = tf.Variable(tf.random_uniform(shape=shape, minval=-epsilon, maxval=epsilon))
    return out



def test_initialization_basic():
    print("Running basic tests...") 
    shape = (1,)
    xavier_mat = xavier_initializer(shape)
    assert xavier_mat.get_shape() == shape

    shape = (1, 2, 3)
    xavier_mat = xavier_initializer(shape)
    assert xavier_mat.get_shape() == shape
    print("Basic (non-exhaustive) Xavier initialization tests pass") 
    
    
if __name__ == '__main__':
    test_initialization_basic()