from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y, {x: [[1,2,3],[4,5,6]]}))
