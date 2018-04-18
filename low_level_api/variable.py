from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

my_variable = tf.get_variable("my_variable", [1, 2, 3])
print(my_variable)
my_variable2 = tf.get_variable("my_variable2", [1, 2, 3], dtype=tf.int32, initializer=tf.zeros_initializer)
print(my_variable2)
other_variable = tf.get_variable("other_variable", dtype=tf.int32, initializer=tf.constant([23, 42]))
print(other_variable)

my_local = tf.get_variable("my_local", shape=(), collections=[tf.GraphKeys.LOCAL_VARIABLES])
my_non_trainable = tf.get_variable("my_non_trainable", shape=(), trainable=False)

tf.add_to_collection("my_collection_name", my_local)
tf.get_collection("my_collection_name")

v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = tf.get_variable("w", initializer=v.initialized_value() + 1)
print(v)
print(w)

x = v + 1
print(x)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

assignment = v.assign_add(1)
tf.global_variables_initializer().run(session=sess)
sess.run(assignment)
### eof
