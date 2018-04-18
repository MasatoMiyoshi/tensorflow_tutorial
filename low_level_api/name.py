from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

c_0 = tf.constant(0, name="c")
print(c_0)
c_1 = tf.constant(2, name="c")
print(c_1)

with tf.name_scope("outer"):
    c_2 = tf.constant(2, name="c")
    print(c_2)

print(tf.device)
### EOF
