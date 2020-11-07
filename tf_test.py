import tensorflow as tf
import numpy as np

np_a = np.empty([20, 32, 14, 14])
np_b = np.empty([20, 32, 14, 14])
np_a.fill(200)
np_b.fill(1)

tf_a = tf.convert_to_tensor(np_a, np.float32)
tf_b = tf.convert_to_tensor(np_b, np.float32)

mse = tf.keras.losses.MeanSquaredError()
print(mse(tf_a, tf_b).numpy())
