import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from tensorflow_addons.layers import CRF

inputs = tf.random.truncated_normal([2, 10, 5])
targets = tf.convert_to_tensor(np.random.randint(5, size=(2, 10)), dtype=tf.int32)

out = tf.keras.layers.Softmax(inputs)

crf_layer = CRF(5)
lens = tf.convert_to_tensor([9, 6], dtype=tf.int32)
log_likelihood, trans_paras = tfa.text.crf_log_likelihood(inputs, targets, lens)
batch_pred_sequence, batch_viterbi_score = tfa.text.crf_decode(inputs, trans_paras, lens)
loss = tf.reduce_sum(-log_likelihood)

print("log_likelihood is: {}".format(log_likelihood))
print("loss is {}".format(loss))
print("batch pred sequence is {}".format(batch_pred_sequence))

decoded_sequence, potentials, sequence_length, chain_kernel = crf_layer(inputs)
print("dfasdf")
