import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
import math

#TODO Learning Rate, epochs, batch_size, batch_load???
#settings
max_context_sentence_len = 60 #TODO change constant
max_query_sentence_len = 70
max_previous_sentence_len = 80
max_target_sentence_len = 80
dim = 300
n_hidden =128
regularizer = layers.l2_regularizer(3e-7)

def position_embeddings(inputs, sequence_length):
        length = tf.shape(inputs)[1]
        channels = tf.shape(inputs)[2]
        max_timescale = 1.0e4
        min_timescale = 1.0
        position = tf.cast(tf.range(length), tf.float32)
        num_timescales = channels // 2
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (tf.to_float(num_timescales) - 1))
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])
        # mask for zero padding
        mask = tf.sequence_mask(sequence_length, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=2)
        signal *= mask
        return signal

def residual_block(inputs, sequence_length, num_blocks, num_conv_blocks,
					kernel_size, num_filters, scope, reuse):
	with tf.variable_scope(scope, reuse=reuse):
		sublayer = 1
		# conv_block * # + self attetion + feed forward
		total_sublayers = (num_conv_blocks + 2) * num_blocks
		dim = inputs.shape[-1]
		if dim != num_filters:
			inputs = tf.layers.dense(inputs, num_filters, activation=tf.nn.relu,
										kernel_regularizer=regularizer,
										kernel_initializer=layers.variance_scaling_initializer())
		for i in range(num_blocks):
			# add positional embedding
			inputs = inputs + position_embeddings(inputs, sequence_length)
			outputs, sublayer = conv_blocks(inputs, num_conv_blocks, kernel_size, num_filters,
													scope="conv_block_{}".format(i), reuse=reuse,
													sublayers=(sublayer, total_sublayers))
			outputs, sublayer = self_attention_block(outputs, sequence_length, (sublayer, total_sublayers),
															scope="attention_block_{}".format(i), reuse=reuse)
			inputs = outputs
			return outputs

def conv_blocks(inputs, num_conv_blocks, kernel_size,
                    num_filters, scope, reuse, sublayers=(1, 1)):
        with tf.variable_scope(scope, reuse=reuse):
            l, L = sublayers
            outputs = inputs

            for i in range(num_conv_blocks):
                residual = outputs
                # apply layer normalization
                normalized = layers.layer_norm(outputs)
                if i % 2 == 0:
                    # apply dropout
                    normalized = tf.layers.dropout(normalized, 0.1)
                outputs = depthwise_separable_conv(normalized, kernel_size, num_filters,
                                                        scope="depthwise_conv_{}".format(i), reuse=reuse)
                outputs = layer_dropout(outputs, residual, 0.1 * float(l) / L)
            return outputs, l

def layer_dropout(inputs, residual, dropout):
        cond = tf.random_uniform([]) < dropout
        return tf.cond(cond, lambda: residual, lambda: tf.layers.dropout(inputs, 0.1) + residual)


def depthwise_separable_conv(inputs, kernel_size, num_filters, scope, reuse, rate=(1, 1)):
	with tf.variable_scope(scope, reuse=reuse):
		# [batch, t, 1, d]
		inputs = tf.expand_dims(inputs, axis=2)
		depthwise_filter = tf.get_variable(shape=[kernel_size, 1, num_filters, 1],
											initializer=layers.variance_scaling_initializer(),
											name="depthwise_filter",
											regularizer=regularizer)
		pointwise_filter = tf.get_variable(shape=[1, 1, num_filters, num_filters],
											name="pointwise_filter",
											initializer=layers.variance_scaling_initializer(),
											regularizer=regularizer)
		bias = tf.get_variable(shape=[num_filters],
								initializer=tf.zeros_initializer(),
								name="bias",
								regularizer=regularizer)

		outputs = tf.nn.separable_conv2d(inputs, depthwise_filter, pointwise_filter,
											rate=rate, strides=(1, 1, 1, 1), padding="SAME")
		outputs = tf.nn.relu(outputs + bias)
		# recover to the original shape [b, t, d]
		outputs = tf.squeeze(outputs, axis=2)
		return outputs

def multihead_attention(queries, sequence_length):
	Q = tf.layers.dense(queries, 128,
						kernel_initializer=layers.variance_scaling_initializer(),
						activation=tf.nn.relu, kernel_regularizer=regularizer)
	K = tf.layers.dense(queries, 128,
						kernel_initializer=layers.variance_scaling_initializer(),
						activation=tf.nn.relu, kernel_regularizer=regularizer)
	V = tf.layers.dense(queries, 128,
						kernel_initializer=layers.variance_scaling_initializer(),
						activation=tf.nn.relu, kernel_regularizer=regularizer)

	Q_ = tf.concat(tf.split(Q, 8, axis=2), axis=0)
	K_ = tf.concat(tf.split(K, 8, axis=2), axis=0)
	V_ = tf.concat(tf.split(V, 8, axis=2), axis=0)
	# attention weight and scaling
	weight = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
	weight /= (128 // 8) ** 0.5
	# key masking : assign -inf to zero padding
	key_mask = tf.sequence_mask(sequence_length, dtype=tf.float32)
	key_mask = tf.tile(key_mask, [8, 1])
	key_mask = tf.tile(tf.expand_dims(key_mask, axis=1), [1, tf.shape(queries)[1], 1])

	paddings = tf.ones_like(weight) * (-2 ** 32 + 1)
	weight = tf.where(tf.equal(key_mask, 0), paddings, weight)
	weight = tf.nn.softmax(weight)

	# query masking - assign zero to where query is zero padding token
	query_mask = tf.sequence_mask(sequence_length, dtype=tf.float32)
	query_mask = tf.tile(query_mask, [8, 1])
	query_mask = tf.expand_dims(query_mask, axis=2)
	weight *= query_mask
	weight = tf.layers.dropout(weight, 0.1)
	outputs = tf.matmul(weight, V_)
	outputs = tf.concat(tf.split(outputs, 8, axis=0), axis=2)

	return outputs

def self_attention_block(inputs, sequence_length, sublayers, scope, reuse):
        with tf.variable_scope(scope, reuse=reuse):
            l, L = sublayers
            inputs = layers.layer_norm(inputs)
            inputs = tf.layers.dropout(inputs, 0.1)
            outputs = multihead_attention(inputs, sequence_length)
            outputs = layer_dropout(outputs, inputs, 0.1 * l / float(L))
            l += 1
            # FFN
            residual = layers.layer_norm(outputs)
            outputs = tf.layers.dropout(outputs, 0.1)
            hiddens = tf.layers.dense(outputs, 128 * 2,
                                      activation=tf.nn.elu)
            fc_outputs = tf.layers.dense(hiddens, 128,
                                         activation=None)
            outputs = layer_dropout(residual, fc_outputs, 0.1 * l / float(L))

        return outputs, l



#placeholders

C = tf.placeholder(tf.float32, [None, max_context_sentence_len, dim])
Q = tf.placeholder(tf.float32, [None, max_query_sentence_len, dim])
previous_answer = tf.placeholder(tf.float32, [None, max_previous_sentence_len ,dim])
Target = tf.placeholder(tf.float32, [None, max_target_sentence_len, dim]) #Target SQL

#TODO for coattention dim+1

# https://pozalabs.github.io/blstm/
with tf.variable_scope('CQ_BiLSTM_Embedding'):
	keep_prob = tf.placeholder(tf.float32)
	lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units = n_hidden)
	lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, state_keep_prob=keep_prob) #TODO state keep prob??? 
	lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units = n_hidden)
	lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, state_keep_prob=keep_prob)
	Coutputs, Cstates = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, C, dtype = tf.float32)     
	Qoutputs, Qstates = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, Q, dtype = tf.float32)
	print(Coutputs) # [batch, C, h]
	print(Cstates) # [batch, h]
	print(Qoutputs) # [batch, Q, h]
	print(Qstates) # [batch, h]

	#TODO is it right to use hiddens state??

	# C/Qoutput = [batch, C/Qmax_len, n_hidden*2]
	Coutput = tf.concat([Coutputs[0], Coutputs[1]], 2)
	Qoutput = tf.concat([Qoutputs[0], Qoutputs[1]], 2) 

with tf.variable_scope('co_attention'):
	# weigth
	W = tf.Variable(tf.random_normal([n_hidden*2, n_hidden*2])) #TODO is this trained??
	# https://neurowhai.tistory.com/112
	# bias
	b = tf.Variable(tf.random_normal([n_hidden*2]))

	Qtanh = tf.tanh(tf.add(tf.tensordot(Qoutput,W, axes=[[2],[0]]), b))
	A = tf.matmul(Qtanh, tf.transpose(Coutput, perm=[0,2,1])) # A.shape = [batch, Qmax, Cmax]
	A_transpose=tf.transpose(A, perm=[0,2,1]) #[batch, Cmax, Qmax]
	#normalize with respect to context/question by softmax
	c_to_q = tf.map_fn(lambda x: tf.nn.softmax(x), A, dtype=tf.float32) 
	q_to_c = tf.map_fn(lambda x: tf.nn.softmax(x), A_transpose, dtype=tf.float32)	
	q1 = tf.matmul(c_to_q, Coutput) #[batch, Qmax, Cmax] * [batch, Cmax, Hidden]
	c1 = tf.matmul(q_to_c, Qoutput) #TODO Qoutput or Qtanh???

with tf.variable_scope('Previous_Answer_Embedding'):
	lstm_fw_cell_ = tf.nn.rnn_cell.LSTMCell(num_units = n_hidden)
	lstm_fw_cell_ = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell_, state_keep_prob=keep_prob) #TODO state keep prob??? 
	lstm_bw_cell_ = tf.nn.rnn_cell.LSTMCell(num_units = n_hidden)
	lstm_bw_cell_ = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell_, state_keep_prob=keep_prob)
	Poutputs, pstates = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_, lstm_bw_cell_, previous_answer, dtype = tf.float32)     
	#TODO is it right to use hiddens state??

	# Poutput = [batch, Pmax_len, n_hidden*2] [b, t, 2d]
	Poutput = tf.concat([Poutputs[0], Poutputs[1]], 2) #TODO Every Pmax_len gets different answer.... understand?
	

with tf.variable_scope('A_to_q1_attention'): #TODO I actually just followed coattention. should be changed
	# weight [2h,2h]
	W1 = tf.Variable(tf.random_normal([n_hidden*2, n_hidden*2]))
	# bias [2h]
	b1 = tf.Variable(tf.random_normal([n_hidden*2]))
	# q1 = [b, q, 2d]
	Q1tanh = tf.tanh(tf.add(tf.tensordot(q1, W1, axes=[[2],[0]]), b1))
	print(Q1tanh)
	# A1 = [b, q, t]
	A1 = tf.matmul(Q1tanh, tf.transpose(Poutput, perm=[0,2,1]))
	print(A1)
	# [b, q, t]
	a_to_q1 = tf.map_fn(lambda x: tf.nn.softmax(x), A1, dtype=tf.float32)
	print(a_to_q1)
	# [b, q, t] * [b, t, 2d] -> [b, q, t, 1] * [b, q, 1, 2d]
	# q2 [b, q, 2d]
	q2 = tf.matmul(a_to_q1, Poutput)
	print(q2)

	## the result is q_2 -> [b,q,t,2d]



# bbs = np.array([30,40,5,1])
# bs = np.array([30,40,1,20])
# cs  = np.matmul(bbs,bs)
# print(cs.shape)
# print("Aaaaaaaaaaaaaaaaaaaaaaaaaaaa")

# Poutput = tf.tile(tf.expand_dims(Poutput, axis=2), [1, 1, max_query_sentence_len, 1])
# q1 = tf.tile(tf.expand_dims(q1, axis=1), [1, t, 1, 1])
# bi = tf.concat(Poutput, q1, axis=-1) # [b, t, q, 4d]
# # [b, t, q, 1] -> [b, t, q]
# score = tf.layers.dense(bi, 1, activation=None, use_bias=False, kernel_regularizer=self.regularizer)
# score = tf.squeeze(score, axis=-1)
# tf.nn.softmax(score)
#score = [b,t,q] , transpose(q1) =[b,q,t,4d]
#
#tf.matmul(score, transpose(q1, perm=[0,2,1,3]))
#tf.tensordot(score, transpose(q1, perm=[0,2,1,3]), axes=[[2], [[1]]])


with tf.variable_scope('q2_to_c1_attention'): #TODO I think... A_to_c1_ should be needed first...?
	W2 = tf.Variable(tf.random_normal([n_hidden*2, n_hidden*2]))
	b2 = tf.Variable(tf.random_normal([n_hidden*2]))
	Q2tanh = tf.tanh(tf.add(tf.tensordot(q2, W2, axes=[[2],[0]]), b2))
	A2 = tf.matmul(Q2tanh, tf.transpose(c1, perm=[0,2,1]))
	A2_transpose = tf.transpose(A2, perm=[0,2,1])
	q2_to_c1 = tf.map_fn(lambda x: tf.nn.softmax(x), A2_transpose, dtype=tf.float32)
	c2 = tf.matmul(q2_to_c1, q2) 
	print(c2)

# the result is c_2 -> [b,c,t,2d]
# the result is q_2 -> [b,q,t,2d]
print("@@@@@@@@@@@@@@@@@@@@@@@@@")
c_2 = tf.placeholder(tf.float32, [None, max_context_sentence_len, 
								max_previous_sentence_len, n_hidden*2])
q_2 = tf.placeholder(tf.float32, [None, max_query_sentence_len, 
								max_previous_sentence_len, n_hidden*2])

qqq = tf.placeholder(tf.float32, [None, max_query_sentence_len, n_hidden])

print("@@@@@@@@@@@@@@@@@@@@@@@@@")
print(c_2)
print(q_2)		


with tf.variable_scope("Model_Encoder_Context_Layer"):
	# inputs = tf.concat(attention_outputs, axis=2)
	# inputs = tf.layers.dense(inputs, self.config.attention_size,
	# 							kernel_regularizer=self.regularizer,
	# 							kernel_initializer=layers.variance_scaling_initializer(),
	# 							activation=tf.nn.relu)
	memories = []
	for i in range(3):
		print(qqq)
		print(max_query_sentence_len)
		outputs = residual_block(qqq, max_query_sentence_len,
										num_blocks=7, num_conv_blocks=2,
										num_filters=128, kernel_size=5,
										scope="Model_Encoder",
										reuse=True if i > 0 else False)
		print("##########################")
		print(outputs)
		if i == 2:
			outputs = tf.layers.dropout(outputs, 0.1)
		memories.append(outputs)
		inputs = outputs


#https://github.com/vincentliuk/Deep-Coattention/blob/master/coattention_train_test.py
#TODO softmax is pointwise????
#TODO with tf.~~ tensorboard
#TODO feed keep_prob

