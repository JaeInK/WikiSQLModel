import tensorflow as tf
from tensorflow.contrib import layers
import math
import os
from config import Config

max_context_sentence_len = 60 #TODO change constant
max_query_sentence_len = 70
max_previous_sentence_len = 80
max_target_sentence_len = 80
dim = 300
n_hidden =128


class Wiki(object):
#class wiki():
    def __init__(self, config):
    #def __init__(self):
        self.config = config
        self.q_len = self.config.max_q_len
        self.c_len = self.config.max_c_len
        self.a_len = self.config.max_a_len
        self.sess = None
        self.saver = None
        self.regularizer = layers.l2_regularizer(self.config.l2_lambda)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.merged = None
        self.summary_writer = None

    def build_model(self):
        # add place holder
        # TODO how to insert max_len???
        ###################################
        self.contexts = tf.placeholder(shape=[None, None], dtype=tf.int32, name="context")
        self.context_legnths = tf.placeholder(shape=[None], dtype=tf.int32, name="c_length")
        self.questions = tf.placeholder(shape=[None, None], dtype=tf.int32, name="question")
        self.question_legnths = tf.placeholder(shape=[None], dtype=tf.int32, name="q_length")
        self.answer_start = tf.placeholder(shape=[None, None], dtype=tf.int32, name="answer_start")
        self.answer_target = tf.placeholder(shape=[None, None, None], dtype=tf.int32, name="answer_target")        
        self.answer_lengths = tf.placeholder(shape=[None], dtype=tf.int32, name="a_length")
        self.max_q_len = tf.placeholder(dtype=tf.int32, name="max_q_len")
        self.max_c_len = tf.placeholder(dtype=tf.int32, name="max_c_len")
        self.max_a_len = tf.placeholder(dtype=tf.int32, name="max_a_len")
        self.dropout = tf.placeholder(dtype=tf.float32, name="dropout")
        self.avg_loss = tf.placeholder(dtype=tf.float32, name="avg_loss")
        self.avg_em = tf.placeholder(dtype=tf.float32, name="avg_em")
        self.avg_acc = tf.placeholder(dtype=tf.float32, name="avg_acc")
        loss_summary = tf.summary.scalar("loss", self.avg_em)
        acc_summary = tf.summary.scalar("accuracy", self.avg_acc)
        em_summary = tf.summary.scalar("em", self.avg_em)
        self.merged = tf.summary.merge([loss_summary, acc_summary, em_summary])
        #self.target = tf.placeholder(tf.float32, [None, self.a_len, self.q_len + self.c_len])      # TODO

        # add embeddings
        zeros = tf.constant([[0.0] * self.config.embedding_size])
        unk_dummy = tf.get_variable(shape=[2, self.config.embedding_size],
                                    initializer=layers.xavier_initializer(), name="special_token")
        # load pre-trained GloVe
        embedding_matrix = tf.Variable(initial_value=self.config.embeddings, trainable=False,
                                       dtype=tf.float32, name="embedding")
        self.embedding_matrix = tf.concat([zeros, unk_dummy, embedding_matrix], axis=0)
        self.embedded_contexts = tf.nn.embedding_lookup(self.embedding_matrix, self.contexts)
        self.embedded_contexts = tf.layers.dropout(self.embedded_contexts, self.dropout)
        self.embedded_questions = tf.nn.embedding_lookup(self.embedding_matrix, self.questions)
        self.embedded_questions = tf.layers.dropout(self.embedded_questions, self.dropout)
        self.embedded_answer_start = tf.nn.embedding_lookup(self.embedding_matrix, self.answer_start)
        self.embedded_answer_start = tf.layers.dropout(self.embedded_answer_start, self.dropout)
        

        
        #placeholders   
        # C = tf.placeholder(tf.float32, [None, max_context_sentence_len, dim])
        # Q = tf.placeholder(tf.float32, [None, max_query_sentence_len, dim])
        # previous_answer = tf.placeholder(tf.float32, [None, max_previous_sentence_len ,dim])
        # Target = tf.placeholder(tf.float32, [None, max_target_sentence_len, dim]) #Target SQL
           
        #TODO for coattention dim+1

        # https://pozalabs.github.io/blstm/
        with tf.variable_scope('CQ_BiLSTM_Embedding'):
            keep_prob = 0.8
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units = n_hidden)
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, state_keep_prob=keep_prob) #TODO state keep prob??? 
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units = n_hidden)
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, state_keep_prob=keep_prob)
            Coutputs, Cstates = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, self.embedded_contexts, dtype = tf.float32)     
            Qoutputs, Qstates = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_questions, dtype = tf.float32)
            #TOD  same LSTM LAYER?
            #TODO is it right to use hiddens state??

            # C/Qoutput = [batch, C/Qmax_len, n_hidden*2]
            Coutput = tf.concat([Coutputs[0], Coutputs[1]], 2)
            Qoutput = tf.concat([Qoutputs[0], Qoutputs[1]], 2) 

        with tf.variable_scope('co_attention'):
            # weigth
            #TODO change to dense layer
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
            Poutputs, pstates = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_, lstm_bw_cell_, self.embedded_answer_start, dtype = tf.float32)     
            #TODO is it right to use hiddens state??

            # Poutput = [batch, Pmax_len, n_hidden*2] [b, t, 2d]
            Poutput = tf.concat([Poutputs[0], Poutputs[1]], 2) #TODO Every Pmax_len gets different answer.... understand?
            
        #TODO is this meaning right?
        with tf.variable_scope('A_to_q1_attention'): #TODO I actually just followed coattention. should be changed
            Ptile1 = tf.tile(tf.expand_dims(Poutput, axis=2), [1,1, self.q_len, 1])
            q1 = tf.tile(tf.expand_dims(q1, axis=1), [1, self.a_len,1,1])
            tri1 = tf.concat([Ptile1, q1, Ptile1*q1], axis=-1) #[b,p,q,3d]
            # [b,p,q,3d] -> [b,p,q,p]
            score1 = tf.layers.dense(tri1, self.a_len, activation=tf.tanh)
            score1 = tf.nn.softmax(score1) #TODO softmax should be changed???
            #TODO think [b,q] as batch!!!!!!
            score1 = tf.transpose(score1, [0,2,1,3]) #[b,q,p,p]
            q1 = tf.transpose(q1, [0,2,1,3]) # [b,q,p,d]
            q2 = tf.matmul(score1, q1) # [b,q,p,p] * [b,q,p,d] -> [b,q,p,d]
            q2 = tf.transpose(q2, [0,2,1,3]) # [b,p,q,d]
            
        with tf.variable_scope('A_to_c1_attention'): #TODO just changed q2_to_c1 to A_to_c1
            Ptile2 = tf.tile(tf.expand_dims(Poutput, axis=2), [1,1,self.c_len,1])
            c1 = tf.tile(tf.expand_dims(c1, axis=1), [1, self.a_len,1,1])
            tri2 = tf.concat([Ptile2, c1, Ptile2*c1], axis=-1) #[b,p,c,3d]
            score2 = tf.layers.dense(tri2, self.a_len, activation=tf.tanh)
            score2 = tf.nn.softmax(score2)
            score2 = tf.transpose(score2, [0,2,1,3])
            c1 = tf.transpose(c1, [0,2,1,3])
            c2 = tf.matmul(score2, c1)
            c2 = tf.transpose(c2, [0,2,1,3]) # [b,p,c,d]
        
        with tf.variable_scope("Stacked_Encoder_Block"):
            c2 = tf.transpose(c2, [1,0,2,3]) #[p,b,c,d]
            print(c2)
            q2 = tf.transpose(q2, [1,0,2,3]) #[p,b,c,d]  
            print(q2)
            contexts = tf.map_fn(lambda x: self.residual_block(x, self.context_legnths,
                num_blocks=7, num_conv_blocks=2, num_filters=128, kernel_size=5, scope="Embedding_Encoder", reuse=True), c2)
            questions = tf.map_fn(lambda x: self.residual_block(x, self.question_legnths,
                num_blocks=7, num_conv_blocks=2, num_filters=128, kernel_size=5, scope="Embedding_Encoder1", reuse=True), q2)
            contexts = tf.transpose(contexts, [1,0,2,3])
            questions = tf.transpose(questions, [1,0,2,3])

        with tf.variable_scope("pointer_network"): 
            #TODO not sure if this is right way to change contexts=[b,p,c,d] ti [b,p,d,1]
            context_summary = tf.transpose(contexts, [0,1,3,2]) #[b,p,d,c]
            context_summary = tf.layers.dense(context_summary, 1, activation=tf.tanh) #[b,p,d,1]
            mat1 = tf.matmul(questions, context_summary) # [b,p,q,d] * [b,p,d,1] -> [b,p,q,1]
            pointer_question = tf.squeeze(mat1, axis=-1) # [b,p,q]
            
            question_summary = tf.transpose(questions, [0,1,3,2])
            question_summary = tf.layers.dense(question_summary, 1, activation=tf.tanh)
            mat2 = tf.matmul(contexts, question_summary)
            pointer_context = tf.squeeze(mat2, axis=-1) # [b,p,c]
           
            #implementing multiPointer using alpha
            pointer_qsum = tf.layers.dense(pointer_question, 32, activation=tf.tanh) #[b,p,32]
            pointer_csum = tf.layers.dense(pointer_context, 32, activation=tf.tanh) #[b,p,32]
            pointer_Sum = tf.concat([pointer_qsum, pointer_csum],2) #[b,p,64]
            pointer_Sum = tf.layers.dense(pointer_Sum, 1, activation=tf.tanh) # [b, p, 1]
            alphas = tf.tile(pointer_Sum, [1,1, self.c_len]) # [b, p, c]
            betas = tf.tile(pointer_Sum, [1,1, self.q_len]) # [b, p, q]
            pointer_context =tf.multiply(pointer_context,alphas)  
            pointer_question = tf.multiply(pointer_question, 1-betas) 
            pointer_vocab = tf.concat([pointer_context, pointer_question], 2) #[b, p, c+q]
            
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pointer_vocab, labels=self.answer_target))

            print(self.loss)
            pointer_vocab = tf.map_fn(lambda x: tf.nn.softmax(x), pointer_vocab, dtype=tf.float32)
            
            self.add_train_op()
            self.init_session()

            # print(pointer_vocab)
            #final output distribution
            #TODO put end token in context
            
    def add_train_op(self):
        lr = tf.minimum(self.config.lr, 0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
        opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
        vars = tf.trainable_variables()
        grads = tf.gradients(self.loss, vars)
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.config.grad_clip)
        self.train_op = opt.apply_gradients(zip(clipped_grads, vars), global_step=self.global_step)

    def evaluate_em(self, start_preds, end_preds, answer_spans, unans_probs, threshold=0.5):
        # if the question is unanswerable, answer span should be last token idx
        dummy_idx = self.context_legnths - 1
        start_preds = tf.where(tf.greater_equal(unans_probs, threshold), dummy_idx, start_preds)
        end_preds = tf.where(tf.greater_equal(unans_probs, threshold), dummy_idx, end_preds)
        ans_start, ans_end = tf.split(answer_spans, 2, axis=1)
        ans_start = tf.squeeze(ans_start, axis=-1)
        ans_end = tf.squeeze(ans_end, axis=-1)
        correct_start = tf.equal(start_preds, ans_start)
        correct_end = tf.equal(end_preds, ans_end)
        total_correct = tf.cast(tf.logical_and(correct_start, correct_end), dtype=tf.float32)
        em = tf.reduce_mean(total_correct)
        return em

    def mask_logits(self, logits, sequence_lengths):
        mask = tf.sequence_mask(sequence_lengths, dtype=tf.float32)
        mask_value = -1e32
        return logits + mask_value * (1 - mask)

    @staticmethod
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

    def layer_dropout(self, inputs, residual, dropout):
        cond = tf.random_uniform([]) < dropout
        return tf.cond(cond, lambda: residual, lambda: tf.layers.dropout(inputs, self.dropout) + residual)

    def residual_block(self, inputs, sequence_length, num_blocks, num_conv_blocks,
                       kernel_size, num_filters, scope, reuse):
        with tf.variable_scope(scope, reuse=reuse):
            sublayer = 1
            # conv_block * # + self attetion + feed forward
            total_sublayers = (num_conv_blocks + 2) * num_blocks
            dim = inputs.shape[-1]
            if dim != num_filters:
                inputs = tf.layers.dense(inputs, num_filters, activation=tf.nn.relu,
                                         kernel_regularizer=self.regularizer,
                                         kernel_initializer=layers.variance_scaling_initializer())
            for i in range(num_blocks):
                # add positional embedding
                inputs = inputs + self.position_embeddings(inputs, sequence_length)
                outputs, sublayer = self.conv_blocks(inputs, num_conv_blocks, kernel_size, num_filters,
                                                     scope="conv_block_{}".format(i), reuse=reuse,
                                                     sublayers=(sublayer, total_sublayers))
                outputs, sublayer = self.self_attention_block(outputs, sequence_length, (sublayer, total_sublayers),
                                                              scope="attention_block_{}".format(i), reuse=reuse)
                inputs = outputs
            return outputs

    def conv_blocks(self, inputs, num_conv_blocks, kernel_size,
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
                    normalized = tf.layers.dropout(normalized, self.dropout)
                outputs = self.depthwise_separable_conv(normalized, kernel_size, num_filters,
                                                        scope="depthwise_conv_{}".format(i), reuse=reuse)
                outputs = self.layer_dropout(outputs, residual, self.dropout * float(l) / L)
            return outputs, l

    def depthwise_separable_conv(self, inputs, kernel_size, num_filters, scope, reuse, rate=(1, 1)):
        with tf.variable_scope(scope, reuse=reuse):
            # [batch, t, 1, d]
            inputs = tf.expand_dims(inputs, axis=2)
            depthwise_filter = tf.get_variable(shape=[kernel_size, 1, num_filters, 1],
                                               initializer=layers.variance_scaling_initializer(),
                                               name="depthwise_filter",
                                               regularizer=self.regularizer)
            pointwise_filter = tf.get_variable(shape=[1, 1, num_filters, num_filters],
                                               name="pointwise_filter",
                                               initializer=layers.variance_scaling_initializer(),
                                               regularizer=self.regularizer)
            bias = tf.get_variable(shape=[num_filters],
                                   initializer=tf.zeros_initializer(),
                                   name="bias",
                                   regularizer=self.regularizer)

            outputs = tf.nn.separable_conv2d(inputs, depthwise_filter, pointwise_filter,
                                             rate=rate, strides=(1, 1, 1, 1), padding="SAME")
            outputs = tf.nn.relu(outputs + bias)
            # recover to the original shape [b, t, d]
            outputs = tf.squeeze(outputs, axis=2)
            return outputs

    def self_attention_block(self, inputs, sequence_length, sublayers, scope, reuse):
        with tf.variable_scope(scope, reuse=reuse):
            l, L = sublayers
            inputs = layers.layer_norm(inputs)
            inputs = tf.layers.dropout(inputs, self.dropout)
            outputs = self.multihead_attention(inputs, sequence_length)
            outputs = self.layer_dropout(outputs, inputs, self.dropout * l / float(L))
            l += 1
            # FFN
            residual = layers.layer_norm(outputs)
            outputs = tf.layers.dropout(outputs, self.dropout)
            hiddens = tf.layers.dense(outputs, self.config.attention_size * 2,
                                      activation=tf.nn.elu)
            fc_outputs = tf.layers.dense(hiddens, self.config.attention_size,
                                         activation=None)
            outputs = self.layer_dropout(residual, fc_outputs, self.dropout * l / float(L))

        return outputs, l

    def multihead_attention(self, queries, sequence_length):
        Q = tf.layers.dense(queries, self.config.attention_size,
                            kernel_initializer=layers.variance_scaling_initializer(),
                            activation=tf.nn.relu, kernel_regularizer=self.regularizer)
        K = tf.layers.dense(queries, self.config.attention_size,
                            kernel_initializer=layers.variance_scaling_initializer(),
                            activation=tf.nn.relu, kernel_regularizer=self.regularizer)
        V = tf.layers.dense(queries, self.config.attention_size,
                            kernel_initializer=layers.variance_scaling_initializer(),
                            activation=tf.nn.relu, kernel_regularizer=self.regularizer)

        Q_ = tf.concat(tf.split(Q, self.config.num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, self.config.num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, self.config.num_heads, axis=2), axis=0)
        # attention weight and scaling
        weight = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        weight /= (self.config.attention_size // self.config.num_heads) ** 0.5
        # key masking : assign -inf to zero padding
        key_mask = tf.sequence_mask(sequence_length, dtype=tf.float32)
        key_mask = tf.tile(key_mask, [self.config.num_heads, 1])
        key_mask = tf.tile(tf.expand_dims(key_mask, axis=1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(weight) * (-2 ** 32 + 1)
        weight = tf.where(tf.equal(key_mask, 0), paddings, weight)
        weight = tf.nn.softmax(weight)

        # query masking - assign zero to where query is zero padding token
        query_mask = tf.sequence_mask(sequence_length, dtype=tf.float32)
        query_mask = tf.tile(query_mask, [self.config.num_heads, 1])
        query_mask = tf.expand_dims(query_mask, axis=2)
        weight *= query_mask
        weight = tf.layers.dropout(weight, self.dropout)
        outputs = tf.matmul(weight, V_)
        outputs = tf.concat(tf.split(outputs, self.config.num_heads, axis=0), axis=2)

        return outputs

    def init_session(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.train_writer = tf.summary.FileWriter("./results/train", self.sess.graph)
        self.dev_writer = tf.summary.FileWriter("./results/dev", self.sess.graph)

    def save_session(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(self.sess, path)

    def restore_session(self, path):
        self.saver.restore(self.sess, path)

    def train(self, questions, question_length, contexts, context_lengths, 
                     answer_start, answer_target, answer_lengths, max_q_len, 
                     max_c_len, max_a_len, dropout=0.1):
        feed_dict = {
            self.questions: questions,
            self.question_legnths: question_length,
            self.contexts: contexts,
            self.context_legnths: context_lengths,
            self.answer_start: answer_start,
            self.answer_target: answer_target,
            self.answer_lengths: answer_lengths,
            self.max_q_len: max_q_len,
            self.max_c_len: max_c_len,
            self.max_a_len: max_a_len,
            self.dropout: dropout
        }
        # output_feed = [self.train_op, self.loss, self.acc, self.preds, self.global_step]
        output_feed = [self.train_op, self.loss, self.global_step]
        #_, loss, acc, pred, step = self.sess.run(output_feed, feed_dict)
        _, loss, step = self.sess.run(output_feed, feed_dict)
        #return loss, acc, pred, step
        return loss, step

    def eval(self, questions, question_length, contexts, context_lengths, 
                     answer_start, answer_target, answer_lengths, max_q_len, 
                     max_c_len, max_a_len, dropout=1.0):
        feed_dict = {
            self.questions: questions,
            self.question_legnths: question_length,
            self.contexts: contexts,
            self.context_legnths: context_lengths,
            self.answer_start: answer_start,
            self.answer_target: answer_target,
            self.answer_lengths: answer_lengths,
            self.max_q_len: max_q_len,
            self.max_c_len: max_c_len,
            self.max_a_len: max_a_len,
            self.dropout: dropout
        }
        #output_feed = [self.acc, self.em, self.loss]
        output_feed = [self.loss]
        #acc, em, loss = self.sess.run(output_feed, feed_dict)
        loss = self.sess.run(output_feed, feed_dict)
        #return acc, em, loss
        return loss

    def write_summary(self, avg_acc, avg_em, avg_loss, mode):
        feed_dict = {
            self.avg_acc: avg_acc,
            self.avg_em: avg_em,
            self.avg_loss: avg_loss
        }
        summary, step = self.sess.run([self.merged, self.global_step], feed_dict)
        if mode == "train":
            self.train_writer.add_summary(summary, step)
        else:
            self.dev_writer.add_summary(summary, step)

    def infer(self, questions, question_length, contexts,
              context_lengths, sentences, sequence_lengths, sentence_lengths):
        feed_dict = {
            self.questions: questions,
            self.question_legnths: question_length,
            self.contexts: contexts,
            self.context_legnths: context_lengths,
            self.sentences: sentences,
            self.sequence_lengths: sequence_lengths,
            self.sentence_lengths: sentence_lengths,
            self.dropout: 0.0
        }
        output_feed = [self.start, self.end]
        start, end = self.sess.run(output_feed, feed_dict)
        return start, end

if __name__ == "__main__":
    a = ["a","b","b","b","b", "c", "c","c"]
    config = Config()
    b = Wiki(config)
    b.build_model()
