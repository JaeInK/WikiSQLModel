from data_util import load_glove
import os


class Config(object):
    def __init__(self):
        self.vocab_file = "data/vocab"
        self.train_query_file = 'data/train.query.txt'
        self.train_context_file = 'data/train.context.txt'
        self.train_tables_file = 'data/train.tables.txt'
        self.train_answer_file = 'data/train.answer.txt'
       
        self.dev_query_file = 'data/dev.query.txt'
        self.dev_context_file = 'data/dev.context.txt'
        self.dev_tables_file = 'data/dev.tables.txt'
        self.dev_answer_file = 'data/dev.answer.txt'
        
        self.dict_file = "data/dict.p"
        self.max_vocab_size = 5e4
        self.max_q_len = 66
        self.max_c_len = 172
        self.max_a_len = 92
        self.debug = True
        self.num_epochs = 20
        self.batch_size = 16
        self.dropout = 0.1
        self.vocab_size = 5e4
        self.embedding_size = 300
        self.lr = 1e-3
        self.lstm_size = 128
        self.filter_size = 96
        self.attention_size = 128
        self.grad_clip = 5
        self.alpha = 1e-1
        self.beta = 1e-1
        self.l2_lambda = 3e-7
        self.num_heads = 8
        self.ans_limit = 20
        self.embeddings = load_glove("data/glove.npz")
        self.dir_output = "results/save/"
        self.dir_model = self.dir_output + "model.weights/"
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
