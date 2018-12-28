#-*- coding: utf-8 -*-
import numpy as np
import json
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
import json_lines
from collections import Counter, OrderedDict
from tqdm import tqdm
import random
import tensorflow as tf

PAD = "<PAD>"
DUMMY = "<d>"
UNK = "<UNK>"
ID_PAD = 0
ID_DUMMY = 1
ID_UNK = 2
special_tokens = [PAD, DUMMY, UNK]

#if there is encoding error. type 'export PYTHONIOENCODING=utf8' in console

#define tokenize by nltk


class Vocab(object):
    def __init__(self, dict_file):
        # dict file is tok2idx
        with open(dict_file, "rb") as f:
            self._word2idx = pickle.load(f)
        self._idx2word = {v: k for k, v in self._word2idx.items()}
        
        
        print("hello")    
        #print(self.max_len())

    @staticmethod
    def build_vocab(query_file, context_file, tables_file, answer_file, vocab_file, dict_file, glove_file, output_file, max_vocab_size):
        
        # GLOVE file 
        # key: token , value: embedding
        tok2embedding = dict()
        
        with open(glove_file, "r", encoding='utf-8') as f:
            for line in tqdm(f, total=int(2.2e6)):
                line = line.strip()
                tokens = line.split(" ") 
                word = tokens[0]
                word_vec = list(map(lambda x: float(x), tokens[1:]))
                tok2embedding[word] = word_vec
        print("finish loading")
        

        # open input files
        with open(query_file, 'rb') as f:
            queryArr= pickle.load(f)
        with open(context_file, 'rb') as f:
            contextArr = pickle.load(f)
        with open(tables_file, 'rb') as f:
            tableHash = pickle.load(f)
        with open(answer_file, 'rb') as f:
            answerArr = pickle.load(f)

        # tok2idx
        # key: token , value: token_id
        tok2idx = dict()
        for token in special_tokens:
                idx = len(tok2idx)
                if token not in tok2idx:
                    tok2idx[token] = idx
        # print(tok2idx)
        
        # tok2idx procedure ....

        train_list = []
        for item in queryArr:
            temp = {}
            temp["query"] = item
            parsed_query = word_tokenize(item)
            temp["table_id"] = parsed_query[len(parsed_query)-2]
            temp["context"] = tableHash[temp["table_id"]][3]
            train_list.append(temp)
        word_list = []

        # query for our model
        for query in queryArr:
            lower_query = query.lower()
            tokened_query = word_tokenize(lower_query)
            for word in tokened_query:
                word_list.append(word)

        # context for our model
        for context in contextArr:
            lower_context = context.lower()
            replaced_context = lower_context.replace("</s>", "")
            tokened_context = word_tokenize(replaced_context)
            for word in tokened_context:
                word_list.append(word)
        
        # answer for our model
        for answer in answerArr:
            for word in answer:
                word_list.append(word.lower())

        # use Counter to order by occurence
        sorted_counter = Counter(word_list)
        sorted_counter = OrderedDict(sorted_counter.most_common())
        count = len(tok2idx)
        with open(vocab_file, "w", encoding="utf-8") as f:
            for word in sorted_counter:
                f.write(word + "\t" + str(sorted_counter[word]) + "\n")
                #print(word + "\t" + str(sorted_counter[word]) + "\n")
                #if word in tok2embedding and count < max_vocab_size:
                if word in tok2embedding:
                    idx = len(tok2idx)
                    tok2idx[word] = idx
                    count += 1

        with open(dict_file, "wb") as f:
            pickle.dump(tok2idx, f)
        # function save_glove
        save_glove(tok2embedding, tok2idx, output_file)
        print("save_glove")               

        

    # word2idx
    # key: word, value: word_id
    def word2idx(self, word):
        if word in self._word2idx:
            return self._word2idx[word]
        else:
            return self._word2idx[UNK]

    def size(self):
        return len(self._word2idx)


# function for batch_loader
def batch_loader(iterable, batch_size, shuffle=False):
    length = len(iterable)
    # shuffle iterable
    if shuffle:
        random.shuffle(iterable)
    for start_idx in range(0, length, batch_size):
        yield iterable[start_idx:min(length, start_idx + batch_size)]

# function for loading_data when training
def load_data(query_file, context_file, tables_file, answer_file, vocab, debug=False):
    with open(query_file, 'rb') as f:
        queryArr= pickle.load(f)
    with open(context_file, 'rb') as f:
        contextArr = pickle.load(f)
    with open(tables_file, 'rb') as f:
        tableHash = pickle.load(f)
    with open(answer_file, 'rb') as f:
        answerArr = pickle.load(f)

    # train_list contain "query", "table_id", "context" for every question set
    train_list = []
    for item in queryArr:
        temp = {}
        temp["query"] = item
        parsed_query = word_tokenize(item)
        temp["table_id"] = parsed_query[len(parsed_query)-2]
        temp["context"] = tableHash[temp["table_id"]][3]
        train_list.append(temp)
    
    tokenized_questions = []
    tokenized_contexts = []
    questions = []
    contexts = []
    answer_start = []
    answer_target = []


    for case in train_list:
        # convert question to sequence of word_id
        question = case["query"]
        lower_question = question.lower()
        q_tokens = word_tokenize(lower_question)
        tokenized_questions.append(q_tokens)
        q_tok2idx = list(map(lambda token: vocab.word2idx(token), q_tokens))
        questions.append(q_tok2idx)
        
        # convert context to sequence of word id
        context = case["context"]
        # remove sentence delimeter, tokenize and append it to list
        lower_context = context.lower()
        c_tokens = word_tokenize(lower_context.replace("</s>", ""))
        tokenized_contexts.append(c_tokens)
        # map token to its idx
        c_tokens = list(map(lambda token: vocab.word2idx(token), c_tokens))
        contexts.append(c_tokens)


    for case in answerArr:
        # convert answer(answer_start, answer_target) to sequence of word_id
        # answer_start
        start = ["/start"]
        for word in case:
            start.append(word.lower())
        start_tokens = list(map(lambda token: vocab.word2idx(token), start))
        answer_start.append(start_tokens)
        target = []
        for word in case:
            target.append(word.lower())
        target.append("/end")
        target_tokens = list(map(lambda token: vocab.word2idx(token), target))
        answer_target.append(target)
    
    data = zip(questions, contexts, tokenized_questions, tokenized_contexts, answer_start, answer_target)
    sorted_data =sorted(data, key=lambda x: len(x[1]))
    questions, contexts, tokenized_questions, tokenized_contexts, answer_start, answer_target = zip(*sorted_data)
    max_q_len = max_len(questions)
    max_c_len = max_len(contexts)
    max_a_len = max_len(answer_start)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(max_q_len, max_c_len, max_a_len)
    # get padding data
    question_len, padded_q= padding(tokenized_questions, max_q_len)
    context_len, padded_c = padding(tokenized_contexts, max_c_len)
    answer_target_len, padded_a_t= padding(answer_target, max_a_len)
    a, b = zero_padding(questions, max_q_len)
    #print(padded_q)
    #print(padded_s)
    
    # [batch, q+c] -> [56355, 238]
    merged_list = []
    for i in range(len(padded_q)):
        merged = padded_q[i] + padded_c[i]
        merged_list.append(merged)
    
    #print(merged_list)
    # print(np.array(merged_list).shape)
    print("#############################")
    print(merged_list[0])
    print(padded_a_t[0])
    print(b.shape)
    answer_target = []


    # very long procedure
    for i in range(len(merged_list)):
        temp1 = []
        for k in range(len(padded_a_t[i])):
            temp2 = [0]*len(merged_list[0])
            if(padded_a_t[i][k] != 0):
                for s in range(len(merged_list[i])):
                    if padded_a_t[i][k] == merged_list[i][s]:
                        temp2[s] = 1
                    else:
                        temp2[s] = 0
                
            else:
                for s in range(len(merged_list[i])):
                    if merged_list[i][s] == "/pad":
                        temp2[s] = 1
                        break
           # print(temp2)
            temp1.append(temp2)
        answer_target.append(temp1)
        print(i)
    print(np.array(answer_target).shape)
    answer_target = np.array(answer_target)
    # print(answer_target[0])
    # with open('data/train.answer.target.txt', 'wb') as f:
    #     pickle.dump(answer_target, f)

    print("@@@@@@@@@@@@@@@@@@@@@@@@@")
    # for i in padded_a_s[0]:
    #     for k in merged:
    #         if
    #for i in range(len(questions)):
    # answer_start_len, padded_a_s, max_a = zero_padding(answer_start)

    return tokenized_questions, tokenized_contexts, questions, contexts, answer_start, answer_target

# max_len padding
def max_len(inputs):
    sequence_length = [len(doc) for doc in inputs]
    max_length = max(sequence_length)
    return max_length

# padding function 
def padding(inputs, max_length):
    sequence_length = [len(doc) for doc in inputs]
    padded_docs = list(map(lambda doc: doc + [ID_PAD] * (max_length - len(doc)), inputs))
    return sequence_length, padded_docs

# padding function
def zero_padding(inputs, max_length):
    # inputs : [batch, sentence_len]
    sequence_length = [len(doc) for doc in inputs]
    padded_docs = list(map(lambda doc: doc + [ID_PAD] * (max_length - len(doc)), inputs))
    return np.array(sequence_length), np.array(padded_docs)


def save_glove(tok2embedding: dict, tok2idx: dict, output_file: str):
    embeddings = np.zeros([len(tok2idx) - 3, 300], dtype=np.float32)
    for i, token in enumerate(tok2idx.keys()):
        # skip special tokens. tok2embedding does not contain special tokens for keys
        try:
            idx = tok2idx[token] - 3
            word_vec = tok2embedding[token]
            embeddings[idx] = word_vec
        except KeyError:
            continue
    num_zeros = np.sum(embeddings, axis=1) == 0
    num_zeros = np.sum(num_zeros)
    print(num_zeros)
    np.savez_compressed(output_file, embeddings=embeddings)

def load_glove(filename):
    with np.load(filename) as f:
        return f["embeddings"]

# # max length of sentence's word tokens
# def max_len(arr):
#     max_l = 0
#     for sent in arr:
#         if max_l < len(sent):
#             max_l = len(sent) 

#     return max_l


#def build_voca_dic():

if __name__ == "__main__":

    vocab = Vocab("data/dict.p")
    query_file = 'data/train.query.txt'
    context_file = 'data/train.context.txt'
    tables_file = 'data/train.tables.txt'
    answer_file = 'data/train.answer.txt'
    vocab_file = "data/vocab"
    embedding_file = "data/glove.npz"
    glove_file = "data/glove.840B.300d.txt"
    dict_file = "data/dict.p"
    max_vocab_size = 5e4
    load_data(query_file, context_file, tables_file, answer_file, vocab, debug=False)