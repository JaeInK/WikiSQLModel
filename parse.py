#-*- coding: utf-8 -*-

import json
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
import json_lines

#if there is encoding error. type 'export PYTHONIOENCODING=utf8' in console

#define tokenize by nltk
"""
def tokenize(_context):
	sentContext = sent_tokenize(_context)
	words=[]
	for sent in sentContext:
		wordContext = word_tokenize(sent)
		for word in wordContext:
			words.append(word)
	return words
"""

tableidArr = []
contextArr = []
queryArr = []
answerArr = []

# make queryArr
with open('train.jsonl', 'rb') as f: # opening file in binary(rb) mode    
   for item in json_lines.reader(f):
        sentence = item['question']
        words = word_tokenize(sentence)
        tableidArr.append(item['table_id'])
        # query = question + table_id
        query = sentence + " Find it from table " + item['table_id'] + "."
        queryArr.append(query)
        
# make tableHash 
tableHash = {}
with open('train.tables.jsonl', 'rb') as f:
    for item in json_lines.reader(f):
        # make context and store it into tableHash
        # context = column names + sql grammar
        context = "Column names are "
        for header in item['header']:
            context = context + header + " /and "
        context = context[:len(context)-2] + " </s>" + "SQL grammars are NONE, MAX, MIN, COUNT, SUM, AVG, =, >, <, /start, /end, /pad, OP, SELECT, WHERE, AND, FROM."
        tableHash[item['id']] = [item['header'], item['types'], item['rows'], context]

# make contextArr
with open('train.jsonl', 'rb') as f: # opening file in binary(rb) mode    
    for item in json_lines.reader(f):
        contextArr.append(tableHash[item["table_id"]][3])


# make answerArr
with open('train.jsonl', 'rb') as f: # opening file in binary(rb) mode    
   for item in json_lines.reader(f):
        agg_ops = ['NONE', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        cond_ops = ['=', '>', '<', 'OP']
        syms = ['SELECT', 'WHERE', 'AND', 'COL', 'TABLE', 'CAPTION', 'PAGE', 'SECTION', 'OP', 'COND', 'QUESTION', 'AGG', 'AGGOPS', 'CONDOPS']
        
        answer = []
        #1 agg function
        answer.append(agg_ops[item['sql']['agg']])
        #2 SELECT
        answer.append("SELECT")
        #3 COLUMN NAME
        column_name = tableHash[item["table_id"]][0][item['sql']['sel']]
        column_name = word_tokenize(column_name)
        for word in column_name:
            answer.append(word)
        #4 FROM
        answer.append("FROM")
        #5 TABLE_ID
        answer.append(item["table_id"])
        #6 WHERE
        answer.append("WHERE")
        #7 CONDS
        for cond in item['sql']['conds']:
            #1 column name
            column_name = tableHash[item["table_id"]][0][cond[0]]
            column_name = word_tokenize(column_name)
            for word in column_name:    
                answer.append(word) 
            #2 operands
            answer.append(cond_ops[cond[1]])
            #3 text
            texts = str(cond[2])
            texts = word_tokenize(texts)
            for word in texts:    
                answer.append(word)
        answerArr.append(answer)
       

with open('data/train.tables.txt', 'wb') as f:
    pickle.dump(tableHash, f)
with open('data/train.query.txt', 'wb') as f:
    pickle.dump(queryArr, f)
with open('data/train.context.txt', 'wb') as f:
    pickle.dump(contextArr, f)
with open('data/train.answer.txt', 'wb') as f:
    pickle.dump(answerArr, f)


# print(queryArr[0])
# print(contextArr[0])
# test = tableidArr[2]
# print(test)
# print("Hello")
# print(tableHash[test])
# print(questionArr[0])
# print(word_tokenize(contextArr[0]))
# max_len = 0
# for sent in questionArr:
#     if max_len < len(sent):
#         max_len = len(sent) 

# print(max_len)
