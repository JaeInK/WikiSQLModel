import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

# elmo = hub.Module("https://tfhub.dev/google/elmo/1")
# embeddings = elmo(["the cat is on the mat", "dogs are in the fog"], signature = "default", as_dict = True)["elmo"]
# print(embeddings)

# word_to_embed = "dog"

# elmo = hub.Module("https://tfhub.dev/google/elmo/2")
# embedding_tensor = elmo([word_to_embed], as_dict=True)["word_emb"] 
# # Use as_dict because I want the whole dict, then I select "word_emb"

# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
#   embedding = sess.run(embedding_tensor)
#   print(embedding.shape)

# words_to_embed = ["dog is happy", "cat is sad" , "google is very important"] 

# elmo = hub.Module("https://tfhub.dev/google/elmo/2")
# embedding_tensor = elmo(words_to_embed) # <-- removed other params

# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
#   embedding = sess.run(embedding_tensor)
#   print(embedding.shape)

# dic = {'a':3, 'b':2, 'c':1}
# print(dic)
# for data in dic:
    



elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
tokens_input = [["the", "cat", "is", "on", "the", "mat", "djflqe"],
["dogs", "are", "in", "the", "fog", ""]]
tokens_length = [6, 5]
embeddings = elmo(inputs={"tokens": tokens_input,"sequence_len": tokens_length},
                                        signature="tokens", as_dict=True)["elmo"]
print(embeddings)