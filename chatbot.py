# Building a ChatBot with Deep NLP

# Importing the libraries
import numpy as np
import tensorflow as tf
import re
import time


# DATA PREPROCESSING

# Importing the dataset
with open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore') as f:
    lines = f.read().split('\n')

with open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore') as f:
    conversations = f.read().split('\n')

# Creating a dictionary that maps each line with its id
id_to_line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5: # to make sure that each line has the desired length
        id_to_line[_line[0]] = _line[4]

# Creating a list of all the conversations
conversations_ids = []
for conversation in conversations[:-1]: #excluding last row of conversations (it is empty)
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(','))

# Separating questions and answers
questions = []
answers = []

for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id_to_line[conversation[i]])
        answers.append(id_to_line[conversation[i+1]])
        
# not sure about this last step:
# second question is the first answer, third question is the second answer and so on...
# include step=2 to separate answers and questions?
# for i in range(0, len(conversation) - 1, 2):
#   ...

# Cleaning the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@,;.:<>{}+=~|?]", "", text)
    return text

clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

# Creating a dictionary that maps each word with its number of ocurrences
word_to_count = {}

for question in clean_questions:
    for word in question.split():
        if word not in word_to_count:
            word_to_count[word] = 1
        else:
            word_to_count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word_to_count:
            word_to_count[word] = 1
        else:
            word_to_count[word] += 1

# Creating two dictionaries that map words in questions and words in answers with a unique integer
threshold = 20 # minimum number of times words have to appear in questions or in answers to be considered
questions_words_to_int = {}

word_number = 0
for word, count in word_to_count.items():
    if count >= threshold:
        questions_words_to_int[word] = word_number
        word_number += 1

answers_words_to_int = {}
word_number = 0
for word, count in word_to_count.items():
    if count >= threshold:
        answers_words_to_int[word] = word_number
        word_number += 1

# Adding 4 tokens to these two dictionaries
# PAD -> for empty positions to ensure that all the sequences have the same length
# EOS -> end of string
# OUT -> words we previously excluded from the 2 dictionaries
# SOS -> start of string
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

for token in tokens:
    questions_words_to_int[token] = len(questions_words_to_int) + 1

for token in tokens:
    answers_words_to_int[token] = len(answers_words_to_int) + 1
    
# Creating the inverse dictionary of the answers_words_to_int dictionary
answers_ints_to_word = {w_i: w for w, w_i in answers_words_to_int.items()}

# Adding the End of String token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += " <EOS>" 

# Converting all the questions and all the answers into integers and
# replacing all the words that were filtered out by <OUT>
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questions_words_to_int:
            ints.append(questions_words_to_int['<OUT>'])
        else:
            ints.append(questions_words_to_int[word])
    questions_into_int.append(ints)

answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answers_words_to_int:
            ints.append(answers_words_to_int['<OUT>'])
        else:
            ints.append(answers_words_to_int[word])
    answers_into_int.append(ints)

# Sorting questions and answers by the length of questions
# to speed up the training phase
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25+1): # maximum length of questions is 25 words
    for i, q in enumerate(questions_into_int):
        if len(q) == length:
            sorted_clean_questions.append(q)
            sorted_clean_answers.append(answers_into_int[i])
            # once again, I'm not so sure about this...
            # check how questions list and answers list were defined...


# BUILDING THE SEQ2SEQ MODEL

# Creating placeholders for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob') # parameter that controls the dropout rate
    return inputs, targets, lr, keep_prob

# Preprocessing the targets
# Add SOS token to the beginning of each row in every batch
# and exclude last column (EOS)
# not so sure why we included EOS in a previous step and are now excluding it...
# we need to exclude it so that the sequences keep the same size, but
# why do we need to include SOS and exclude EOS?!?
def preprocess_targets(targets, word_to_int, batch_size):
    left_side = tf.fill([batch_size, 1], word_to_int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets

# Creating the Encoder RNN Layer
def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, 
                                                       cell_bw = encoder_cell, 
                                                       sequence_length = sequence_length, 
                                                       inputs = rnn_inputs, 
                                                       dtype = tf.float32)
    # we want just the second element
    return encoder_state

# Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, 
                        sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1], decoder_cell.output_size)
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0], 
                                                                              attention_keys, 
                                                                              attention_values, 
                                                                              attention_score_function, 
                                                                              attention_construct_function, 
                                                                              name = 'attn_dec_train')
    decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, 
                                                                  training_decoder_function, 
                                                                  decoder_embedded_input, 
                                                                  sequence_length, 
                                                                  scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)











    