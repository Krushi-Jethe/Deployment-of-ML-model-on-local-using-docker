import os
import requests
import tensorflow as tf
import numpy as np 
import time
from matplotlib import pyplot as plt

                       
class word_level_process_data():
    def __init__(self,text):
        self.text = text
    def word_level_process(self):
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(self.text)    
        print(f'\n The total words are :{len(tokenizer.word_index)+1} ')
        input_ = input('\n\nDo u want to print the word dictionary? (y/n)\n\n')
        if input_ == 'y':
            print(f'\n\nThe word index dictionary is :\n\n {tokenizer.word_index}')
        return tokenizer
    
    
    
    
class data_set_word_level():
    def __init__(self,text,tokenizer):
        self.text = text
        self.tokenizer = tokenizer
    def gen_data(self):
        # Initialize the sequences list
        input_sequences = []

        # Loop over every line
        for line in self.text:

            # Tokenize the current line
            token_list = self.tokenizer.texts_to_sequences([line])[0]

            # Loop over the line several times to generate the subphrases
            for i in range(1, len(token_list)):

                # Generate the subphrase
                n_gram_sequence = token_list[:i+1]

                # Append the subphrase to the sequences list
                input_sequences.append(n_gram_sequence)

        # Get the length of the longest line
        max_sequence_len = max([len(x) for x in input_sequences])

        # Pad all sequences
        input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

        # Create inputs and label by splitting the last token in the subphrases
        X, labels = input_sequences[:,:-1],input_sequences[:,-1]

        # Convert the label into one-hot arrays
        Y = tf.keras.utils.to_categorical(labels, num_classes=len(self.tokenizer.word_index)+1)
        
        return X , Y , max_sequence_len
