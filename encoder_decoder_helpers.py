# -*- coding: utf-8 -*-
# Code adopted from lstm_seq2seq.py in the keras examples, with MIT license
# (https://github.com/fchollet/keras/blob/master/examples/lstm_seq2seq.py)

import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.models import model_from_json

# Convert lists of input tokens and target tokens to one-hot format of encoder input, 
# decoder input, and decoder target
def convert_onehot(input_list, target_list, input_lookup, target_lookup, 
                   input_len, target_len):
    encoder_input = np.zeros((len(input_list), input_len, len(input_lookup)), 
                             dtype='float32')
    decoder_input = np.zeros((len(input_list), target_len, len(target_lookup)), 
                             dtype='float32')
    decoder_target = np.zeros((len(target_list), target_len, len(target_lookup)), 
                              dtype='float32')
    for r, (input_token, target_token) in enumerate(zip(input_list, target_list)):
        for t, char in enumerate(input_token):
            encoder_input[r, t, input_lookup[char]] = 1.
        for t, char in enumerate(target_token):
            decoder_input[r, t, target_lookup[char]] = 1.
            if t > 0:
                decoder_target[r, t - 1, target_lookup[char]] = 1.
    return encoder_input, decoder_input, decoder_target

# Construct encoder-decoder model and the inference models (encoder and decoder)
def encoder_decoder(input_vocab_size, target_vocab_size, hidden_dim):
    # Define the encoder model and keep the states
    encoder_inputs = Input(shape = (None, input_vocab_size))
    encoder = LSTM(hidden_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    
    # Set up the decoder with the encoder states as initial states
    decoder_inputs = Input(shape = (None, target_vocab_size))
    decoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    
    decoder_dense = Dense(target_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Construct the encoder decoder model
    encoder_decoder_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    # Define the inference models
    encoder_model = Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = Input(shape=(hidden_dim, ))
    decoder_state_input_c = Input(shape=(hidden_dim, ))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, 
                                                     initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs] + decoder_states)
    return encoder_decoder_model, encoder_model, decoder_model

# Decode a one-hot presentation of a token to its corresponding string
def decode_token(token, target_idx_lookup, target_char_lookup, decoder, encoder, 
                 max_length):
    # Encode the token as state vectors
    state_values = encoder.predict(token)
    
    # Initialize target sequence of length 1
    target_seq = np.zeros((1, 1, len(target_idx_lookup)))
    target_seq[0, 0, target_idx_lookup['\t']] = 1.
    
    stop = False
    decoded_token = ''
    while not stop:
        output_tokens, h, c = decoder.predict([target_seq] + state_values)
        

        # Pick the character with the highest logit value
        char_idx = np.argmax(output_tokens[0, -1, :])
        char = target_char_lookup[char_idx]
        
        # Exit when the max length is reached or the stop character is reached
        if (char == '\n') or (len(decoded_token) > max_length):
            stop = True
        else:
            # If not reaching the end, append the character to the decoded token
            decoded_token += char
            
        # Update the target sequence
        target_seq = np.zeros((1, 1, len(target_idx_lookup)))
        target_seq[0, 0, char_idx] = 1.
        
        # Update the states
        state_values = [h, c]
    return decoded_token

# Decode a 3D np array of one-hot presentations into a list of strings
def batch_decode(encoder_input, target_idx_lookup, target_char_lookup, decoder, encoder, 
                 max_length):
    prediction = []
    for i in range(encoder_input.shape[0]):
        print('{}/{}'.format(i + 1, encoder_input.shape[0]))
        str_token = decode_token(encoder_input[i:(i+1)], target_idx_lookup, target_char_lookup, 
                                               decoder, encoder, max_length)
        prediction.append(str_token)
    return prediction

# Save a dictionary of Keras models
def save_models(models_dict, directory):
    for name, model in models_dict.items():
        with open(directory + '/' + name + '.json', 'w') as file:
            file.write(model.to_json())
        model.save_weights(directory + '/' + name + '.h5')

# Load a list of Keras models
def load_models(model_names, directory):
    models = {}
    for model_name in model_names:
        with open(directory + '/' + model_name + '.json', 'r') as file:
            json_str = file.read()
        model = model_from_json(json_str)
        model.load_weights(directory + '/' + model_name + '.h5')
        models[model_name] = model
    return models