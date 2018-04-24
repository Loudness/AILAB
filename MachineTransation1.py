
#encoding: utf-8


###################################################
# Machine Translation Lab
###################################################

# The idea is to translate from English to French
###################################################

                               
from dataload import simpleLoader
from tests import machineTranslationTests as tests
import collections
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


def run_machineTranslationLab1():

    
    print("Show if GPU/CPU is active")
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    print()

    # Load English data
    english_sentences = simpleLoader.load_data('data/translation/small_vocab_en')
    # Load French data
    french_sentences = simpleLoader.load_data('data/translation/small_vocab_fr')

    print('Dataset Loaded!')


    #Peek at imported data
    for sample_i in range(2):
        print('small_vocab_en Line {}:  {}'.format(sample_i + 1, english_sentences[sample_i]))
        print('small_vocab_fr Line {}:  {}'.format(sample_i + 1, french_sentences[sample_i]))


    english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
    french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

    #Show some stats
    print('{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
    print('{} unique English words.'.format(len(english_words_counter)))
    print('10 Most common words in the English dataset:')
    print(u'"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
    print()
    print('{} French words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
    print('{} unique French words.'.format(len(french_words_counter)))
    print('10 Most common words in the French dataset:')
    print(u'"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')


    def tokenize(x):
        """
        Tokenize x
        :param x: List of sentences/strings to be tokenized
        :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
        """
        myTokenizer = Tokenizer()
        myTokenizer.fit_on_texts(x)

        #lets see some stats	
        print("### Tokenizer stats ###") 
        print("word_counts") 
        print(myTokenizer.word_counts)
        print("document_count") 
        print(myTokenizer.document_count)
        print("word_index") 
        print(myTokenizer.word_index)
        print("word_docs") 
        print(myTokenizer.word_docs)
        # integer encode documents
        encoded_docs = myTokenizer.texts_to_sequences(x)
        #return None, None
        return encoded_docs, myTokenizer

    tests.test_tokenize(tokenize)

    # Tokenize Example output
    text_sentences = [
        'The quick brown fox jumps over the lazy dog .',
        'By Jove , my quick study of lexicography won a prize .',
        'This is a short sentence .']
    text_tokenized, text_tokenizer = tokenize(text_sentences)
    print(text_tokenizer.word_index)
    print()
    for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
        print('Sequence {} in x'.format(sample_i + 1))
        print('  Input:  {}'.format(sent))
        print('  Output: {}'.format(token_sent))

    ### Padding ###
    # When batching the sequence of word ids together, each sequence needs to be the same length. 
    # Since sentences are dynamic in length, we can add padding to the end of the sequences to make them the same length.

    def pad(x, length=None):
        """
        Pad x
        :param x: List of sequences.
        :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
        :return: Padded numpy array of sequences
        """
        padded = pad_sequences(x, maxlen=length, dtype='int32', padding='post', truncating='post')

        return padded


    tests.test_pad(pad)

    # Pad Tokenized output
    test_pad = pad(text_tokenized)
    for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
        print('Sequence {} in x'.format(sample_i + 1))
        print('  Input:  {}'.format(np.array(token_sent)))
        print('  Output: {}'.format(pad_sent))




    def preprocess(x, y):
        """
        Preprocess x and y
        :param x: Feature List of sentences
        :param y: Label List of sentences
        :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
        """
        preprocess_x, x_tk = tokenize(x)
        preprocess_y, y_tk = tokenize(y)

        preprocess_x = pad(preprocess_x)
        preprocess_y = pad(preprocess_y)

        # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
        preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

        return preprocess_x, preprocess_y, x_tk, y_tk


        ####

    preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = preprocess(english_sentences, french_sentences)
    
    max_english_sequence_length = preproc_english_sentences.shape[1]
    max_french_sequence_length = preproc_french_sentences.shape[1]
    english_vocab_size = len(english_tokenizer.word_index)
    french_vocab_size = len(french_tokenizer.word_index)

    print('Data Preprocessed')
    print("Max English sentence length:", max_english_sequence_length)
    print("Max French sentence length:", max_french_sequence_length)
    print("English vocabulary size:", english_vocab_size)
    print("French vocabulary size:", french_vocab_size)




    def logits_to_text(logits, tokenizer):
        """
        Turn logits from a neural network into text using the tokenizer
        :param logits: Logits from a neural network
        :param tokenizer: Keras Tokenizer fit on the labels
        :return: String that represents the text of the logits
        """
        index_to_words = {id: word for word, id in tokenizer.word_index.items()}
        index_to_words[0] = '<PAD>'

        return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

    print('`logits_to_text` function loaded.')


    def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
        """
        Build and train a basic RNN on x and y
        :param input_shape: Tuple of input shape
        :param output_sequence_length: Length of output sequence
        :param english_vocab_size: Number of unique English words in the dataset
        :param french_vocab_size: Number of unique French words in the dataset
        :return: Keras model built, but not trained
        """
         
        # Build the layers
        from keras.models import Sequential
        from keras.layers import SimpleRNN
        learning_rate = 0.03
        print("Input Shape:%s" % (input_shape,))      # (137861, 21, 1)
        print("Output Sequence Length:%s" % output_sequence_length)      # 21
        model=Sequential()
        model.add(SimpleRNN(french_vocab_size, input_shape = (input_shape[1], input_shape[2]), return_sequences=True))
        model.add(Dense(french_vocab_size, activation = 'softmax'))
         
        model.summary()

        model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(learning_rate), metrics=['accuracy'])
        return model

     

    """
    tests.test_simple_model(simple_model)
    
    # Reshaping the input to work with a basic RNN

    tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
    tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

    # Load a simple RNN
    simple_rnn_model = simple_model(tmp_x.shape, max_french_sequence_length, english_vocab_size+1, french_vocab_size+1)

    # Train the neural network
    simple_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

    # Print prediction(s)
    print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))
    """ 


    def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
        """
        Build and train a RNN model using word embedding on x and y
        :param input_shape: Tuple of input shape
        :param output_sequence_length: Length of output sequence
        :param english_vocab_size: Number of unique English words in the dataset
        :param french_vocab_size: Number of unique French words in the dataset
        :return: Keras model built, but not trained
        """
        from keras.models import Sequential
        from keras.layers import Embedding, LSTM, InputLayer, SimpleRNN

        learning_rate = 0.01
        print("Input Shape:%s" % (input_shape,))      # (137861, 21, 1)
        print("Output Sequence Length:%s" % output_sequence_length)      # 21
        model2=Sequential()
        #model2.add(Embedding(input_dim=english_vocab_size, output_dim=output_sequence_length))
        #model2.add(Embedding(input_dim=french_vocab_size, output_dim=output_sequence_length))
        #model2.add( Embedding(english_vocab_size, french_vocab_size, input_length=input_shape[1]))
       
        #model2.add(SimpleRNN(french_vocab_size, input_shape = (input_shape[1], input_shape[2]), return_sequences=True))
        #model2.add(Dense(french_vocab_size, activation = 'softmax'))
        
        #model2.add(InputLayer(input_shape=input_shape[1:]))
        model2.add(Embedding(input_dim=english_vocab_size, output_dim=output_sequence_length, mask_zero=False, input_shape=input_shape[1:]) )
        model2.add( GRU(output_sequence_length, return_sequences=True))
       
        model2.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
        model2.compile(loss=sparse_categorical_crossentropy,optimizer=Adam(learning_rate),metrics=['accuracy'])
        model2.summary()
        return model2

   
    """ 
    tests.test_embed_model(embed_model)    
    
    tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
    tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2]))

    print("preproc_french_sentences.shape[1]:" + str(preproc_french_sentences.shape[1]))   #21
    print("max_french_sequence_length:" + str(max_french_sequence_length))                 #21
    print("english_vocab_size:" + str(english_vocab_size))             #199
    print("french_vocab_size:" + str(french_vocab_size))               #345
    print("len(english_tokenizer.word_index):" + str(len(english_tokenizer.word_index)) )          #199
    print("len(french_tokenizer.word_index):" + str(len(french_tokenizer.word_index))  )           #345

 
    the_embed_model = embed_model(tmp_x.shape, max_french_sequence_length, english_vocab_size+1, french_vocab_size+1)
    #the_embed_model = embed_model(tmp_x.shape, preproc_french_sentences.shape[1], english_vocab_size+1, french_vocab_size+1)
    # Train the neural network
    the_embed_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

    # Print prediction(s)
    print(logits_to_text(the_embed_model.predict(tmp_x[:1])[0], french_tokenizer))          
    """ 

    def bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
        """
        Build and train a bidirectional RNN model on x and y
        :param input_shape: Tuple of input shape
        :param output_sequence_length: Length of output sequence
        :param english_vocab_size: Number of unique English words in the dataset
        :param french_vocab_size: Number of unique French words in the dataset
        :return: Keras model built, but not trained
        """
        from keras.models import Sequential
        from keras.layers import Embedding, LSTM, InputLayer, SimpleRNN

        learning_rate = 0.01
       
        model3=Sequential()
        model3.add(Bidirectional(GRU(output_sequence_length, return_sequences=True),input_shape=input_shape[1:]))
        model3.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
        model3.compile(loss=sparse_categorical_crossentropy,optimizer=Adam(learning_rate),metrics=['accuracy'])
        return model3
    """     
    tests.test_bd_model(bd_model)


    #Reshape
    tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
    tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2]))
    # Load a simple RNN
    the_bd_model = embed_model(tmp_x.shape, max_french_sequence_length, english_vocab_size+1, french_vocab_size+1)

    # Train the neural network
    the_bd_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

    # Print prediction(s)
    print(logits_to_text(the_bd_model.predict(tmp_x[:1])[0], french_tokenizer)) 
    """

    def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
        """
        Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN on x and y
        :param input_shape: Tuple of input shape
        :param output_sequence_length: Length of output sequence
        :param english_vocab_size: Number of unique English words in the dataset
        :param french_vocab_size: Number of unique French words in the dataset
        :return: Keras model built, but not trained
        """
        from keras.models import Sequential
        from keras.layers import Embedding, LSTM, InputLayer, SimpleRNN, Bidirectional

        learning_rate = 0.01
        MULTIPLY = 2
     
        #Lets continue as previous examples, but adding more layers.   
        the_final_model=Sequential()
        #model_final.add(Bidirectional(GRU(output_sequence_length, return_sequences=True),input_shape=input_shape[1:]))
        #model_final.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
        #model_final.compile(loss=sparse_categorical_crossentropy,optimizer=Adam(learning_rate),metrics=['accuracy'])

        

        the_final_model.add(Embedding(input_dim=english_vocab_size, output_dim=output_sequence_length, mask_zero=False, input_shape=input_shape[1:]) )
        the_final_model.add(Bidirectional( GRU(output_sequence_length * MULTIPLY , return_sequences=False)))
        #model_final.add(Embedding(english_vocab_size, 256, input_length=input_shape[1]))
        #model_final.add(Bidirectional(GRU(512, return_sequences=False)))                    
        the_final_model.add(RepeatVector(output_sequence_length))
        #the_final_model.add(Bidirectional(GRU(512, return_sequences=True)))  
        the_final_model.add(Bidirectional( GRU(output_sequence_length * MULTIPLY , return_sequences=True)))
        the_final_model.add(TimeDistributed(Dense(french_vocab_size, activation="softmax")))
        the_final_model.compile(loss=sparse_categorical_crossentropy,optimizer=Adam(learning_rate),metrics=['accuracy'])
        return the_final_model
    
    tests.test_model_final(model_final)


    print('Final Model Loaded')
    # TODO: Train the final model


    def final_predictions(x, y, x_tk, y_tk):
        """
        Gets predictions using the final model
        :param x: Preprocessed English data
        :param y: Preprocessed French data
        :param x_tk: English tokenizer
        :param y_tk: French tokenizer
        """
        #Train neural network using model_final
        lengEngTok = len(x_tk.word_index)+1
        lenFraTok = len(y_tk.word_index)+1
        model = None

        model = model_final( x.shape, y.shape[1], lengEngTok, lenFraTok)

        model.summary()

        model.fit(x, y, batch_size=1024, epochs=10, validation_split=0.2)
    
        ## DON'T EDIT ANYTHING BELOW THIS LINE
        y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
        y_id_to_word[0] = '<PAD>'

        sentence = 'he saw a old yellow truck'
        sentence = [x_tk.word_index[word] for word in sentence.split()]
        sentence = pad_sequences([sentence], maxlen=x.shape[-1], padding='post')
        sentences = np.array([sentence[0], x[0]])
        predictions = model.predict(sentences, len(sentences))

        print('Sample 1:')
        print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
        print('Il a vu un vieux camion jaune')
        print('Sample 2:')
        print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
        print(' '.join([y_id_to_word[np.max(x)] for x in y[0]]))


    final_predictions(preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer)


