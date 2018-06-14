# SETTING
MODEL = 'lai_rcnn' 
IS_SAMPLE = False
EMBEDDING_DIM = 300  
NB_EPOCHS = 100
NB_REAL_EX = 10 # 10
USE_VAL_SET = True # same as USE_EARLY_STOP
VALIDATION_SPLIT = 0.10

from data_handler import *
from keras import backend as K
from keras.layers import Conv1D
from keras.layers import Dense, Input, Lambda, LSTM, TimeDistributed, GRU, SimpleRNN
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras.engine.topology import Layer
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from time import gmtime, strftime
from keras import initializers
from keras import regularizers
from keras import constraints
import numpy as np
import string
import sys
import os
import errno
import argparse

ps = PorterStemmer()
#seed = 7
#np.random.seed(seed)
parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")

####################################
""" PARSING STETTING """
####################################

# dataset
parser.add_argument('--agnews', action='store_true')
parser.add_argument('--yelpp', action='store_true')
parser.add_argument('--yelp_full', action='store_true')
parser.add_argument('--acl', action='store_true')
parser.add_argument('--sst', action='store_true')
# preprocessing version
parser.add_argument('--ver_a', action='store_true')
parser.add_argument('--ver_b', action='store_true')
parser.add_argument('--ver_c', action='store_true')
parser.add_argument('--ver_d', action='store_true')
parser.add_argument('--ver_e', action='store_true')
parser.add_argument('--ver_f', action='store_true')
parser.add_argument('--ver_g', action='store_true')
parser.add_argument('--ver_h', action='store_true')
parser.add_argument('--ver_i', action='store_true')
parser.add_argument('--ver_j', action='store_true')
parser.add_argument('--ver_k', action='store_true')
parser.add_argument('--ver_l', action='store_true')
parser.add_argument('--ver_m', action='store_true')
parser.add_argument('--ver_n', action='store_true')
parser.add_argument('--ver_o', action='store_true')
parser.add_argument('--ver_p', action='store_true')
parser.add_argument('--ver_q', action='store_true')
parser.add_argument('--ver_r', action='store_true')
# word embedding
parser.add_argument('--skip', action='store_true')
parser.add_argument('--glove_6b', action='store_true')
parser.add_argument('--glove_42b', action='store_true')
parser.add_argument('--glove_840b', action='store_true')
parser.add_argument('--fast', action='store_true')
parser.add_argument('--rand', action='store_true')
parser.add_argument('--average', action='store_true')
parser.add_argument('--gensim', action='store_true')
parser.add_argument('--gensimfast', action='store_true')
# word-embedding trainable check
parser.add_argument('--train', action='store_true')
parser.add_argument('--untrain', action='store_true')
# tune mode vs. test mode
parser.add_argument('--tune', action='store_true')
parser.add_argument('--test', action='store_true')
# model selection
parser.add_argument('--original', action='store_true')
parser.add_argument('--proposed', action='store_true')
# hyperparameters
parser.add_argument('-p', '--params', nargs='+')


# dataset argument parsing
if parser.parse_args().agnews == True: DATASET = 'agnews'
elif parser.parse_args().yelpp == True: DATASET = 'yelpp'
elif parser.parse_args().yelp_full == True: DATASET = 'yelp_full'
elif parser.parse_args().acl == True: DATASET = 'acl'
elif parser.parse_args().sst == True: DATASET = 'sst'
else:
    print("[arg error!] please add at least one dataset argument")
    exit()
# prep-version argument parsing
if parser.parse_args().ver_a == True:PRPR_VER = 'ver_a'
elif parser.parse_args().ver_b == True: PRPR_VER = 'ver_b'
elif parser.parse_args().ver_c == True: PRPR_VER = 'ver_c'
elif parser.parse_args().ver_d == True: PRPR_VER = 'ver_d'
elif parser.parse_args().ver_e == True: PRPR_VER = 'ver_e'
elif parser.parse_args().ver_f == True: PRPR_VER = 'ver_f'
elif parser.parse_args().ver_g == True: PRPR_VER = 'ver_g'
elif parser.parse_args().ver_h == True: PRPR_VER = 'ver_h'
elif parser.parse_args().ver_i == True: PRPR_VER = 'ver_i'
elif parser.parse_args().ver_j == True: PRPR_VER = 'ver_j'
elif parser.parse_args().ver_k == True: PRPR_VER = 'ver_k'
elif parser.parse_args().ver_l == True: PRPR_VER = 'ver_l'
elif parser.parse_args().ver_m == True: PRPR_VER = 'ver_m'
elif parser.parse_args().ver_n == True: PRPR_VER = 'ver_n'
elif parser.parse_args().ver_o == True: PRPR_VER = 'ver_o'
elif parser.parse_args().ver_p == True: PRPR_VER = 'ver_p'
elif parser.parse_args().ver_q == True: PRPR_VER = 'ver_q'
elif parser.parse_args().ver_r == True: PRPR_VER = 'ver_r'
else:
    print("[arg error!] please add at least one preprocessing-version argument")
    exit()
# word embedding argument parsing
if parser.parse_args().skip == True: WORD_EMBEDDING = 'skip_word2vec_300d'
elif parser.parse_args().glove_6b == True: WORD_EMBEDDING = 'glove_6b_300d'
elif parser.parse_args().glove_42b == True: WORD_EMBEDDING = 'glove_42b_300d'
elif parser.parse_args().glove_840b == True: WORD_EMBEDDING = 'glove_840b_300d'	
elif parser.parse_args().fast == True: WORD_EMBEDDING = 'fasttext_300d'
elif parser.parse_args().rand == True: WORD_EMBEDDING = 'rand_300d'
elif parser.parse_args().average == True: WORD_EMBEDDING = 'average_300d'
elif parser.parse_args().gensim == True: WORD_EMBEDDING = 'gensim_skip_300d'
elif parser.parse_args().gensimfast == True: WORD_EMBEDDING = 'gensim_fast_300d'
else:
    print("[arg error!] please add at least one word-embedding argument")
    exit()
# word embedding trainble argument parsing
if parser.parse_args().train == True: IS_TRAINABLE = True
elif parser.parse_args().untrain == True: IS_TRAINABLE = False
else:
    print("[arg error!] please add at least one trainable argument")
    exit() 
# tune mode vs. test mode
if parser.parse_args().tune == True: TUNE_MODE = True
elif parser.parse_args().test == True: TUNE_MODE = False
else:
    print("[arg error!] please add at least one trainable argument")
    exit() 
# model selection
if parser.parse_args().original == True: MODEL_VER = 'original'
elif parser.parse_args().proposed == True: MODEL_VER = 'proposed'
else:
    print("[arg error!] please add at least one model argument")
    exit()     
# hyperparameters
param_list = parser.parse_args().params[0].split(',')

BATCH_SIZE = int(param_list[0]) # [64, 128, 256]
PATIENCE = int(param_list[1]) # [5, 10, 15]     
C_DIM = int(param_list[2]) # [100, 200, 300]    
F_DIM = int(param_list[3]) # [100, 200, 300]
Y_DIM = int(param_list[4]) # [100, 200, 300]
RNN_MOD = param_list[-1] # LSTM / GRU / SimpleRNN
    
    
#################################################
""" PATH STETTING & MAKE DIRECTORIES """
#################################################   
### Path setting
ABSOLUTE_PATH = os.getcwd()
FULL_PATH = os.path.join(ABSOLUTE_PATH, MODEL, DATASET)
#if EXP_NAME == '':
#    EXP_NAME = strftime("%Y-%m-%d_%Hh%Mm", gmtime()) 


if TUNE_MODE == True:
    FULL_PATH = os.path.join(FULL_PATH, '(tune)___'+WORD_EMBEDDING+'___'+str(IS_TRAINABLE)+'___'+PRPR_VER+'___'+'-'.join(param_list), "")
elif TUNE_MODE == False:
    FULL_PATH = os.path.join(FULL_PATH, WORD_EMBEDDING+'___'+str(IS_TRAINABLE)+'___'+PRPR_VER+'___'+'-'.join(param_list), "")

    
if not os.path.exists(os.path.dirname(FULL_PATH)):
    try:
        os.makedirs(os.path.dirname(FULL_PATH))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise    
    
    
############################
""" PREPROCESSING """
############################    
# open file to save our log.
orig_stdout = sys.stdout
f_description = open(FULL_PATH+'description.txt', 'w')
sys.stdout = f_description

## Model hyper-parameters
print('Batch size: ', BATCH_SIZE)
print('Epoach size: ', NB_EPOCHS)

## Load vocabulary
vocab_filename = os.path.join('dataset-description', DATASET, 'vocab_'+PRPR_VER+'.txt')
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())

## Load, Preprocessing, Vectorize Dataset
Xtrain, ytrain, Xdev, ydev, X_test, y_test, is_dev_set, tokenizer, len_info = data_preprocessing(
                                                                        MODEL,
                                                                        DATASET,
                                                                        PRPR_VER,
                                                                        vocab,
                                                                        IS_SAMPLE) # is sample true or false?                                                                     
VOCA_SIZE = len(tokenizer.word_index) + 1
NUM_CLASSES = len_info[0] 

######################################
""" LOAD PRETRAINED WORD EMBEDDING """
######################################
if WORD_EMBEDDING == 'average_300d':
    temp_tensor = []
    for emb in ['skip_word2vec_300d', 'glove_6b_300d', 'fasttext_300d']:
        temp_matrix = load_pretrained_embedding(
                                            tokenizer.word_index, 
                                            emb,
                                            DATASET, # pickle 
                                            PRPR_VER, # pickle
                                            EMBEDDING_DIM,
                                            IS_SAMPLE,
                                            vocab) 
        temp_tensor.append(temp_matrix)
    embedding_matrix = np.array(list(map(lambda x:sum(x)/float(len(x)), zip(*temp_tensor))))

else:
    embedding_matrix = load_pretrained_embedding(
                                        tokenizer.word_index, 
                                        WORD_EMBEDDING,
                                        DATASET, # pickle 
                                        PRPR_VER, # pickle
                                        EMBEDDING_DIM,
                                        IS_SAMPLE,
                                        vocab)    

######################################
""" DEFINE MODEL """
######################################    



def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        #ait = K.dot(uit, self.u)
        ait = dot_product(uit, self.u) # for tensorflow
        
        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number epsilon to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]



def define_rcnn_model(VOCA_SIZE, cnt):

    org_word_seq = Input(shape = (None, ), dtype = "int32")
    left_context_seq = Input(shape = (None, ), dtype = "int32")
    right_context_seq = Input(shape = (None, ), dtype = "int32")

    if WORD_EMBEDDING == 'rand_300d':
        embedder = Embedding(VOCA_SIZE, # vocab_size
                            EMBEDDING_DIM,
                            embeddings_initializer = 'uniform',
                            trainable=IS_TRAINABLE) # is trainable?    
    
    else: # use pretrained word embedding
        embedder = Embedding(VOCA_SIZE, # vocab_size
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=IS_TRAINABLE)

    word_embed_seq = embedder(org_word_seq)
    left_context_emb_seq = embedder(left_context_seq)
    rght_context_emb_seq = embedder(right_context_seq)

    
    if MODEL_VER == 'original': 
        if RNN_MOD == 'LSTM': 
            left_context_vector = LSTM(C_DIM, return_sequences = True)(left_context_emb_seq)
            right_context_vector = LSTM(C_DIM, return_sequences = True, go_backwards = True)(rght_context_emb_seq)
        elif RNN_MOD == 'GRU': 
            left_context_vector = GRU(C_DIM, return_sequences = True)(left_context_emb_seq)
            right_context_vector = GRU(C_DIM, return_sequences = True, go_backwards = True)(rght_context_emb_seq)        
        elif RNN_MOD == 'SimpleRNN': 
            left_context_vector = SimpleRNN(C_DIM, return_sequences = True)(left_context_emb_seq)
            right_context_vector = SimpleRNN(C_DIM, return_sequences = True, go_backwards = True)(rght_context_emb_seq)        
        concat_all = concatenate([left_context_vector, word_embed_seq, right_context_vector], axis = 2)  
        
        latent_semantic_vector = TimeDistributed(Dense(Y_DIM, activation = "tanh"))(concat_all) 
        max_pool_output_vector = Lambda(lambda x: K.max(x, axis = 1), output_shape = (Y_DIM, ))(latent_semantic_vector) 
        output = Dense(NUM_CLASSES, input_dim = (Y_DIM), activation = "softmax")(max_pool_output_vector) 
    
    elif MODEL_VER == 'proposed': 
        if RNN_MOD == 'LSTM': 
            left_context_vector = LSTM(C_DIM, return_sequences = True)(left_context_emb_seq)
            right_context_vector = LSTM(C_DIM, return_sequences = True, go_backwards = True)(rght_context_emb_seq)
        elif RNN_MOD == 'GRU': 
            left_context_vector = GRU(C_DIM, return_sequences = True)(left_context_emb_seq)
            right_context_vector = GRU(C_DIM, return_sequences = True, go_backwards = True)(rght_context_emb_seq)        
        elif RNN_MOD == 'SimpleRNN': 
            left_context_vector = SimpleRNN(C_DIM, return_sequences = True)(left_context_emb_seq)
            right_context_vector = SimpleRNN(C_DIM, return_sequences = True, go_backwards = True)(rght_context_emb_seq) 
        conv_a = Conv1D(filters=F_DIM, kernel_size=3, padding='same', activation='relu')(word_embed_seq)   
        conv_b = Conv1D(filters=F_DIM, kernel_size=4, padding='same', activation='relu')(word_embed_seq)  
        conv_c = Conv1D(filters=F_DIM, kernel_size=5, padding='same', activation='relu')(word_embed_seq)         
        concat_all = concatenate([left_context_vector, word_embed_seq, right_context_vector, conv_a, conv_b, conv_c], axis = 2)  
        
        latent_semantic_vector = TimeDistributed(Dense(Y_DIM, activation = "tanh"))(concat_all) 
        max_pool_output_vector = Lambda(lambda x: K.max(x, axis = 1), output_shape = (Y_DIM, ))(latent_semantic_vector) 
        #att_output = AttentionWithContext()(latent_semantic_vector)
        #concat_output = concatenate([max_pool_output_vector, att_output], axis = -1)
        output = Dense(NUM_CLASSES, input_dim = (Y_DIM+Y_DIM), activation = "softmax")(max_pool_output_vector)    
    

    model = Model(inputs = [org_word_seq, left_context_seq, right_context_seq], outputs = output)
    model.compile(optimizer = "rmsprop", 
                  loss = "categorical_crossentropy", 
                  metrics = ["accuracy"])
    if cnt==1:              
        print('\n')
        print('[ MODEL SUMMARY ]')
        print(model.summary())

    return model
    
######################################
""" TRAIN AND EVALUATE MODEL """
######################################    
### PREPARRING DATASET FOR TRAINING
def shift_right(doc): return [VOCA_SIZE-1] + list(doc[:-1])
def shift_left(doc): return list(doc[1:]) + [VOCA_SIZE-1]

if is_dev_set == True: # if originally there exist dev dataset,
    if USE_VAL_SET == False:
        #X_train = Xtrain[:] + Xdev[:]
        #y_train = ytrain[:] + ydev[:]   
        X_train = np.concatenate((Xtrain,Xdev),axis=0)
        y_train = np.concatenate((ytrain,ydev),axis=0)          
        
        left_X_train = np.array(list(map(shift_right, X_train)))
        right_X_train = np.array(list(map(shift_left, X_train))) 
        
    elif USE_VAL_SET == True: # can do early stopping
        X_train = Xtrain[:]
        y_train = ytrain[:]
        X_val = Xdev[:]
        y_val = ydev[:]
        
        left_X_train = np.array(list(map(shift_right, X_train)))
        right_X_train = np.array(list(map(shift_left, X_train)))      
        left_X_val = np.array(list(map(shift_right, X_val)))
        right_X_val = np.array(list(map(shift_left, X_val)))
    
elif is_dev_set==False: # split train into dev and train because originally there not exist dev dataset

    if USE_VAL_SET == False:
        X_train = Xtrain[:]
        y_train = ytrain[:]
        
        left_X_train = np.array(list(map(shift_right, X_train)))
        right_X_train = np.array(list(map(shift_left, X_train))) 
        
    elif USE_VAL_SET == True: # can do early stopping
        nb_validation_samples = int(VALIDATION_SPLIT * Xtrain.shape[0])
        X_train = Xtrain[:-nb_validation_samples]
        y_train = ytrain[:-nb_validation_samples]
        X_val = Xtrain[-nb_validation_samples:]
        y_val = ytrain[-nb_validation_samples:]
        
        left_X_train = np.array(list(map(shift_right, X_train)))
        right_X_train = np.array(list(map(shift_left, X_train)))
        left_X_val = np.array(list(map(shift_right, X_val)))
        right_X_val = np.array(list(map(shift_left, X_val)))        

# for test set
left_X_test = np.array(list(map(shift_right, X_test)))
right_X_test = np.array(list(map(shift_left, X_test)))

cv_scores = []
val_accs = []
cnt = 1

while(True):
 
    # Load pre-defined Model
    model = define_rcnn_model(VOCA_SIZE, cnt)

    print('\n\n********************* ', cnt, ' - TRAINING START *********************')
    
    if USE_VAL_SET == False: 
        history = model.fit(
                        [X_train, left_X_train, right_X_train], 
                        y_train,
                        batch_size = BATCH_SIZE,
                        epochs = NB_EPOCHS,
                        verbose=2)       
    
    elif USE_VAL_SET == True:    
        early_stopping = EarlyStopping(monitor='val_acc', patience=PATIENCE, verbose=0)
        history = model.fit(
                        [X_train, left_X_train, right_X_train], 
                        y_train,
                        validation_data=(
                            [X_val, left_X_val, right_X_val], 
                            y_val),
                        batch_size = BATCH_SIZE,
                        epochs = NB_EPOCHS,
                        callbacks=[early_stopping],
                        verbose=2)
       
        
    print('\n[ HISTORY DURING TRAINING ]')
    print('\nhistory[loss]')
    print(history.history['loss'])
    print('\nhistory[acc]')
    print(history.history['acc'])    
    if USE_VAL_SET == True:
        print('\nhistory[val_loss]')
        print(history.history['val_loss'])
        print('\nhistory[val_acc]')
        print(history.history['val_acc'])
        print('max val acc:' + str(np.max(history.history['val_acc'])*100))
        val_accs.append(np.max(history.history['val_acc'])*100)

    
    with open(FULL_PATH+'results.txt', 'a') as r_f:
        r_f.write('\n\n********************* '+ str(cnt)+' - EVALUATION START *********************')    

        ## Accuracy
        r_f.write('\n[ ACCURACY ]')
        #_, acc1 = model.evaluate(X_train, y_train, verbose=0)
        #print('Train Accuracy: %f' % (acc1*100))
        _, acc2 = model.evaluate([X_test, left_X_test, right_X_test], y_test, verbose=0)
        cv_scores.append(acc2*100)
        r_f.write('\n*Test Accuracy: '+str(acc2*100))

        ## Classification report
        yhat = model.predict([X_test, left_X_test, right_X_test], verbose=0)
        y_hat = list(map(argmax, yhat))
        y_true = [np.where(r==1)[0][0] for r in y_test]
        r_f.write('\n')
        r_f.write('\n[ MICRO-AVERAGED SCORE ]')
        r_f.write('\n   precision:\t\t'+str(metrics.precision_score(y_true, y_hat, average='micro')))
        r_f.write('\n   recall:\t\t'+str(metrics.recall_score(y_true, y_hat, average='micro')))
        r_f.write('\n   f1-score:\t\t'+str(metrics.f1_score(y_true, y_hat, average='micro')))
        r_f.write('\n')
        r_f.write('\n[ MACRO-AVERAGED SCORE ]')
        r_f.write('\n   precision:\t\t'+str(metrics.precision_score(y_true, y_hat, average='macro')))
        r_f.write('\n   recall:\t\t'+str(metrics.recall_score(y_true, y_hat, average='macro')))
        r_f.write('\n   f1-score:\t\t'+str(metrics.f1_score(y_true, y_hat, average='macro')))
        r_f.write('\n')
        r_f.write(classification_report(y_true, y_hat))    
     
    
    # serialize weights to HDF5
    model.save_weights(FULL_PATH + str(cnt)+"___model.h5"+'___'+str(IS_TRAINABLE)+'___'+WORD_EMBEDDING+'___'+PRPR_VER)
    
    if cnt == NB_REAL_EX:
        summary_result_file_name = '(avg)'+str(round(np.mean(cv_scores),3))+'__(max)'+str(round(np.max(cv_scores),3))
        with open(FULL_PATH+summary_result_file_name+'.txt', 'w') as sr_f:
            sr_f.write('Mean: '+str(np.mean(cv_scores)))
            sr_f.write('\nMax: '+str(np.max(cv_scores)))
            sr_f.write('\nMin: '+str(np.min(cv_scores)))
            sr_f.write('\nstd: '+str(np.std(cv_scores)))
            sr_f.write('\n')
            sr_f.write('\n[ACCURACY LIST(sv_scores list)]\n')
            for item in cv_scores:
                sr_f.write(str(item))
                sr_f.write(' ')
                
                     
        val_acc_file_name = 'val=(avg)'+str(round(np.mean(val_accs),5))+'__(max)'+str(round(np.max(val_accs),5))
        with open(FULL_PATH+val_acc_file_name+'.txt', 'w') as sr_f:
            sr_f.write('Mean: '+str(np.mean(val_accs)))
            sr_f.write('\nMax: '+str(np.max(val_accs)))
            sr_f.write('\nMin: '+str(np.min(val_accs)))
            sr_f.write('\nstd: '+str(np.std(val_accs)))
            sr_f.write('\n')
            sr_f.write('\n[ACCURACY LIST(sv_scores list)]\n')
            for item in val_accs:
                sr_f.write(str(item))
                sr_f.write(' ')                

                
        parameter_set = 'BS'+str(BATCH_SIZE)+'_PA'+str(PATIENCE)+'_C'+str(C_DIM)+'_Y'+str(Y_DIM)
        with open(FULL_PATH+parameter_set+'.txt', 'w') as sr_f:
            sr_f.write('Null')
   
        break
        
    cnt+=1
        
# close files
sys.stdout = orig_stdout
f_description.close()

print('\n\n>> '+MODEL+', '+WORD_EMBEDDING+', '+DATASET+', '+PRPR_VER+', '+str(IS_TRAINABLE)+' : Complete! (from main.py)\n\n')    
    
    
    
    
    
    
    
    
    