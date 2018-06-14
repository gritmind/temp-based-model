# -*- coding: utf-8 -*-
#from main import *
import sys
import os
import argparse
parser = argparse.ArgumentParser()
os.chdir('src') 

# tune mode vs. test mode
parser.add_argument('--tune', action='store_true')
parser.add_argument('--test', action='store_true')



############################
""" model tuning mode """
############################
if parser.parse_args().tune == True:
    command_str_list = [ 
        
        # Parameter Description
        # -p (BATCH_SIZE), (PATIENCE), (C_DIM), (F_DIM), (Y_DIM), (RNN_MODULE) 
        
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --original --tune -p 64,10,200,-1,200,LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --original --tune -p 128,10,200,-1,200,LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --original --tune -p 256,10,200,-1,200,LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --original --tune -p 128,10,100,-1,200,LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --original --tune -p 128,10,200,-1,200,LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --original --tune -p 128,10,300,-1,200,LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --original --tune -p 128,10,200,-1,50,LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --original --tune -p 128,10,200,-1,100,LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --original --tune -p 128,10,200,-1,200,LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --original --tune -p 128,10,200,-1,300,LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --original --tune -p 128,10,200,-1,400,LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --original --tune -p 128,10,200,-1,200,GRU",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --original --tune -p 128,10,200,-1,200,LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --original --tune -p 128,10,200,-1,200,SimpleRNN",
        
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --proposed --tune -p 64,10,200,200,200, LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --proposed --tune -p 128,10,200,200,200, LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --proposed --tune -p 256,10,200,200,200, LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --proposed --tune -p 128,10,100,200,200, LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --proposed --tune -p 128,10,200,200,200, LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --proposed --tune -p 128,10,300,200,200, LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --proposed --tune -p 128,10,200,100,200, LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --proposed --tune -p 128,10,200,200,200, LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --proposed --tune -p 128,10,200,300,200, LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --proposed --tune -p 128,10,200,200,50, LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --proposed --tune -p 128,10,200,200,100, LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --proposed --tune -p 128,10,200,200,200, LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --proposed --tune -p 128,10,200,200,300, LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --proposed --tune -p 128,10,200,200,400, LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --proposed --tune -p 128,10,200,200,200, GRU",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --proposed --tune -p 128,10,200,200,200, LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --proposed --tune -p 128,10,200,200,200, SimpleRNN"
        
    ]

    
############################
""" model test mode  """
############################
if parser.parse_args().test == True:  
    command_str_list = [ 
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --original --test -p 128,10,200,-1,50,LSTM",
        "python lai_rcnn_test.py --sst --ver_q --glove_840b --train --proposed --test -p 128,10,200,300,300,GRU"
    ]    
    
    
   
for command in command_str_list:
    os.system(command)
    print('\n\n       Done! (from root.py): ' + command + '\n\n')
os.chdir('..')

