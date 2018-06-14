# -*- coding: utf-8 -*-
#from main import *
import sys
import os

os.chdir('src') 

command_str_list = [ 

    "python build_vocab.py --sst --ver_q"

]

for command in command_str_list:
    os.system(command)
    print('\n\n       Done! (from root.py): ' + command + '\n\n')

os.chdir('..')
