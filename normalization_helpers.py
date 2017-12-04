#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import inflect
from num2words import num2words
import re

p = inflect.engine()

train_data = pd.read_csv('data/train.csv', index_col = 0)
cardinal_data = train_data[train_data['class'] == 'CARDINAL']

# Convert roman numerals to integers. Return zero if there is no conversion.
def rom_to_int(string):
    table=[['M',1000],['CM',900],['D',500],['CD',400],['C',100],['XC',90],['L',50],['XL',40],
           ['X',10],['IX',9],['V',5],['IV',4],['I',1]]
    returnint = 0
    for pair in table:
        continueyes = True
        while continueyes:
            if len(string) >= len(pair[0]):

                if string[0:len(pair[0])] == pair[0]:
                    returnint += pair[1]
                    string = string[len(pair[0]):]
                else: continueyes = False
            else: continueyes = False
    return returnint   

# Normalize a string for the CARDINAL class 
def cardinal(x):
    try:
        if rom_to_int(x) !=0:
            x = str(rom_to_int(x))
        if re.match('.*[A-Za-z]+.*', x):
            return x
        x = re.sub(',', '', x, count = 10)

        if(re.match('.+\..*', x)):
            x = p.number_to_words(float(x))
        elif re.match('\..*', x): 
            x = p.number_to_words(float(x))
            x = x.replace('zero ', '', 1)
        else:
            x = p.number_to_words(int(x)) 
        x = re.sub('-', ' ', x, count=10)
        x = re.sub(' and','',x, count =10)
        x = re.sub(', ', ' ', x, count=10)
        return x
    except:
        return x

cardinal_data['normalized'] = cardinal_data.apply(lambda r: cardinal(r['before']), 
             axis=1)

wrong = cardinal_data[cardinal_data['after'] != cardinal_data['normalized']]