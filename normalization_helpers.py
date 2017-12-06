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
digit_data = train_data[train_data['class'] == 'DIGIT']
letter_data = train_data[train_data['class'] == 'LETTERS']
ordinal_data = train_data[train_data['class'] == 'ORDINAL']
address_data = train_data[train_data['class'] == 'ADDRESS']
telephone_data = train_data[train_data['class'] == 'TELEPHONE']
electronic_data = train_data[train_data['class'] == 'ELECTRONIC']
fraction_data = train_data[train_data['class'] == 'FRACTION']

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

# Normalize a string for the DIGIT class
def digit(x): 
    try:
        x = re.sub('[^0-9]', '',x)
        result_string = ''
        for i in x:
            digit = cardinal(i)
            digit = re.sub('zero', 'o', digit)
            result_string = result_string + digit + ' '
        result_string = result_string.strip()
        return result_string
    except:
        return(x)

# Normalize a string for the LETTERS class
def letters(x):
    try:
        x = re.sub('[^a-zA-Z]', '', x)
        x = x.lower()
        result_string = ''
        for i in range(len(x)):
            result_string = result_string + x[i] + ' '
        return(result_string.strip())  
    except:
        return x
    
# Normalize a string for the ORDINAL class
def ordinal(x):
    try:
        result_string = ''
        x = x.replace(',', '')
        x = x.replace('[\.]$', '')
        if re.match('^[0-9]+$',x):
            x = num2words(int(x), ordinal=True)
            return(x.replace('-', ' '))
        if re.match('.*V|X|I|L|D',x):
            if re.match('.*th|st|nd|rd',x):
                x = x[0:len(x)-2]
                x = rom_to_int(x)
                result_string = re.sub('-', ' ',  num2words(x, ordinal=True))
            else:
                x = rom_to_int(x)
                result_string = 'the '+ re.sub('-', ' ',  num2words(x, ordinal=True))
        else:
            x = x[0:len(x)-2]
            result_string = re.sub('-', ' ',  num2words(float(x), ordinal=True))
        return(result_string)  
    except:
        return x
    
# Normalize a string for the ADDRESS class
def address(x):
    try:
        x = re.sub('[^0-9a-zA-Z]+', '', x)
        result_string = ''
        for i in range(0,len(x)):
            if bool(re.match('[A-Z]|[a-z]',x[i])):
                result_string = result_string + x[i].lower() + ' '
            else:
                result_string = result_string + digit(x[i]) + ' '
                
        return(result_string.strip())        
    except:    
        return(x)

# Normalize a string for the TELEPHONE class
def telephone(x):
    x = x.replace('-','.').replace(')','.')
    text = p.number_to_words(x,group =1, decimal = "sil",zero = 'o').replace(',','')
    return text.lower()

# Normalize a string for the ELECTRONIC class
def electronic(x):
    try:
        replacement = {'.' : 'dot', ':' : 'colon', '/':'slash', '-' : 'dash', '#' : 'hash tag', }
        result_string = ''
        if re.match('.*[A-Za-z].*', x):
            for char in x:
                if re.match('[A-Za-z]', char):
                    result_string = result_string + letters(char) + ' '
                elif char in replacement:
                    result_string = result_string + replacement[char] + ' '
                elif re.match('[0-9]', char):
                    if char == 0:
                        result_string = result_string + 'o '
                    else:
                        number = cardinal(char)
                        for n in number:
                            result_string = result_string + n + ' ' 
            return result_string.strip()                
        else:
            return(x)
    except:    
        return(x)
    
# Normalize a string for the FRACTION class
def fraction(x):
    try:
        y = x.split('/')
        result_string = ''
        y0 = cardinal(y[0])
        y1 = ordinal(y[1])
        if int(y[1]) == 4:
            result_string = y0 + ' quarter'
        elif int(y[1]) == 2:
            result_string = y0 + ' half'
        else:    
            result_string = y0 + ' ' + y1
        if int(y[0]) > 1:
            result_string = result_string + 's'
        return(result_string)
    except:    
        return(x)

digit_data['normalized'] = digit_data.apply(lambda r: digit(r['before']), axis=1)

cardinal_data['normalized'] = cardinal_data.apply(lambda r: cardinal(r['before']), 
             axis=1)
letter_data['normalized'] = letter_data.apply(lambda r: letters(r['before']), 
           axis=1)
ordinal_data['normalized'] = ordinal_data.apply(lambda r: ordinal(r['before']), 
            axis=1)
address_data['normalized'] = address_data.apply(lambda r: address(r['before']), 
            axis=1)

telephone_data['normalized'] = telephone_data.apply(lambda r: telephone(r['before']), 
              axis=1)

electronic_data['normalized'] = electronic_data.apply(lambda r: electronic(r['before']), 
              axis=1)

fraction_data['normalized'] = fraction_data.apply(lambda r: fraction(r['before']), 
              axis=1)


wrong_cardinal = cardinal_data[cardinal_data['after'] != cardinal_data['normalized']]
wrong_digit = digit_data[digit_data['after'] != digit_data['normalized']]
wrong_letter = letter_data[letter_data['after'] != letter_data['normalized']]
wrong_ordinal = ordinal_data[ordinal_data['after'] != ordinal_data['normalized']]
wrong_address = address_data[address_data['after'] != address_data['normalized']]
wrong_telephone = telephone_data[telephone_data['after'] != telephone_data['normalized']]
wrong_electronic = electronic_data[electronic_data['after'] != electronic_data['normalized']]
wrong_fraction = fraction_data[fraction_data['after'] != fraction_data['normalized']]





