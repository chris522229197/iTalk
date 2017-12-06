#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file contains functions to normalize written tokens to their spoken forms, for
# each class.

# Code adopted from https://www.kaggle.com/savannahvi/3-simple-steps-lb-9878-with-new-data 
# and https://www.kaggle.com/neerjad/class-wise-regex-functions-l-b-0-995

import inflect
from num2words import num2words
import re
p = inflect.engine()

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

# Normalize a string for the TIME class
def time(x):
    try:
        x = re.sub('\.','',x)
        if len(x.split(':')) == 2:
            x = re.sub(':',' ',x)
            x = re.sub('([^0-9])', r' \1 ', x)
            x = re.sub('\s{2,}', ' ', x)
            time_list = x.split(' ')
            t_list = [i for i in time_list if i != ""] 
            for i,v in enumerate(t_list):
                if v == '00':
                    t_list[i] = ''
                elif v.isdigit():
                    t_list[i] = num2words(int(v))
                else:
                    t_list[i] = v.lower()
            t = " ".join(t_list)
        elif len(x.split(':')) == 3:
            x_list = x.split(':')
            time_list = [num2words(int(num)) for num in x_list]
            time_units = []
            if int(x_list[0]) != 1:
                time_units.append('hours')
            else:
                time_units.append('hour')
            if int(x_list[1]) != 1:
                time_units.append('minutes')
            else:
                time_units.append('minute')
            if int(x_list[2]) != 1:
                time_units.append('seconds')
            else:
                time_units.append('second')
            t_list = [time_list[0],time_units[0],time_list[1],time_units[1],time_list[2],time_units[2]]
            t = " ".join(t_list)
        else:
            x = re.sub('([^0-9])', r' \1 ', x)
            x = re.sub('\s{2,}', ' ', x)
            time_list = x.split(' ')
            t_list = [i for i in time_list if i != ""] 
            for i,v in enumerate(t_list):
                if v == '00':
                    t_list[i] = ''
                elif v.isdigit():
                    t_list[i] = num2words(int(v))
                else:
                    t_list[i] = v.lower()
            t = " ".join(t_list)       
            
        time = re.sub(',',"",t)
        time = re.sub('  ', ' ', time)
        time_final = re.sub('-'," ",time)
        return(time_final)
    except:
        return(x)

# Normalize a string for the class DECIMAL
def decimal(x):
    try:
        x = re.sub(',','',x)
        deci_list = x.split('.')
        deci_list.insert(1,'point')
        if deci_list[0] =="":
            deci_list = deci_list[1:]
        else:
            deci_list[0] = num2words(int(deci_list[0]))
        #dealing with decimals after . (ex: .91 = point nine one)
        decimals_list = [num2words(int(num)) for num in deci_list[-1]]
        decimals = " ".join(decimals_list)
        decimals = re.sub('zero','o',decimals)
        deci_list[-1] = decimals
        return(" ".join(deci_list))
    except:
        return(x)

# Normalize a string for year
def year(x):
    year_list = [num for num in str(x)]
    if len(year_list)==4 and (year_list[0] == "1" or (year_list[0] == "2" and year_list[2]!='0')):
        year_list.insert(2, " ")
        year = "".join(year_list)
        year = year.split(' ')
        year = [num2words(int(num)) for num in year]
        year = " ".join(year)
    elif len(year_list)==4 and (year_list[0] == "1" or (year_list[0] == "2" and year_list[2]=='0')):
        year = "".join(year_list)
        year = num2words(int(year))
    elif  len(year_list)==2 and year_list[0]=='9':
        new_year_list = ["nineteen"]
        new_year_list.append("".join(year_list))
        new_year_list[1] = num2words(int(new_year_list[1]))
        year = " ".join(new_year_list)        
    elif len(year_list)==2 and year_list[0]!='0':
        new_year_list = ["twenty"]
        new_year_list.append("".join(year_list))
        new_year_list[1] = num2words(int(new_year_list[1]))
        year = " ".join(new_year_list)
    elif len(year_list)==2 and year_list[0]=='0':
        new_year_list = ['o']
        num = num2words(int(year_list[1]))
        new_year_list.append(num)
        year = " ".join(new_year_list)
    year = re.sub(',',"",year)
    year_final = re.sub('-'," ",year)
    return(year_final)

# Normalize a string for the class DATE
def date(x):
    try:
        x = re.sub(',','',x)    
        day = {"01":'first', "02":"second" , "03":'third', "04":'fourth', "05":'fifth', "06":'sixth', 
               "07":'seventh', "08":'eighth', "09":'ninth', "10":'tenth', "11":'eleventh',
        "12":'twelfth', "13":'thirteenth', "14":'fourteenth', "15":'fifteenth', 
        "16":'sixteenth', "17":'seventeenth', "18":'eighteenth', "19":'nineteenth', 
        "20":'twentieth', "21":'twenty-first', "22":'twenty-second', "23":'twenty-third', 
        "24":'twenty-fourth', "25":'twenty-fifth', "26":'twenty-sixth', "27":'twenty-seventh', 
        "28":'twenty-eighth', "29":'twenty-ninth', "30":'thirtieth', "31":'thirty-first', 
        "1":'first', "2":"second" , "3":'third', "4":'fourth', "5":'fifth', "6":'sixth', 
        "7":'seventh', "8":'eighth', "9":'ninth'}
    
        month = {"01":"January","02":"February","03":"March","04":"April","05":"May","06":"June",
             "07":"July", "08":"August","09":"September","1":"January","2":"February","3":"March", 
             "4":"April","5":"May","6":"June", "7":"July", "8":"August", "9":"September", 
             "10":"October","11":"November", "12":"December"}
    
        ord_days = {"1st":'first', "2nd":"second" , "3rd":'third', "4th":'fourth', "5th":'fifth', 
                    "6th":'sixth', "7th":'seventh', "8th":'eighth', "9th":'ninth', "10th":'tenth', 
                    "11th":'eleventh', "12th":'twelfth', "13th":'thirteenth', "14th":'fourteenth', 
                    "15th":'fifteenth', "16th":'sixteenth', "17th":'seventeenth', "18th":'eighteenth', 
                    "19th":'nineteenth', "20th":'twentieth', "21th":'twenty-first', 
                    "22nd":'twenty-second', "23rd":'twenty-third', "24th":'twenty-fourth', 
                    "25th":'twenty-fifth', "26th":'twenty-sixth', "27th":'twenty-seventh', 
                    "28th":'twenty-eighth', "29th":'twenty-ninth', "30th":'thirtieth', 
                    "31st":'thirty-first'}
        
        x = re.sub(',','',x)
        #Changing dates in form month/day/year
        if len(x.split("/")) == 3:
            date = x.split("/")
            date[0] = month[date[0]]
            date[1] = day[date[1]]
            date[2] = year(date[2])
            x_final = " ".join(date).lower() 
        #Changing dates in form day.month.year
        elif len(x.split(".")) == 3:
            date = x.split(".")
            date[1] = month[date[1]]
            date[0] = day[date[0]]+" of"
            date[2] = year(date[2])
            x_final = " ".join(date).lower() 
        # Dates written out
        elif len(x.split(' ')) > 1:  #testing for words (well sentences) like days and numbers with units
            date_list = x.split(' ')
            for i,v in enumerate(date_list):
                if v in ord_days:  #checking for date case 15th OF Jan.
                    if i == 0:
                        date_list[i] = "the "+ord_days[v]+" of"
                    else:
                        date_list[i] = ord_days[v]
                if v.isdigit():
                    if i == 0 and len(v)<3:
                        date_list[i] = "the "+day[v]+" of"
                    elif len(v)<3:
                        date_list[i] = day[v]
                    elif len(v)==4:
                        date_list[i] = year(v)
                x_final = " ".join(date_list)
        elif len(x) == 4:
            x_final = year(x)
        else:
            #in case we missed some (take a loss)
            x_final = x
            
        x_final = re.sub(',',"",x_final)
        x_final = re.sub('-'," ",x_final)
        return(x_final.lower())
    except:
        return(x)

# Check if a string is a float
def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Normalize a string for class MEASURE
def measure(x):
    try:
        x = re.sub(',','',x)
        sdict = {}
        sdict['km2'] = 'square kilometers'
        sdict['km'] = 'kilometers'
        sdict['kg'] = 'kilograms'
        sdict['lb'] = 'pounds'
        sdict['dr'] = 'doctor'
        sdict['sq'] = 'square'
        sdict['m²'] = 'square meters'
        sdict['in'] = 'inches'
        sdict['oz'] = 'ounces'
        sdict['gal'] = 'gallons'
        sdict['m'] = 'meters'
        sdict['m2'] = 'square meters'
        sdict['m3'] = 'cubic meters'
        sdict['mm'] = 'millimeters'
        sdict['ft'] = 'feet'
        sdict['mi'] = 'miles'
        sdict['ha'] = "hectare"
        sdict['mph'] = "miles per hour"
        sdict['%'] = 'percent'
        sdict['GB'] = 'gigabyte'
        sdict['MB'] = 'megabyte'
        SUB = {ord(c): ord(t) for c, t in zip(u"₀₁₂₃₄₅₆₇₈₉", u"0123456789")}
        SUP = {ord(c): ord(t) for c, t in zip(u"⁰¹²³⁴⁵⁶⁷⁸⁹", u"0123456789")}
        OTH = {ord(c): ord(t) for c, t in zip(u"፬", u"4")}
    
        x = x.translate(SUB)
        x = x.translate(SUP)
        x = x.translate(OTH)
        
        if len(x.split(' ')) > 1:
            m_list = x.split(' ')
        elif "/" in x:
            m_list = x.split('/')
            m_list.insert(1,'per')
        else:
            x = re.sub('([^0-9\.])', r' \1 ', x)
            x = re.sub('\s{2,}', ' ', x)
            measure_list = x.split(' ')
            measure_list = [i for i in measure_list if i != ""]
            m_list = [measure_list[0],"".join(measure_list[1:])]
        for i,v in enumerate(m_list):
            if v.isdigit():
                srtd = v.translate(SUB)
                srtd = srtd.translate(SUP)
                srtd = srtd.translate(OTH)
                m_list[i] = num2words(float(srtd))
            elif is_float(v):
                m_list[i] = decimal(v)
            elif v in sdict:
                m_list[i] = sdict[v]
        measure = " ".join(m_list)
        measure = re.sub(',',"",measure)
        measure_final = re.sub('-'," ",measure)
        return(measure_final)
    except:
        return(x)

# Normalize a string for the class MONEY
def money(x):
    try:
        money = re.sub('([$£€])', r'\1 ', x)
        money = re.sub('\s{2,}', ' ', money)
        money = re.sub(r',', '', money)
        money_list = money.split(' ')
        if money_list[0] == '$':
            money_list.append('dollars')
            money_list = money_list[1:]
        elif money_list[0] == '£':
            money_list.append('pounds')
            money_list = money_list[1:]
        elif money_list[0] == '€':
            money_list.append('euros')
            money_list = money_list[1:]    
        for i in range(len(money_list)):
            if money_list[i].isdigit():
                money_list[i] = num2words(int(money_list[i]))
            elif is_float(money_list[i]):
                money_list[i] = num2words(float(money_list[i]))
                money_list[i] = re.sub(r' zero', '', money_list[i])
        x = ' '.join(money_list)
        x = re.sub(r',', '', x)
        x = re.sub(r'-', ' ', x)
        return(x)
    except:
        return(x)
    
# Normalize a token based on its class
def normalize_token(token, token_class):
    try:
        if token_class == 'DATE':
            return date(token)
        elif token_class == 'LETTERS':
            return letters(token)
        elif token_class == 'CARDINAL':
            return cardinal(token)
        elif token_class == 'DECIMAL':
            return decimal(token)
        elif token_class == 'MEASURE':
            return measure(token)
        elif token_class == 'MONEY':
            return money(token)
        elif token_class == 'ORDINAL':
            return ordinal(token)
        elif token_class == 'TIME':
            return time(token)
        elif token_class == 'ELECTRONIC':
            return electronic(token)
        elif token_class == 'DIGIT':
            return digit(token)
        elif token_class == 'FRACTION':
            return fraction(token)
        elif token_class == 'TELEPHONE':
            return telephone(token)
        elif token_class == 'ADDRESS':
            return address(token)
        else:
            return token
    except:
        return token
    