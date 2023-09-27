# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Question 1
# Step 1: Define a readin function
def read_file(fn):
    txt = open(fn, 'r')
    print(txt.read().strip())

# Step 2: Running the function
#print(read_file('text1.txt'))

# Question 2
# A simple method 
def file_read_first_n(fn, nlines):
    ans = []
    with open(fn) as f:
        lns = f.readlines()
        
    for ln in lns[:2]:
        print(ln.strip())
        
    f.close()
        
# An advanced method
def file_read_first_n_more(fn, nlines):
    from itertools import islice
    with open(fn, 'r') as f:
        for ln in islice(f, nlines):
            print(ln.strip())
    
    f.close()
        
#file_read_first_n_more('text1.txt', 2)

# Question 3
def file_write(fn):
    with open(fn, 'w') as myfile:
        myfile.write('Renjie Wan')
        myfile.write('Jun Qi')
    txt = open(fn)
    print(txt.read())
    myfile.close()
    
#file_write('output.txt')

# Question 4
def file_read(fname):
        with open (fname, "r") as myfile:
                data=myfile.readlines()
                print(data)

#file_read('text1.txt')

# Question 6
def longest_word(fn):
    tmp_w = ''
    tmp_l = 0
    with open(fn, 'r') as infile:
        words = infile.read().split()
        
    for word in words:
        if len(word) > tmp_l:
            tmp_w = word
            tmp_l = len(word)
            
    return tmp_w

def longest_word1(filename):
    with open(filename, 'r') as infile:
              words = infile.read().split()
    max_len = max(len(word) for word in words)
    return [word for word in words if len(word) == max_len]

#print(longest_word1('text1.txt'))
    

# Question 8
from collections import Counter
def word_count(fn):
    with open(fn) as f:
        return Counter(f.read().split())
    
#print('Number of words in the file: ', word_count('text1.txt'))

# Question 9
import random
def random_line(fn):
    lines = open(fn).read().splitlines()
    
    return random.choice(lines)

#print(random_line('text1.txt'))

# Question 10
f = open('text1.txt', 'r')
#print(f.closed)
f.close()
#print(f.closed)   

# Question 11
import string, os
if not os.path.exists("letters"):
    os.makedirs("letters")
for letter in string.ascii_uppercase:
    with open(letter + ".txt", "w") as f:
        f.writelines(letter)
        
