import argparse
import numpy as np
import sys
import pandas as pd
import math
import scipy
import pdb
import re
from sklearn.neighbors import NearestNeighbors

n_neighbors = 10

def parse_vocab(path):
  fp = open(path,'r')
  vocab = []
  line = 'a'
  while line:
    line = fp.readline()
    vocab.append(line[0:-1])
  return(vocab)


def parse_data(path,d, vocab):
    fp = open(path, 'r')
    vectors = []
    words = []
    vec_dict = {}
    word_dict = {}
    line = '0'
    count = 0
    while line:
       try:
        line = fp.readline()
       except:
        continue
       try:
        word = re.search('[a-zA-Z]*',line).group(0)
        vector_str = re.search('[a-zA-Z]* (.*)',line).group(1).split(' ')
       except:
        continue
       #print(word)
       #print(line)
       #print(vector_str)

       if len(vector_str)!=d:
        continue

       vector = [float(e) for e in vector_str]
       if word in vocab:
        words.append(word)
        vectors.append(vector)
        vec_dict[sum(vector)] = word
        word_dict[word] = vector
        count+=1
    print(count)
    fp.close()
    return(words,vectors,vec_dict,word_dict)

vocab_path = 'nouns.txt'
path = '20k50d.csv'
d = 50

vocab = parse_vocab(vocab_path)

words,vectors,vec_dict,word_dict = parse_data(path,d,vocab)
neigh = NearestNeighbors(5, 1.0)
neigh.fit(vectors)
print(vocab)
while True:
    word = input("Pick a word:")
    #word = 'car'
    try:
        vector = word_dict[word]
    except:
        continue
    index_vec = neigh.kneighbors([vector], n_neighbors, return_distance=False)
    res = [words[i] for i in index_vec[0][1:n_neighbors-1]]
    print(res)
