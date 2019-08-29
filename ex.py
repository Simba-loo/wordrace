import argparse
import numpy as np
from numpy import linalg as la
import sys
import pandas as pd
import math
import scipy
import pdb
import re
from sklearn.neighbors import NearestNeighbors

n_neighbors = 10


def parse_vocab(path):
    fp = open(path, 'r')
    vocab = []
    line = 'a'
    while line:
        line = fp.readline()
        vocab.append(line[0:-1])
    return (vocab)


def parse_data(path, d, vocab):
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
            word = re.search('[a-zA-Z]*', line).group(0)
            vector_str = re.search('[a-zA-Z]* (.*)', line).group(1).split(' ')
        except:
            continue
        #print(word)
        #print(line)
        #print(vector_str)

        if len(vector_str) != d:
            continue

        vector = [float(e) for e in vector_str]
        if word in vocab:
            words.append(word)
            vectors.append(vector)
            vec_dict[sum(vector)] = word
            word_dict[word] = vector
            count += 1
    print("Number of words in database: " + str(count))
    fp.close()
    return (words, vectors, vec_dict, word_dict)


vocab_path = 'nouns.txt'
path = '20k50d.csv'
d = 50

vocab = parse_vocab(vocab_path)

words, vectors, vec_dict, word_dict = parse_data(path, d, vocab)
neigh = NearestNeighbors(8, 1.0)
neigh.fit(vectors)


def dist(word1, word2):
    vec1 = np.array(word_dict[word1])
    vec2 = np.array(word_dict[word2])
    diff = vec1 - vec2
    return la.norm(diff)


# print("Distance from guitar to war: " + str(dist("guitar", "war")))

num_steps = 0
src_word = "guitar"
dst_word = "building"
history = [src_word]

debug_mode = True

while True:
    word = input("Pick a word: ")
    #word = 'car'
    try:
        vector = word_dict[word]
    except:
        print("The word \"" + word +
              "\" is not in our dictionary, try another one.")
        continue
    print("Distance to " + dst_word + ": " + str(dist(word, dst_word)))
    history.append(word)
    distances, index_vec = neigh.kneighbors([vector], n_neighbors)
    res = [words[i] for i in index_vec[0][1:n_neighbors - 1]]
    if debug_mode:
        print("Pick between:\n" + "".join([
            "\t" + w + " (" + str(dist(w, dst_word)) + ")\n"
          for w in res]))
    else:
        print("Pick between:\n" + "".join(["\t" + str(w) + "\n" for w in res]))
