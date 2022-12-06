# https://github.com/mtrusca/HAABSA_PLUS_PLUS

from dataReader2016 import read_data_2016
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random

def loadDataAndEmbeddings(config,loadData):

    FLAGS = config

    if loadData == True:
        source_count, target_count = [], []
        source_word2idx, target_phrase2idx = {}, {}

        print('reading training data...')
        train_data = read_data_2016(FLAGS.train_data, source_count, source_word2idx, target_count, target_phrase2idx, FLAGS.train_path)
        print('reading test data...')
        test_data = read_data_2016(FLAGS.test_data, source_count, source_word2idx, target_count, target_phrase2idx, FLAGS.test_path)

        wt = np.random.normal(0, 0.05, [len(source_word2idx), 300])
        word_embed = {}
        count = 0.0
        with open(FLAGS.pretrain_file, 'r',encoding="utf8") as f:
            for line in f:
                content = line.strip().split()
                if content[0] in source_word2idx:
                    wt[source_word2idx[content[0]]] = np.array(list(map(float, content[1:])))
                    count += 1
                    
        print('finished embedding context vectors...')

        #print data to txt file
        outF= open(FLAGS.embedding_path, "w")
        for i, word in enumerate(source_word2idx):
            outF.write(word)
            outF.write(" ")
            outF.write(' '.join(str(w) for w in wt[i]))
            outF.write("\n")
        outF.close()
        print((len(source_word2idx)-count)/len(source_word2idx)*100)
        
        return train_data[0], test_data[0], train_data[4], test_data[4]

    else:
        #get statistic properties from txt file
        train_size, train_polarity_vector = getStatsFromFile(FLAGS.train_path)
        test_size, test_polarity_vector = getStatsFromFile(FLAGS.test_path)

        return train_size, test_size, train_polarity_vector, test_polarity_vector

def loadAverageSentence(config,sentences,pre_trained_context):
    FLAGS = config
    wt = np.zeros((len(sentences), FLAGS.edim))
    for id, s in enumerate(sentences):
        for i in range(len(s)):
            wt[id] = wt[id] + pre_trained_context[s[i]]
        wt[id] = [x / len(s) for x in wt[id]]

    return wt

def getStatsFromFile(path):
    polarity_vector= []
    with open(path, "r") as fd:
        lines = fd.read().splitlines()
        size = len(lines)/3
        for i in range(0, len(lines), 3):
            #polarity
            polarity_vector.append(lines[i + 2].strip().split()[0])
    return size, polarity_vector
