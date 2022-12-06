from embeddings.Synonyms import GetSynonyms
from transformers import BertTokenizer
from config import *


def get_sentence_with_synonyms(synonyms, sent):
    """
    gives a list of synonyms from a sentence
    """
    sentence_list = sent.split(" ")
    copy_sentence = sentence_list.copy()
    counter = 0
    for word in copy_sentence:
        counter += 1
        list_synonyms = synonyms.get_lex_representations_without_himself(word)
        sentence_list[counter:counter] = list_synonyms

    return sentence_list


def divide_words_in_sentence(tokenizer, sent):
    """
    returns a tokenized sentence and makes sure that target sign isn't tokenized
    """
    list_with_dividing = tokenizer.tokenize(sent)
    sentence = ' '.join(list_with_dividing)
    if '$ t $' in sentence:
        sentence2 = sentence.replace('$ t $', '$T$')
    else:
        sentence2 = sentence
    return sentence2


class TestData:

    # determine line where test data starts
    number = 0
    if FLAGS.year == 2016:
        number = 5640
    elif FLAGS.year == 2015:
        number = 3864

    count = 0
    counter2 = -1
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenize the raw data
    # and append synonyms on the test part of the data
    with open('data/externalData/' + 'raw_data' + str(FLAGS.year) + '.txt', 'r') as raw_data:
        line_list = raw_data.readlines()
        with open('data/temporaryData/' + 'data' + str(FLAGS.year) + '.txt', 'w') as test_data:
            # append the raw training data with synonyms and tokenization
            for i in range(0, number):
                if count % 3 == 0 or counter2 % 3 == 0:
                    sentence_with_dividing = divide_words_in_sentence(tokenizer, line_list[i])
                    test_data.write(sentence_with_dividing + '\n')
                else:
                    test_data.write(''.join(line_list[i]))
                count += 1
                counter2 += 1
            count = 0
            counter2 = 2
            for i in range(number, len(line_list)):
                sentence = line_list[i]
                sentence = sentence.replace('\n', '')
                if count % 3 == 0 or counter2 % 3 == 0:
                    synonym = GetSynonyms()
                    sentence_with_synonyms = get_sentence_with_synonyms(synonym, sentence)
                    sentence2 = ' '.join(sentence_with_synonyms)
                    sentence_with_synonyms_and_dividing = divide_words_in_sentence(tokenizer, sentence2)
                    test_data.write(sentence_with_synonyms_and_dividing + '\n')
                else:
                    test_data.write(''.join(sentence) + '\n')
                count += 1
                counter2 += 1

    all_whole_sentence = []
    count = -1
    count2 = 1
    # make the tokens in the data unique
    with open('data/temporaryData/' + 'data' + str(FLAGS.year) + '.txt', 'r') as test_data1:
        line_list = test_data1.readlines()
        for i in range(0, len(line_list)):
            sentence_list = line_list[i].split(" ")
            sentence_list_without_next_line = []
            for j in range(0, len(sentence_list)):
                word = sentence_list[j]
                if '\n' in word:
                    word2 = word.replace('\n', '')
                else:
                    word2 = word
                sentence_list_without_next_line.append(word2)
            all_whole_sentence.append(sentence_list_without_next_line)
        with open('data/temporaryData/'+'data'+str(FLAGS.year)+'_unique.txt', 'w') as test_data_unique:
            dic_words = {}
            for i in range(0, len(all_whole_sentence)):
                count += 1
                count2 += 1
                if count % 3 == 0 or count2 % 3 == 0:
                    for j in range(0, len(all_whole_sentence[i])):
                        if all_whole_sentence[i][j] == '$T$':
                            pass
                        elif not all_whole_sentence[i][j] in dic_words:
                            dic_words[all_whole_sentence[i][j]] = 0
                        else:
                            past_value = dic_words[all_whole_sentence[i][j]]
                            dic_words[all_whole_sentence[i][j]] = past_value + 1

                        if not all_whole_sentence[i][j] == '$T$':
                            all_whole_sentence[i][j] = all_whole_sentence[i][j] + '_' + str(dic_words[all_whole_sentence[i][j]])

                    sent = ' '.join(all_whole_sentence[i])
                    test_data_unique.write(sent + '\n')
                else:
                    test_data_unique.write(line_list[i])

    # split data in train data and test data
    with open('data/temporaryData/' + 'data'+str(FLAGS.year)+'_unique.txt', 'r') as test_data:
        line_list = test_data.readlines()
        with open('data/programGeneratedData/' + 'train_data' + str(FLAGS.year) + '.txt', 'w') as train_data:
            for i in range(0, number):
                train_data.write(line_list[i])

    with open('data/temporaryData/' + 'data' + str(FLAGS.year) + '_unique.txt', 'r') as td:
        line_list = td.readlines()
        with open('data/programGeneratedData/' + 'test_data' + str(FLAGS.year) + '.txt', 'w') as test_data:
            for i in range(number, len(line_list)):
                test_data.write(line_list[i])

    # format test data for making embeddings
    with open('data/programGeneratedData/' + 'test_data' + str(FLAGS.year) + '.txt', 'r') as fin:
        line_list = fin.readlines()
        target_list = []
        for i in range(1, len(line_list), 3):
            target_list.append(line_list[i])
        with open('data/temporaryData/' + 'test_data' + str(FLAGS.year) + '_sentences.txt', 'w') as test_d:
            for i in range(0, len(line_list), 3):
                sentence2 = line_list[i].replace('$T$', line_list[i+1].replace('\n', ''))
                test_d.write(sentence2)
