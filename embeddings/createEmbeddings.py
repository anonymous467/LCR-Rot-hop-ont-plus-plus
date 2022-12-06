import numpy as np
from Synonyms import GetSynonyms
from transformers import BertTokenizer, BertModel
import bert_encoder
import torch
import argparse
from config2 import load_hyperparam

soft_positions_untokenized = []  # list with list of softpositions from untokenized sentences
new_soft_positions = []
tokens_sentences = []
segments = []
visible_matrices = []
is_original = []
new_originals = []
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
model.eval()
string_year = "2015"
number = 0
if string_year == "2015":
    number = 3864
elif string_year == "2016":
    number = 5640

sofpos = True  # set on False, for results with ordinary softpositions
use_vm = True  # set on False, for results without using the visible matrix


def get_sentence_with_synonyms(synonyms, sent):
    """
    input: synonym object, sentence string
    proces: while appending the synonyms, the softpositions and originality are also determined
    output: list of sentence with synonyms appended
    """
    sentence_list = sent.split(" ")  # make list of words in sentence
    copy_sentence = sentence_list.copy()  # make copy of sentence list

    # initialize counters and lists
    counter = 0
    cnt = 0
    original = []
    soft_position = []

    # append synonyms for every word in the sentence (if available)
    for word in copy_sentence:
        counter += 1
        original.append(1)
        if synonyms is None:
            pass
        else:
            list_synonyms = synonyms.get_lex_representations_without_himself(word)
            sentence_list[counter:counter] = list_synonyms
            counter += len(list_synonyms)

        soft_position.append(cnt)  # append softposition of the word

        if synonyms is None:
            pass
        else:
            for i in range(0, len(" ".join(list_synonyms).split(" "))):
        
                if " ".join(list_synonyms).split(" ")[0] == '':  # if no synonyms
                    pass
                else:                           # if there are synonyms
                    soft_position.append(cnt)   # synoyms get the same softposition as the original word
                    original.append(0)          # synoyms are not original, so append a 0 to the original list
        cnt += 1

    # append original and soft position list to
    is_original.append(original)
    soft_positions_untokenized.append(soft_position)

    # return the list with synonyms appended
    return sentence_list


def divide_words_in_sentence(number_sentence, tokenizer, sent):
    """
    tokenize each word and determine new softpositions and originality
    """
    # initialize the new soft position list and the orginal list
    new_soft_position = []
    new_original = []
    if number_sentence is not None:
        for i in range(0,len(sent.split(" "))):
            tok = tokenizer.tokenize(sent.split(" ")[i]) # slit
            pos = soft_positions_untokenized[number_sentence][i]
            ori = is_original[number_sentence][i]
            for j in tok:
                new_original.append(ori)
                new_soft_position.append(pos)

        new_originals.append(new_original)
        new_soft_positions.append(new_soft_position)
        list_with_dividing = tokenizer.tokenize(sent)

        # initialize counters
        count1 = 0
        count = 0

        for i in range(0,len(new_soft_positions[number_sentence])): # makes softpositions unique for tokens for original word and not unique for tokens of synonyms
            if new_original[i]== 1:
                count = 0
                new_soft_positions[number_sentence][i] = count1
                count1 += 1
            elif new_original[i] == 0:
                count+=1
                new_soft_positions[number_sentence][i] = i - count

        sentence = ' '.join(list_with_dividing)  # make sentence out of sentence list

        tokens_sentences.append(sentence.split(" ")) # append sentence to list tokens list
        
    else:
        list_with_dividing = tokenizer.tokenize(sent)
        sentence = ' '.join(list_with_dividing)
        
    # replace tokenized target in sentence with original target
    if '$ t $' in sentence:
        sentence2 = sentence.replace('$ t $', '$T$')
    else:
        sentence2 = sentence
    
    # return tokenized sentence
    return sentence2

def makeSegments():
    """
    determines the segment id's
    """
    for sentence in tokens_sentences: # itereert over alle zinnen, dan over alle tokens en zet segment id's neer en stuurt lijst naar een grotere lijst
        seg = []
        s_count = 0

        for token in sentence:

            if s_count  == 0 or s_count %2 == 0:
                seg.append(0) #can change it to zero or 1
            elif s_count == 1 or s_count %2 != 0:
                seg.append(1)
            #if token == "-" or token=='–': #remove hashes to alternate between segments for minus sign or dash
                #s_count += 1

        # append segment id to segments list
        segments.append(seg)

def makeVisibleMatrices():
    """
    makes for each sentence the visible matrix
    """
    for positions,original in zip(new_soft_positions,new_originals):
        visible_matrices.append(get_visible_matrix(positions,original))

def get_visible_matrix(positions,original):
        """
        gets the visible matrix of a sentence tree
        """
        # initialize the visible matrix
        visible_matrix = np.zeros((1,len(positions), len(positions)))

        # make up the visible matrix
        for i in range(0,len(positions)):
            for j in range(0, len(positions)):
                visible_matrix[0,i, j] = float(determine_visibility(original,positions,i, j))

        # return the visible matrix
        return visible_matrix


def determine_visibility(original,positions, row_number, column_number):
    """
    method that determines if a row number and column number can see each other in the sentence tree
    """
    result = 0
    row_is_first_occurrence = is_first_occurrence(original, positions, row_number)
    column_is_first_occurrence = is_first_occurrence(original, positions, column_number)
    if not row_is_first_occurrence or not column_is_first_occurrence:
        result = -10000.0           # we have chosen for -10000 instead of -inf, just like the k-bert code
    if same_soft_position(positions, row_number, column_number):
        result = 0

    return result


def same_soft_position(positions, position1, position2):
    """
    method that determines if two soft positions are the same or not
    """
    if positions[position1] == positions[position2]:
        return True

    return False


def is_first_occurrence(original, positions,number):
    """
    method that determines if the soft position is the first occurrence in the sentence tree
    """
    soft_position_before = -1
    soft_position_number = positions[number]
    if number != 0:
        soft_position_before = positions[number - 1]
    if soft_position_before == soft_position_number and original[number] == 0:
        return False

    return True


class Embeddings:

    list_of_synonyms = GetSynonyms()

# tijdens het synoniemen toevoegen en tokenizen ook soft positions maken en in een lijst opslaan of gelijk elke zin een tensor geven met de juiste embeddings.
# zelfde geld voor de segments en tokens, dus hieronder alle embeddings geven, de visible matrix uitrekenen en alles in tensors zetten.

    count = 0
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with open('data/externalData/raw_data/' + string_year + '.txt', 'r') as raw_data:
        line_list = raw_data.readlines()
        with open('data/temporaryData/' + 'data_temp ' +string_year + '.txt', 'w') as data:
            # append the raw training data
            sentence_number = 0
            for i in range(0, number):
                sentence = line_list[i]
                if count % 3 == 0:  # if it is a sentence line
                    # add CLS and SEP and remove target sign
                    sentence = "[CLS] " + line_list[i].replace('$T$', line_list[i+1].replace('\n', '')) + " [SEP]"
                    # add no synonyms (because synonyms equals None)
                    sentence_without_synonyms = get_sentence_with_synonyms(None, sentence)
                    sentence2 = ' '.join(sentence_without_synonyms)
                    # tokenize sentence
                    sentence_with_dividing = divide_words_in_sentence(sentence_number,tokenizer, sentence2)
                    data.write(sentence_with_dividing + '\n')
                    sentence_number += 1
                else:  # if it is a target line or sentiment line
                    sentence_with_dividing = divide_words_in_sentence(None, tokenizer, sentence)
                    data.write(sentence_with_dividing + '\n')
                count += 1
            # append the raw test data and add synonyms
            for i in range(number, len(line_list)):
                sentence = line_list[i]
                if count % 3 == 0: # if it is a sentence line
                    # add CLS and SEP and remove target sign
                    sentence = "[CLS] " + line_list[i].replace('$T$', line_list[i+1].replace('\n', '')) + " [SEP]"
                    # add synonyms to sentence
                    synonym = GetSynonyms()
                    sentence_with_synonyms = get_sentence_with_synonyms(synonym, sentence)
                    sentence2 = ' '.join(sentence_with_synonyms)
                    # tokenize sentence
                    sentence_with_synonyms_and_dividing = divide_words_in_sentence(sentence_number,tokenizer, sentence2)
                    data.write(sentence_with_synonyms_and_dividing + '\n')
                    sentence_number += 1
                else: # if it is a target line or sentiment line
                    sentence_with_dividing = divide_words_in_sentence(None, tokenizer, sentence)
                    data.write(sentence_with_dividing + '\n')
                count += 1

embeddings = []
visibleMatrices = []


def makeEmbeddings():
    """
    makes the embeddings
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    for i in range(1800,len(tokens_sentences)):# can change range to less strain the pc
        tensor = torch.zeros((1,len(new_soft_positions[i]),768))
        token_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(tokens_sentences[i])])
        segment_tensor = torch.tensor([segments[i]])
        pos_tensor = torch.tensor([new_soft_positions[i]])
        if sofpos:
            output = model(token_tensor,None,segment_tensor,pos_tensor)
        else:
            output = model(token_tensor,None,segment_tensor,None)
        tensor = output.hidden_states.__getitem__(00)

        embeddings.append(tensor)
        visibleMatrices.append(torch.tensor(visible_matrices[i]))
        print(i)
        if i==0:
            print(segments[i])
            print(tokens_sentences[i])
            print(new_soft_positions[i])


makeSegments()  # after all the tokens are appended, make segments
makeVisibleMatrices() # make the visible matrices of the sentences
makeEmbeddings() # make the embeddings

# give all words in sentence and target unique count
all_whole_sentence = []
all_counted_tokens = []
count = -1
count2 = 1
with open('data/temporaryData/' + 'data_temp ' +string_year + '.txt', 'r') as test_data1:
    line_list = test_data1.readlines()
    for i in range(0, len(line_list)):
        sentence_list = line_list[i].split(" ")
        # remove the /n from certain words
        sentence_list_without_next_line = []
        for j in range(0, len(sentence_list)):
            word = sentence_list[j]
            if '\n' in word:
                word2 = word.replace('\n', '')
            else:
                word2 = word
            sentence_list_without_next_line.append(word2)
        all_whole_sentence.append(sentence_list_without_next_line)
    with open('data/temporaryData/' + 'data_temp ' + string_year + '_unique.txt', 'w') as data_unique:
        dic_words = {}
        for i in range(0, 5640):
            count += 1
            count2 += 1
            if count % 3 == 0 :
                for j in range(0, len(all_whole_sentence[i])):
                    if all_whole_sentence[i][j] =='$T$' or  all_whole_sentence[i][j] =='[SEP]' or all_whole_sentence[i][j] =="[CLS]":
                        pass
                    elif not all_whole_sentence[i][j] in dic_words:
                        dic_words[all_whole_sentence[i][j]] = 0
                    else:
                        past_value = dic_words[all_whole_sentence[i][j]]
                        dic_words[all_whole_sentence[i][j]] = past_value + 1

                    if not all_whole_sentence[i][j] == '$T$' and all_whole_sentence[i][j] !='[SEP]' and all_whole_sentence[i][j] !="[CLS]":
                        all_whole_sentence[i][j] = all_whole_sentence[i][j] + '_' + str(dic_words[all_whole_sentence[i][j]])
                all_counted_tokens.append(all_whole_sentence[i])
                sent = ' '.join(all_whole_sentence[i])
                data_unique.write(sent + '\n')
            else:
                data_unique.write(line_list[i])
        for i in range(5640,len(all_whole_sentence)):
            count += 1
            count2 += 1
            if count % 3 == 0:
                for j in range(0, len(all_whole_sentence[i])):
                    if all_whole_sentence[i][j] == '$T$'or all_whole_sentence[i][j] =='[SEP]' or all_whole_sentence[i][j] =="[CLS]":
                        pass
                    elif not all_whole_sentence[i][j] in dic_words:
                        dic_words[all_whole_sentence[i][j]] = 0
                    else:
                        past_value = dic_words[all_whole_sentence[i][j]]
                        dic_words[all_whole_sentence[i][j]] = past_value + 1

                    if not all_whole_sentence[i][j] == '$T$'and all_whole_sentence[i][j] !='[SEP]' and all_whole_sentence[i][j] !="[CLS]":
                        all_whole_sentence[i][j] = all_whole_sentence[i][j] + '_' + str(dic_words[all_whole_sentence[i][j]])
                all_counted_tokens.append(all_whole_sentence[i])
                sent = ' '.join(all_whole_sentence[i])
                data_unique.write(sent + '\n')
            else:
                data_unique.write(line_list[i])


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config_path", default="./google_config.json", type=str,help="Path of the config file.")
args = parser.parse_args()
args = load_hyperparam(args) # Load the hyperparameters from the config file.

encoder = bert_encoder.BertEncoder(args, model) # make object of bert_encoder

tokens = all_counted_tokens # list of list of all tokens for a sentence
vm = visibleMatrices # a list of visible matrices for each sentence in tensor-shape of 1*token_numb*token_numb.
Embeddings = embeddings # a list with intial embeddings for each token in tensor-shape of 1*token_numb*768
hidden_states = []  # list of hidden states
token_hidden_states = [] # list with for each token, the token and the hidden states

count = 0
for i in range(1800,len(tokens_sentences)):
    tensor = torch.zeros((1,len(new_soft_positions[i]),len(new_soft_positions[i])))
    if use_vm:
        # calculate all hidden states for all tokens in a sentence
        hidden = encoder.forward(Embeddings[count], None, vm[count])
    else:
        # calculate all hidden states for all tokens in a sentence, without visible matrix
        hidden = encoder.forward(Embeddings[count], None, tensor)
    hidden_states.append(hidden)
    print(i)
    count += 1

counter = 0
for j in range(0,len(tokens_sentences)):
    print( j)
    token_count = 0
    for token in tokens[j]: # iterate over all tokens in a sentence
        if token == "[CLS]" or token == "[SEP]":
            token_count += 1
        else:
            list_of_embeddings = hidden_states[counter][0][token_count].tolist() # make a list of the embedding per token
            token_count += 1 # count the tokens
            string_list_of_embeddings = [str(i) for i in list_of_embeddings]  # convert numbers to strings
            string_list_of_embeddings.insert(0, token)  # append the token in the front of the list
            token_hidden_states.append(string_list_of_embeddings)  # append the embedding for a word to the big list
    counter += 1

indicator_sentence = '_normal'
if not use_vm and not sofpos:
    indicator_sentence = "_without_VM_and_softpos"
elif not use_vm:
    indicator_sentence = "_without_VM"
elif not sofpos:
    indicator_sentence = "_without_softpos"

with open('data/programGeneratedData/' + 'allEmbeddings' + string_year + indicator_sentence + '.txt','w') as outf:
    for c in token_hidden_states:
        print(" ".join(c), file= outf)       #print all embeddings to txt file
    outf.close()
