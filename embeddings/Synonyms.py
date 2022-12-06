from owlready import *
from owlready2 import *


class GetSynonyms:
    def __init__(self):
        onto_path.append("data/externalData")  # Path to ontology
        self.onto = get_ontology("ontology.owl")  # Name of ontology
        self.onto = self.onto.load()
        self.classes = set(self.onto.classes())
        self.my_dict = {}
        self.list_of_list_of_synonyms = []

        # make dictionary with a list of synonyms as values
        for onto_class in self.classes:
            self.my_dict[onto_class] = onto_class.lex

        # make a list of the keys in the dictionary
        list_of_keys = [*self.my_dict]

        # for every list of synonyms, append the list to the big list
        for i in range(len(self.my_dict)):
            # make list of synonyms of place i
            key_of_synonym_list = list_of_keys[i]
            list_syn = self.my_dict[key_of_synonym_list]
            # append list of synonyms to the big list
            self.list_of_list_of_synonyms.append(list_syn)

    def get_lex_representations(self, word):
        '''
        returns a list of all the lexical respresentations of a word
        '''
        for synonym_list in self.list_of_list_of_synonyms:
            for w in synonym_list:
                if w == word:
                    return synonym_list

        # return empty list if word is not in the list
        return []

    def get_lex_representations_without_himself(self, word):
        '''
        returns a list of all the lexical respresentations of a word without himself
        '''
        result_list = self.get_lex_representations(word).copy()
        if len(result_list) != 0:
            result_list.remove(word)

        return result_list

    def get_lex_representations_uppercase_accepted(self, word):
        '''
        returns a list of all the lexical respresentations of a word, where uppercase input is accepted
        '''
        return self.get_lex_representations(word.lower())

    def get_lex_representations_without_himself_uppercase_accepted(self, word):
        '''
        returns a list of all the lexical respresentations of a word without himself and accepts
        uppercase words
        '''
        result_list = self.get_lex_representations_uppercase_accepted(word).copy()
        if len(result_list) != 0:
            result_list.remove(word.lower())

        return result_list

