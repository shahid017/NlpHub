#!/usr/bin/env python
# coding: utf-8

# In[7]:



# coding: utf-8

# In[1]:


import spacy
import nltk
from nltk.stem import SnowballStemmer
import string
from textblob import TextBlob
from spacy_langdetect import LanguageDetector
from spacy_readability import Readability
import operator
import re
import pandas as pd
import os
import json
import time
import pdfminer
from fnmatch import fnmatch
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from matplotlib import pyplot as plt
stemmer = SnowballStemmer('english')
try:
    import en_core_web_md
    nlp = en_core_web_md.load()
except OSError:
    print("Downloading language model for the spaCy POS tagger do not worry, this will only happen once")
    os.system('python -m spacy download en_core_web_md')
import en_core_web_md
nlp = en_core_web_md.load()
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
extra_words = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself',
              'yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself',
              'they','them','their','theirs','themselves','what','which','who','whom','this','that',
              'these','those','am','is','are','was','were','be','been','being','have','has','had',
              'having','do','does','did','doing','a','an','the','and','but','if','or','because','as',
              'until','while','of','at','by','for','with','about','against','between','into','through',
              'during','before','after','above','below','to','from','up','down','in','out','on','off',
              'over','under','again','further','then','once','here','there','when','where','why','how',
              'all','any','both','each','few','more','most','other','some','such','no','nor','not',
              'only','own','same','so','than','too','very','s','t','can','will','just','don','should',
              'now']


# In[16]:



class NlpHub():
    '''This class contains all NLP operations you can perform using NlpHub.
    '''

    def __init__(self, input_str):
        self.input_str = input_str


    def lemma(self):
        '''
        Lemmatization function accepts a string and return the lemmatized version of that input string.
        This function uses Spacy.

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.

        Return
        ----------
        type : str
            returns a lemmatized string,
        '''
        try:
            doc = nlp(self)
            sent = ""
            for token in doc:
                if token.lemma_ == '-PRON-':
                    sent += ' '+ token.text
                else:
                    sent += ' '+ token.lemma_
            return sent.lstrip()
        except:
            raise Exception("Something bad happend. Make sure input is a string.")


    def stem(self):
        '''
        Stem function accepts a string and return the stemmed version of that input string.
        This function uses Spacy.
        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.

        Return
        ----------
        type : str
            returns a stemmed string,
        '''
        stemmed_sent = []
        for w in self.split(" "):
            stemmed_sent.append("".join(stemmer.stem(w)))
        sent = " ".join(stemmed_sent)
        return sent


    def capitalize(self):
        '''
        Capitalize function accepts a string and return the capitalized version of that input string.
        This function uses String library method capitalize.
        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.

        Return
        ----------
        type : str
            returns a capitalized string,
        '''
        rs = self.capitalize()
        return rs


    def case_change(self, case='lower'):
        '''
        Case Change function accepts a string and case and return the case Changed version of that input string.
        This function uses String library method upper and lower.
        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.

        type: str
            This argument is used to define the case you want. Acceptable values are lower and upper.

        Return
        ----------
        type : str
            returns a case Changed string,
        '''
        out_str = ""
        if case=='upper':
            out_str = self.upper()
        elif case=='lower':
            out_str = self.lower()
        return out_str


    def filter_alpha(self):
        '''
        Filter Alphabet function accepts a string and return the filtered version of that input string.
        This function uses String library method isalpha. This method removes any words containing digits etc.
        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.

        Return
        ----------
        type : str
            returns a filtered string,
        '''
        sent = self
        word_list = []
        if len(sent.split()) > 1:
            words = sent.split()
            for w in words:
                if w.isalpha() == True:
                    word_list.append(w)
            sentence = " ".join(word_list)
        return sentence


    def find(self, tofind):
        '''
        Find function accepts a string and and a string to find return the index of match in input string.
        This function uses String library method find.
        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.

        type: str
            This argument is used to define the string you want to find in given string.

        Return
        ----------
        type : int
            returns a index in string,
        '''
        sent = self
        return sent.find(tofind)


    def remove_punct(self):
        '''
        Remove Punctuation function accepts a string and return the version of that input string with no puncts.
        This function uses String punctuations. This method removes any punctuation from string.
        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.

        Return
        ----------
        type : str
            returns a filtered string,
        '''
        sent = self
        no_punct = ""
        for char in sent:
            if char not in punctuations:
                no_punct = no_punct + char
        return no_punct


    def remove_sw(self, sw_list=[]):
        '''
        Remove Stopwords function accepts a string and a optional stopword list  and return the string without any stopwords.
        This functions uses String stopword list and custom added stopwords.
        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.
        sw_list : list
            This argument is optional and if you want to use your own list, you can pass a list of stopwords.

        Return
        ----------
        type : str
            returns a string without stopwords,
        '''
        if sw_list:
            temp_split_arr = self.split(' ')
            no_stop_word_list = [el for el in temp_split_arr if el.lower().rstrip() not in sw_list]
            sent = " ".join(no_stop_word_list)
        else:
            temp_split_arr = self.split(' ')
            no_stop_word_list = [el for el in temp_split_arr if el.lower().rstrip() not in extra_words]
            sent = " ".join(no_stop_word_list)
        return sent


    def replace(self, to_replace, replace_by):
        '''
        Replace function accepts a string, a string to replace and a string to replace by and return the string after replacing the intended words.
        This functions uses String method replace.
        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.
        to_replace : str
            The word to you to replace in string.
        replace_by : str
            The words you want to put in the string.

        Return
        ----------
        type : str
            returns a string after replacing,
        '''
        sent = self
        return_sent = sent.replace(to_replace, replace_by)
        return return_sent


    def split(self, split_on):
        '''
        Split function accepts a string and a string to split and return the string after split it on given parameter.
        This functions uses String method split.
        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.
        split_on : str
            The word to you to replace in string.

        Return
        ----------
        type : list
            returns a list of strings after spliting it in given parameter,
        '''
        sent = self
        result = sent.split(split_on)
        if len(result) > 1:
            return result


    def strip(self, strip_it):
        '''
        Strip function accepts a string and a string to strip and return the string after stripping given parameter from left and right.
        This functions uses String method strip.
        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.
        strip_it : str
            The word/char you to strip from string.

        Return
        ----------
        type : str
            returns a string after stripping given parameter,
        '''
        sent = self
        if len(sent.split()) > 1:
            words = sent.split()
            new_sent = []
            for w in words:
                left_strip_sent = w.strip(strip_it)
                new_sent.append(left_strip_sent)
            return_Sent = " ".join(new_sent)
        return return_Sent


    def sentiment(self):
        '''
        Sentiment function accepts a string and do sentiment analysis using textblob library.

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.

        Return
        ----------
        type : str
            returns a sentiment of String Positive, Negative or Neutral,
        '''
        sentence = TextBlob(self)
        sentiment = sentence.sentiment[0]
        if sentiment < -0.3:
            sent = "Negative"
        elif sentiment > 0.3:
            sent = "Positive"
        else:
            sent = "Neutral"
        return sent


    def dependency(self):
        '''
        Dependency function accepts a string and returns dependency dictionary with words as keys and their
        dependency as values. Uses Spacy library

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.

        Return
        ----------
        type : dictionary
            returns a dict,
        '''
        doc = nlp(self)
        dep = {}
        for token in doc:
            dep[token.text] = token.dep_
        return dep


    def language(self):
        '''
        language function accepts a string and return the language of string using Spacy library.

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.

        Return
        ----------
        type : str
            returns a language like en for English or sp for Spanish,
        '''
        nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
        doc = nlp(self)
        lang = doc._.language
        return lang['language']


    def named_entity(self):
        '''
        Named Entity function accepts a string and returns Named Entity dict with words as keys and their
        entity as values. Uses Spacy library

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.

        Return
        ----------
        type : dictionary
            returns a dict,
        '''
        doc = nlp(self)
        ner = {}
        for ent in doc.ents:
            ner[ent.text] = ent.label_
        return ner


    def pos_tagging(self):
        '''
        Parts of Speech Tagging function accepts a string and returns Parts of Speech dict with words as keys and their
        pos as values. Uses Spacy library

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.

        Return
        ----------
        type : dictionary
            returns a dict,
        '''
        doc = nlp(self)
        pos = {}
        for token in doc:
            pos[token.text] = token.pos_
        return pos


    def readability_score(self):
        '''
        Readability Score function accepts a string and returns flesch kincaid Readability Score dict. Uses Spacy library

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.

        Return
        ----------
        type : dictionary
            returns a dict,
        '''
        nlp.add_pipe(Readability())
        main_dict = {}
        doc = nlp(self)
        main_dict["textGradeLevel"] = doc._.flesch_kincaid_grade_level
        main_dict["textReadingEase"] = doc._.flesch_kincaid_reading_ease
        return main_dict


    def similar_word(self, match_word):
        '''
        Similar Word function accepts a string and a word to match and returns a string with maximum similarity.
        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.

        Return
        ----------
        type : str
            returns a str,
        '''
        word_to_check_tok = nlp(match_word)
        text_string = self
        formatted_article_text = re.sub('[^a-zA-Z]', ' ', text_string )
        formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
        doc = nlp(formatted_article_text)
        token_dict = {}
        for token1 in doc:
            if token1.vector_norm:
                token_dict[token1.text] = float(token1.similarity(word_to_check_tok[0]))
        if len(token_dict) >=1:
            max_key = max(token_dict.items(), key=operator.itemgetter(1))[0]
        return max_key


    def text_similarity(self, match_str):
        '''
        Text Similarity function accepts two string and returns similarity.
        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.

        Return
        ----------
        type : float
            returns a flaot number,
        '''
        X = self
        Y = match_str
        x_list = X.split()
        y_list = Y.split()
        x_set = {w for w in x_list if not w in extra_words}
        y_set = {w for w in y_list if not w in extra_words}
        list_x =[]
        list_y =[]
        return_vector = x_set.union(y_set)
        for w in return_vector:
            if w in x_set: list_x.append(1)
            else: list_x.append(0)
            if w in y_set: list_y.append(1)
            else: list_y.append(0)
        c = 0
        for i in range(len(return_vector)):
            c+= list_x[i]*list_y[i]
        if sum(list_x) == 0 or sum(list_y) == 0:
            similarity = 0
        else:
            cosine = c / float((sum(list_x)*sum(list_y))**0.5)
            similarity = cosine
        return similarity


    def clustering(self, cluster_list):
        '''
        Clustering function accepts two arguments, a string and list of strings and returns a list of dicts.
        Clustering computes similarity of each word in string with each cluster word in the cluster_list
        and add word to cluster with maximum similarity.

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.
        cluster_list : List
            This argument accepts a list containing strings as cluster names

        Return
        ----------
        type : list
            returns a list of dicts,
        '''
        tokens = nlp(self)
        cluster_dict = {}
        for i in range(len(cluster_list)):
            cluster_dict["cluster"+str(i)] = cluster_list[i]
        key_list = list(cluster_dict.keys())
        final_dict = {}
        for token in tokens:
            if token.vector_norm:
                values  = {}
                if len(token.text) > 1:
                    for key in key_list:
                        word = nlp(cluster_dict[key])
                        values[key]  = token.similarity(word)
                    max_value_cluster = max(values.items(), key=operator.itemgetter(1))[0]
                    token_word = token.text
                    final_dict[token_word] = max_value_cluster
        final_list = []
        for ke in key_list:
            clu = {}
            cluster_name = cluster_dict[ke]
            cluster_dict[ke]= [k for k,v in final_dict.items() if v == ke]
            clu[cluster_name] = cluster_dict[ke]
            final_list.append(clu)
        return final_list


    def extract_terms(self, no_terms):
        '''
        Extract Terms function accepts two arguments, a string and int. Int represents the number of important
        terms to extract from given string.

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.
        cluster_list : int
            This argument accepts a int representing number of terms to extract.

        Return
        ----------
        type : list
            returns a list of strings,
        '''
        text_string = self
        formatted_article_text = re.sub('[^a-zA-Z]', ' ', text_string )
        formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
        doc = nlp(formatted_article_text)
        word_frequencies = {}
        for token in doc:
            word = token.text
            lexeme = nlp.vocab[word]
            if lexeme.is_stop == False:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

        sorted_word_frequencies = {k: v for k, v in sorted(word_frequencies.items(),
                                    key=lambda item: item[1], reverse=True)}
        final_extracted_terms = [word for word in list(sorted_word_frequencies) if len(word) > 2]
        final_extracted_terms = final_extracted_terms[:no_terms]
        return final_extracted_terms


    def tfidf(self, json_version=False):
        '''
        tfidf function accepts a string document and generate tfidf matrix using sklearn tfidfVectorizer.

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.
        json_version : Bool
            json argument is by default json=Flase and function will return Dataframe. If json=True passed,
            function will return json version.

        Return
        ----------
        type : dataframe/json
            returns a matrix or json based on parameters,
        '''
        sentences = self.split('.')
        tfidf_vectorized = TfidfVectorizer()
        values = tfidf_vectorized.fit_transform(sentences)
        feature_names = tfidf_vectorized.get_feature_names()
        df = pd.DataFrame(values.toarray(), columns = feature_names)
        if json_version==True:
            data = df.to_json(orient='index')
            json_object = json.loads(data)
            json_formatted_str = json.dumps(json_object, indent=2, ensure_ascii=False)
            return json_formatted_str
        else:
            return df


    def __part_of(doc):
        subjpass = 0
        for tok, tok_dep in doc.items():
            if tok_dep == "subjpass":
                subjpass = 1
            x = ''
            y = ''
            if subjpass == 1:
                for tok, tok_dep in doc.items():
                    if tok_dep.endswith("subjpass") == True or tok_dep.endswith("nsubj") == True:
                        y = tok
                    if tok_dep.endswith("obj") == True:
                        x = tok
            else:
                for tok, tok_dep in doc.items():
                    if tok_dep.endswith("subjpass") == True or tok_dep.endswith("nsubj") == True:
                        y = tok
                    if tok_dep.endswith("obj") == True:
                        x = tok
        return x,y


    def __getKeysByValue(dictOfElements, valueToFind):
        listOfKeys = list()
        listOfItems = dictOfElements.items()
        for item  in listOfItems:
            if item[1] == valueToFind:
                listOfKeys.append(item[0])
        return  listOfKeys


    def __personal_of(doc):
        person_keys = NlpHub.__getKeysByValue(doc, "PERSON")
        return person_keys


    def extract_relation(self):
        '''
        Extract Terms function accepts two arguments, a string and int. Int represents the number of important
        terms to extract from given string.

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.
        cluster_list : int
            This argument accepts a int representing number of terms to extract.

        Return
        ----------
        type : list
            returns a list of strings,
        '''
        sent_namedEntity = NlpHub.named_entity(self)
        print(sent_namedEntity)
        sent_dependency = NlpHub.dependency(self)
        x,y = NlpHub.__part_of(sent_dependency)
        if x and y:
            ner = sent_namedEntity
            if x in ner:
                if ner[x] == "PERSON" or ner[x] == "ORG":
                    relation = ""+ x +" and "+ y +" have relation of \"Part-of\""
            if y in ner:
                if ner[y] == "PERSON" or ner[y] == "ORG":
                    relation = ""+ x +" and "+ y +" have relation of \"Part-of\""
        person_list = NlpHub.__personal_of(sent_namedEntity)
        if len(person_list) > 1:
            relation = ""+ person_list[0] +" and "+ person_list[1] +" have relation of \"Personal Affiliation\""
            return relation


    def tokenize_sentences(self):
        '''
        Tokenize sentence function accepts a string. That string is splitted into sentence using sentence boundary
        detection. You can pass a complete book to split into sentences.

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.

        Return
        ----------
        type : list
            returns a list of sentences,
        '''
        doc = nlp(self)
        sentences = [send for send in doc.sents]
        return sentences


    def word_cloud(self):
        '''
        Word Cloud fucntions accepts string and plot a word cloud. The bigger a word, the more important it is.

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.

        Output
        ----------
        type : Plot
            plots wordCloud,
        '''
        cloud = WordCloud().generate(self)
        plt.imshow(cloud, interpolation= 'bilinear')
        plt.axis('off')
        plt.show()


    def pdf_to_txt(self):
        '''
       PDF to Text as the name suggests, convert pdf to txt file. Input file name if in same directory otherwise provide full path. 

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string file name.

        Output
        ----------
        type : string
            returns txt version of file,
        '''
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = 'utf-8'  # 'utf16','utf-8'
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr,laparams=laparams)
        fp = open(self, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
            interpreter.process_page(page)
        fp.close()
        device.close()
        str_txt = retstr.getvalue()
        retstr.close()
        return str_txt





