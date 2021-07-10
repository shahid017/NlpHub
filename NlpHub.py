#!/usr/bin/env python
# coding: utf-8

# In[7]:



# coding: utf-8

# In[1]:


import spacy
import nltk
from operator import itemgetter
from nltk.stem import SnowballStemmer
import string
from textblob import TextBlob
from spacy_langdetect import LanguageDetector
from spacy_readability import Readability
import operator
import re
from collections import Counter
import pandas as pd
from gtts import gTTS
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
import docx2txt
from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from matplotlib import pyplot as plt
stemmer = SnowballStemmer('english')
try:
    import en_core_web_md
except OSError:
    raise Exception("Please download spacy model by executing this command. python -m spacy download en_core_web_md")
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



class Transformers():
    '''This class contains all basic NLP operations which transform document or words such as lemmatization or stemming etc.
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
        except Exception as e:
            raise Exception(e)


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
        try:
            stemmed_sent = []
            for w in self.split(" "):
                stemmed_sent.append("".join(stemmer.stem(w)))
            sent = " ".join(stemmed_sent)
            return sent
        except Exception as e:
            raise Exception(e)


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
        try:
            rs = self.capitalize()
            return rs
        except Exception as e:
            raise Exception(e)


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
        try:
            out_str = ""
            if case=='upper':
                out_str = self.upper()
            elif case=='lower':
                out_str = self.lower()
            return out_str
        except Exception as e:
            raise Exception(e)


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
        try:
            sent = self
            word_list = []
            if len(sent.split()) > 1:
                words = sent.split()
                for w in words:
                    if w.isalpha() == True:
                        word_list.append(w)
                sentence = " ".join(word_list)
            return sentence
        except Exception as e:
            raise Exception(e)


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
        try:
            sent = self
            return sent.find(tofind)
        except Exception as e:
            raise Exception(e)


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
        try:
            sent = self
            no_punct = ""
            for char in sent:
                if char not in punctuations:
                    no_punct = no_punct + char
            return no_punct
        except Exception as e:
            raise Exception(e)


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
        try:
            if sw_list:
                temp_split_arr = self.split(' ')
                no_stop_word_list = [el for el in temp_split_arr if el.lower().rstrip() not in sw_list]
                sent = " ".join(no_stop_word_list)
            else:
                temp_split_arr = self.split(' ')
                no_stop_word_list = [el for el in temp_split_arr if el.lower().rstrip() not in extra_words]
                sent = " ".join(no_stop_word_list)
            return sent
        except Exception as e:
            raise Exception(e)


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
        try:
            sent = self
            return_sent = sent.replace(to_replace, replace_by)
            return return_sent
        except Exception as e:
            raise Exception(e)


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
        try:
            sent = self
            result = sent.split(split_on)
            if len(result) > 1:
                return result
            else:
                return sent
        except Exception as e:
            raise Exception(e)


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
        try:
            sent = self
            if len(sent.split()) > 1:
                words = sent.split()
                new_sent = []
                for w in words:
                    left_strip_sent = w.strip(strip_it)
                    new_sent.append(left_strip_sent)
                return_Sent = " ".join(new_sent)
            return return_Sent
        except Exception as e:
            raise Exception(e)


class Annotators():
    '''This class contains all basic NLP operations which annotate document or words such as named_entity, Dependency etc.
    '''

    def __init__(self, input_str):
        self.input_str = input_str


    def sentiment(self, threshold=0.3):
        '''
        Sentiment function accepts a string and do sentiment analysis using textblob library.
        You can pass threshold value.

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.

        Return
        ----------
        type : str
            returns a sentiment of String Positive, Negative or Neutral,
        '''
        try:
            sentence = TextBlob(self)
            sentiment = sentence.sentiment[0]
            if sentiment < -(threshold):
                sent = "Negative"
            elif sentiment > threshold:
                sent = "Positive"
            else:
                sent = "Neutral"
            return sent
        except Exception as e:
            raise Exception(e)


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
        try:
            doc = nlp(self)
            dep = {}
            for token in doc:
                dep[token.text] = token.dep_
            return dep
        except Exception as e:
            raise Exception(e)


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
        try:
            nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
            doc = nlp(self)
            lang = doc._.language
            return lang['language']
        except Exception as e:
            raise Exception(e)


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
        try:
            doc = nlp(self)
            ner = {}
            for ent in doc.ents:
                ner[ent.text] = ent.label_
            return ner
        except Exception as e:
            raise Exception(e)


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
        try:
            doc = nlp(self)
            pos = {}
            for token in doc:
                pos[token.text] = token.pos_
            return pos
        except Exception as e:
            raise Exception(e)


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
        try:
            nlp.add_pipe(Readability())
            main_dict = {}
            doc = nlp(self)
            main_dict["textGradeLevel"] = doc._.flesch_kincaid_grade_level
            main_dict["textReadingEase"] = doc._.flesch_kincaid_reading_ease
            return main_dict
        except Exception as e:
            raise Exception(e)


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
        try:
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
        except Exception as e:
            raise Exception(e)


    def text_similarity(self, match_str, match_path=False):
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
        try:
            X = self
            if match_path == False:
                Y = match_str
            else:
                Y = open(match_str).read()
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
        except Exception as e:
            raise Exception(e)


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
        try:
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
        except Exception as e:
            raise Exception(e)


    def extract_terms(self, terms_number):
        '''
        Extract Terms function accepts two arguments, a string and int. Int represents the number of important
        terms to extract from given string.

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string.
        terms_number : int
            This argument accepts a int representing number of terms to extract.

        Return
        ----------
        type : list
            returns a list of strings,
        '''
        try:
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
            final_extracted_terms = final_extracted_terms[:terms_number]
            return final_extracted_terms
        except Exception as e:
            raise Exception(e)


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
        try:
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
        except Exception as e:
            raise Exception(e)


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
        person_keys = Annotators.__getKeysByValue(doc, "PERSON")
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
        try:
            sent_namedEntity = Annotators.named_entity(self)
            sent_dependency = Annotators.dependency(self)
            x,y = Annotators.__part_of(sent_dependency)
            if x and y:
                ner = sent_namedEntity
                if x in ner:
                    if ner[x] == "PERSON" or ner[x] == "ORG":
                        relation = ""+ x +" and "+ y +" have relation of \"Part-of\""
                if y in ner:
                    if ner[y] == "PERSON" or ner[y] == "ORG":
                        relation = ""+ x +" and "+ y +" have relation of \"Part-of\""
            person_list = Annotators.__personal_of(sent_namedEntity)
            if len(person_list) > 1:
                relation = ""+ person_list[0] +" and "+ person_list[1] +" have relation of \"Personal Affiliation\""
                return relation
        except Exception as e:
            raise Exception(e)


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
        try:
            doc = nlp(self)
            sentences = [send for send in doc.sents]
            return sentences
        except Exception as e:
            raise Exception(e)


class Analyzers():
    '''This class contains all basic NLP operations which transform and plot on data.
    '''

    def __init__(self, input_str):
        self.input_str = input_str


    def word_cloud(self, plot=True):
        '''
        Word Cloud fucntion accepts string and plot a word cloud. The bigger a word, the more important it is.
        if plot is flase, it will return word cloud object. By default plot is True.
        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string. Plot take boolean value.

        Output
        ----------
        type : Plot
            plots wordCloud or returns cloud object.
        '''
        try:
            word_cloud_object = WordCloud().generate(self)
            if plot==True:
                plt.imshow(word_cloud_object, interpolation= 'bilinear')
                plt.axis('off')
                plt.show()
            else:
                return word_cloud_object
        except Exception as e:
            raise Exception(e)


    def word_frequency(self, plot=True, stemming=True, word_list="", top_n=10):
        '''
        word frequency fucntion accepts string and make dictionary of word frequency.
        if plot is flase, it will return word frequency object. By default plot is True.
        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string. Plot take boolean value.

        Output
        ----------
        type : Plot
            plots word frequency or returns word frequency object.
        '''
        try:
            return_dict = {}
            if stemming == True:
                doc_text = Transformers.stem(self)
            counts = dict()
            if word_list !="":
                words = word_list.split(",")
                for word in words:
                    word_count = len(re.findall(f'\\b{word.lower()}\\b', doc_text.lower()))
                    counts[word] = word_count
            else:
                document_words = doc_text.lower().split()
                counts = dict(Counter(document_words))
            if top_n !="":
                top_values = dict(sorted(counts.items(), key = itemgetter(1), reverse = True)[:int(top_n)])
            else:
                top_values = counts
            if plot==True:
                plt.bar(range(len(top_values)), list(top_values.values()), align='center')
                plt.xticks(range(len(top_values)), list(top_values.keys()))
                plt.show()
            else:
                return top_values
        except Exception as e:
            raise Exception(e)


class Convertors():
    '''This class contains all NLP operations you can perform using basicNLP.
    '''

    def __init__(self, input_file):
        self.input_file = input_file

    def pdf_to_txt(self, save_to=""):
        '''
       PDF to Text as the name suggests, convert pdf to txt file. Input file name if in same directory otherwise provide full path.
       takes save_to paramter if you want to save extracted text directly into some file. Just pass output file path in save_to.

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string file name.

        type: path
            save_to argument is used to take path if you want to save file.

        Output
        ----------
        type : string
            returns txt version of file or save path.
        '''
        try:
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
            if save_to:
                with open(save_to, 'w') as save_file:
                    save_file.write(str_txt)
                return save_to
            else:
                return str_txt
        except Exception as e:
            raise Exception(e)


    def docx_to_txt(self, save_to=""):
        '''
       Docx to Text as the name suggests, convert docx to txt file. Input file name if in same directory otherwise provide full path.
       takes save_to paramter if you want to save extracted text directly into some file. Just pass output file path in save_to.

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string file name.

        type: path
            save_to argument is used to take path if you want to save file.

        Output
        ----------
        type : string
            returns txt version of file or save path.
        '''
        try:
            txt_str = docx2txt.process(self)
            if save_to:
                with open(save_to, 'w') as save_file:
                    save_file.write(txt_str)
                return save_to
            else:
                return txt_str
        except Exception as e:
            raise Exception(e)


    def txt_to_docx(self, save_to=""):
        '''
       Text to Docx as the name suggests, write text to docx file. Input text you want to write to docx.
       takes docx file path as save_to. Just pass output file path in save_to.

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string file name.

        type: path
            save_to argument is used to take path if you want to save file.

        Output
        ----------
        type : string
            returns save path.
        '''
        try:
            if save_to:
                with open(save_to, 'w') as save_file:
                    save_file.write(self)
                return save_to
            else:
                return txt_str
        except Exception as e:
            raise Exception(e)


    def txt_to_pdf(self, save_to=""):
        '''
       Text to PDF as the name suggests, write text to pdf file. Input text you want to write to pdf.
       takes pdf file path as save_to. Just pass output file path in save_to.

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string file name.

        type: path
            save_to argument is used to take path if you want to save file.

        Output
        ----------
        type : string
            returns save path.
        '''
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size = 15)
            doc_text = str(self.encode('utf-8').decode('latin-1'))
            pdf.write(5, doc_text)
            pdf.output(save_to)
            return save_to
        except Exception as e:
            raise Exception(e)


    def txt_to_speech(self, save_to="", speaking_speed="slow"):
        '''
       Text to Speech as the name suggests, converts text to speech(mp3) file. Input text you want to convert to mp3.
       takes mp3 file path as save_to. Just pass output file path in save_to.

        Input
        ----------
        type : str
            The self argument is used as input and it accepts a string file name.

        type: path
            save_to argument is used to take path if you want to save file.

        Output
        ----------
        type : string
            returns mp3 speech version of text.
        '''
        try:
            language = 'en'
            if speaking_speed == "slow":
                myobj = gTTS(text=self, lang=language, slow=True)
            else:
                myobj = gTTS(text=self, lang=language, slow=False)
            myobj.save(save_to)
            return save_to
        except Exception as e:
            raise Exception(e)
