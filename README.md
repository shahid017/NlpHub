# NlpHub

NlpHub is a wrapper around Spacy, NLTK and some other libraries simple NLP tasks. You can perform simple tasks with just 3 or 4 lines of code. NlpHub can perform basic NLP operations with less code and less hassle.

Installation
You can install NlpHub via pip. Write this command in terminal and voila, you have installed NlpHub.

    pip install NlpHub
It will auto install dependencies and you don't need to worry about anything.

# Import
To use NlpHub in your Jupyter notebook or python file, You can import it like this.

    from NlpHub import basicNLP

# Functions

NlpHub class has multiple functions that you can use. Some of them are explained here.

## Function 1 basicNLP.lemma()
Lemmatization function accepts a string and return the lemmatized version of that input string.

## Function 2 basicNLP.stem()
Stem function accepts a string and return the stemmed version of that input string.

## Function 3 basicNLP.capitalize()
Capitalize function accepts a string and return the capitalized version of that input string.

## Function 4 basicNLP.case_change()
Case Change function accepts a string and case and return the case Changed version of that input string. Takes case as parameter. call as NlpHub.case_change(str, 'lower or upper')

## Function 5 basicNLP.filter_alpha()
 Filter Alphabet function accepts a string and return the filtered version of that input string.

## Function 6 basicNLP.find()
Find function accepts a string and and a string to find return the index of match in input string. Call as NlpHub.find(str, 'str to find')

## Function 7 basicNLP.remove_punct()
Remove Punctuation function accepts a string and return the version of that input string with no puncts.

## Function 8 basicNLP.remove_sw()
Remove Stopwords function accepts a string and a optional stopword list  and return the string without any stopwords.

## Function 9 basicNLP.replace()
Replace function accepts a string, a string to replace and a string to replace by and return the string after replacing the intended words. Call as NlpHub.find(str, 'str to replace', 'str to replace by')

## Function 10 basicNLP.split()
Split function accepts a string and a string to split and return the string after split it on given parameter. Call as NlpHub.split(str , ',')

## Function 11 basicNLP.strip()
Strip function accepts a string and a string to strip and return the string after stripping given parameter from left and right.

## Function 12 basicNLP.sentiment()
Sentiment function accepts a string and do sentiment analysis using textblob library. Return str

## Function 13 basicNLP.dependency()
Dependency function accepts a string and returns dependency dictionary with words as keys and their dependency as values. Uses Spacy library

## Function 14 basicNLP.language()
language function accepts a string and return the language of string using Spacy library.

## Function 15 basicNLP.named_entity()
Named Entity function accepts a string and returns Named Entity dict with words as keys and their entity as values. Uses Spacy library

## Function 16 basicNLP.pos_tagging()
Parts of Speech Tagging function accepts a string and returns Parts of Speech dict with words as keys and their POS as values. Uses Spacy library
readability_score
## Function 17 basicNLP.readability_score()
Readability Score function accepts a string and returns flesch kincaid Readability Score dict. Uses Spacy library

## Function 18 basicNLP.similar_word()
Similar Word function accepts a string and a word to match and returns a string with maximum similarity.

## Function 19 basicNLP.text_similarity()
Text Similarity function accepts two string and returns similarity.

## Function 20 basicNLP.clustering()
Clustering function accepts two arguments, a string and list of strings and returns a list of dicts.
        Clustering computes similarity of each word in string with each cluster word in the cluster_list
        and add word to cluster with maximum similarity.
        
## Function 21 basicNLP.extract_terms()
Extract Terms function accepts two arguments, a string and int. Int represents the number of important
        terms to extract from given string.

## Function 22 basicNLP.tfidf()
tfidf function accepts a string document and generate tfidf matrix using sklearn tfidfVectorizer.       

## Function 23 basicNLP.extract_relation()
Extract Relation accepts a string and find any relation between entities of that string.

## Function 24 basicNLP.tokenize_sentences()
Tokenize sentence function accepts a string. That string is splitted into sentence using sentence boundary
        detection. You can pass a complete book to split into sentences.

## Function 25 basicNLP.word_cloud()
Word Cloud fucntions accepts string and plot a word cloud. The bigger a word, the more important it is.

## Function 26 basicNLP.pdf_to_txt()
PDF to Text as the name suggests, convert pdf to txt file. Input file name if in same directory otherwise provide full path. 

# About Author
Hi there, My name is Muhammad Shahid and I am working as Data Scientist in Five Rivers Technologies. I have done BSCS from COMSATS University Islamabad and MSCS from NUCES FAST Islamabad. I have expertise in Data Science, Machine Learning(ML) and Natural Language Processing(NLP). If you have any query about this project, you may reach me at chshahidhamdam@gmail.com

