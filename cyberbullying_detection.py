#!/usr/bin/env python
# coding: utf-8
# In[1]:
import snscrape.modules.twitter as sntwitter
from deep_translator import GoogleTranslator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
import re
import string # from some string manipulation tasks
from collections import Counter

from string import punctuation # solving punctuation problems
from nltk.corpus import stopwords # stop words in sentences
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer # For stemming the sentence
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer # For stemming the sentence
from contractions import contractions_dict # to solve contractions
from autocorrect import Speller #correcting the spellings

from sklearn.metrics import classification_report, confusion_matrix

#Data preprocessing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#Naive Bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
nltk.download('punkt')
import emoji

from nltk.corpus import stopwords

nltk.download('wordnet')
import unidecode
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image #to load our image

import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger')

# In[2]:


def case_convert(text):
    return text.lower()
def correct_word(text):
    return  Speller()(text)
def convert_emoji():
    df.tweet = [emoji.demojize(text) for text in df.tweet.values]

def remove_specials(text):
    return re.sub(r"[^a-zA-Z]"," ",text)


# In[3]:


def remove_shorthands(text):
    CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    string = ""
    for word in text.split(" "):
        if word.strip() in list(CONTRACTION_MAP.keys()):
            string = string + " " + CONTRACTION_MAP[word.strip()]
        else:
            string = string + " " + word
    
    return string.strip()


# In[4]:


def remove_abbreviations(text):
    abbreviations = {
        "$" : " dollar ",
        "€" : " euro ",
        "4ao" : "for adults only",
        "a.m" : "before midday",
        "a3" : "anytime anywhere anyplace",
        "aamof" : "as a matter of fact",
        "acct" : "account",
        "adih" : "another day in hell",
        "afaic" : "as far as i am concerned",
        "afaict" : "as far as i can tell",
        "afaik" : "as far as i know",
        "afair" : "as far as i remember",
        "afk" : "away from keyboard",
        "app" : "application",
        "approx" : "approximately",
        "apps" : "applications",
        "asap" : "as soon as possible",
        "asl" : "age, sex, location",
        "atk" : "at the keyboard",
        "ave." : "avenue",
        "aymm" : "are you my mother",
        "ayor" : "at your own risk", 
        "b&b" : "bed and breakfast",
        "b+b" : "bed and breakfast",
        "b.c" : "before christ",
        "b2b" : "business to business",
        "b2c" : "business to customer",
        "b4" : "before",
        "b4n" : "bye for now",
        "b@u" : "back at you",
        "bae" : "before anyone else",
        "bak" : "back at keyboard",
        "bbbg" : "bye bye be good",
        "bbc" : "british broadcasting corporation",
        "bbias" : "be back in a second",
        "bbl" : "be back later",
        "bbs" : "be back soon",
        "be4" : "before",
        "bfn" : "bye for now",
        "blvd" : "boulevard",
        "bout" : "about",
        "brb" : "be right back",
        "bros" : "brothers",
        "brt" : "be right there",
        "bsaaw" : "big smile and a wink",
        "btw" : "by the way",
        "bwl" : "bursting with laughter",
        "c/o" : "care of",
        "cet" : "central european time",
        "cf" : "compare",
        "cia" : "central intelligence agency",
        "csl" : "can not stop laughing",
        "cu" : "see you",
        "cul8r" : "see you later",
        "cv" : "curriculum vitae",
        "cwot" : "complete waste of time",
        "cya" : "see you",
        "cyt" : "see you tomorrow",
        "dae" : "does anyone else",
        "dbmib" : "do not bother me i am busy",
        "diy" : "do it yourself",
        "dm" : "direct message",
        "dwh" : "during work hours",
        "e123" : "easy as one two three",
        "eet" : "eastern european time",
        "eg" : "example",
        "embm" : "early morning business meeting",
        "encl" : "enclosed",
        "encl." : "enclosed",
        "etc" : "and so on",
        "faq" : "frequently asked questions",
        "fawc" : "for anyone who cares",
        "fb" : "facebook",
        "fc" : "fingers crossed",
        "fig" : "figure",
        "fimh" : "forever in my heart", 
        "ft." : "feet",
        "ft" : "featuring",
        "ftl" : "for the loss",
        "ftw" : "for the win",
        "fwiw" : "for what it is worth",
        "fyi" : "for your information",
        "g9" : "genius",
        "gahoy" : "get a hold of yourself",
        "gal" : "get a life",
        "gcse" : "general certificate of secondary education",
        "gfn" : "gone for now",
        "gg" : "good game",
        "gl" : "good luck",
        "glhf" : "good luck have fun",
        "gmt" : "greenwich mean time",
        "gmta" : "great minds think alike",
        "gn" : "good night",
        "g.o.a.t" : "greatest of all time",
        "goat" : "greatest of all time",
        "goi" : "get over it",
        "gps" : "global positioning system",
        "gr8" : "great",
        "gratz" : "congratulations",
        "gyal" : "girl",
        "h&c" : "hot and cold",
        "hp" : "horsepower",
        "hr" : "hour",
        "hrh" : "his royal highness",
        "ht" : "height",
        "ibrb" : "i will be right back",
        "ic" : "i see",
        "icq" : "i seek you",
        "icymi" : "in case you missed it",
        "idc" : "i do not care",
        "idgadf" : "i do not give a damn fuck",
        "idgaf" : "i do not give a fuck",
        "idk" : "i do not know",
        "ie" : "that is",
        "i.e" : "that is",
        "ifyp" : "i feel your pain",
        "IG" : "instagram",
        "iirc" : "if i remember correctly",
        "ilu" : "i love you",
        "ily" : "i love you",
        "imho" : "in my humble opinion",
        "imo" : "in my opinion",
        "imu" : "i miss you",
        "iow" : "in other words",
        "irl" : "in real life",
        "j4f" : "just for fun",
        "jic" : "just in case",
        "jk" : "just kidding",
        "jsyk" : "just so you know",
        "l8r" : "later",
        "lb" : "pound",
        "lbs" : "pounds",
        "ldr" : "long distance relationship",
        "lmao" : "laugh my ass off",
        "lmfao" : "laugh my fucking ass off",
        "lol" : "laughing out loud",
        "ltd" : "limited",
        "ltns" : "long time no see",
        "m8" : "mate",
        "mf" : "motherfucker",
        "mfs" : "motherfuckers",
        "mfw" : "my face when",
        "mofo" : "motherfucker",
        "mph" : "miles per hour",
        "mr" : "mister",
        "mrw" : "my reaction when",
        "ms" : "miss",
        "mte" : "my thoughts exactly",
        "nagi" : "not a good idea",
        "nbc" : "national broadcasting company",
        "nbd" : "not big deal",
        "nfs" : "not for sale",
        "ngl" : "not going to lie",
        "nhs" : "national health service",
        "nrn" : "no reply necessary",
        "nsfl" : "not safe for life",
        "nsfw" : "not safe for work",
        "nth" : "nice to have",
        "nvr" : "never",
        "nyc" : "new york city",
        "oc" : "original content",
        "og" : "original",
        "ohp" : "overhead projector",
        "oic" : "oh i see",
        "omdb" : "over my dead body",
        "omg" : "oh my god",
        "omw" : "on my way",
        "p.a" : "per annum",
        "p.m" : "after midday",
        "pm" : "prime minister",
        "poc" : "people of color",
        "pov" : "point of view",
        "pp" : "pages",
        "ppl" : "people",
        "prw" : "parents are watching",
        "ps" : "postscript",
        "pt" : "point",
        "ptb" : "please text back",
        "pto" : "please turn over",
        "qpsa" : "what happens", #"que pasa",
        "ratchet" : "rude",
        "rbtl" : "read between the lines",
        "rlrt" : "real life retweet", 
        "rofl" : "rolling on the floor laughing",
        "roflol" : "rolling on the floor laughing out loud",
        "rotflmao" : "rolling on the floor laughing my ass off",
        "rt" : "retweet",
        "ruok" : "are you ok",
        "sfw" : "safe for work",
        "sk8" : "skate",
        "smh" : "shake my head",
        "sq" : "square",
        "srsly" : "seriously", 
        "ssdd" : "same stuff different day",
        "tbh" : "to be honest",
        "tbs" : "tablespooful",
        "tbsp" : "tablespooful",
        "tfw" : "that feeling when",
        "thks" : "thank you",
        "tho" : "though",
        "thx" : "thank you",
        "tia" : "thanks in advance",
        "til" : "today i learned",
        "tl;dr" : "too long i did not read",
        "tldr" : "too long i did not read",
        "tmb" : "tweet me back",
        "tntl" : "trying not to laugh",
        "ttyl" : "talk to you later",
        "u" : "you",
        "u2" : "you too",
        "u4e" : "yours for ever",
        "utc" : "coordinated universal time",
        "w/" : "with",
        "w/o" : "without",
        "w8" : "wait",
        "wassup" : "what is up",
        "wb" : "welcome back",
        "wtf" : "what the fuck",
        "wtg" : "way to go",
        "wtpa" : "where the party at",
        "wuf" : "where are you from",
        "wuzup" : "what is up",
        "wywh" : "wish you were here",
        "yd" : "yard",
        "ygtr" : "you got that right",
        "ynk" : "you never know",
        "zzz" : "sleeping bored and tired"
    }

    string = ""
    for word in text.split(" "):
        if word.strip() in list(abbreviations.keys()):
            string = string + " " + abbreviations[word.strip()]
        else:
            string = string + " " + word
    return string.strip()


# In[5]:


def remove_stopwords(text):
    stopwords_list = stopwords.words('english')
    string = ""
    for word in text.split(" "):
        if word.strip() in stopwords_list:
            continue
        else:
            string = string + " " + word
    return string


# In[6]:


def remove_links(text):
    remove_https = re.sub(r'http\S+', '', text)
    remove_com = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)
    return remove_com


# In[7]:


def remove_accents(text):
    return unidecode.unidecode(text)


# In[8]:


def normalize_spaces(text):
    return re.sub(r"\s+"," ",text)


# In[9]:


lemmatizer = WordNetLemmatizer()

def pos_tagger(nltk_tag):
	if nltk_tag.startswith('J'):
		return wordnet.ADJ
	elif nltk_tag.startswith('V'):
		return wordnet.VERB
	elif nltk_tag.startswith('N'):
		return wordnet.NOUN
	elif nltk_tag.startswith('R'):
		return wordnet.ADV
	else:		
		return None



def lemmatize(sentence):
	pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
	wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
	lemmatized_sentence = []
	for word, tag in wordnet_tagged:
		if tag is None:
			lemmatized_sentence.append(word)
		else:	
			lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
	lemmatized_sentence = " ".join(lemmatized_sentence)
	return lemmatized_sentence


# In[10]:


def preprocess_text(text):
    # text=correct_word(text)
    text=remove_stopwords(text)
    text=case_convert(text)
    text=remove_abbreviations(text)
    text=remove_links(text)
    text=remove_shorthands(text)
    text=remove_accents(text)
    text=remove_specials(text)
    text=lemmatize(text)
    text=normalize_spaces(text)
    return text


# In[11]:


vec = pickle.load(open('count_vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
fit_lenc = pickle.load(open('fit_lenc.pkl', 'rb'))


st.set_page_config(page_title="cyberbullying Detection", page_icon=":rotating_light:", layout="wide")

st.title('Cyberbullying Detection WebApp')
st.write('---')


with st.container():
    left_column, right_column = st.columns(2)
    with left_column:

        st.subheader("Hi, I am [Debojyoti](https://debojyoti31.github.io/) :wave:")
        # st.write("[:globe_with_meridians:  My Website  :globe_with_meridians:](https://debojyoti31.github.io/)")
        st.write(
            """
            This is a NLP Model trained on 50000 tweets labelled according to the class of Cyberbullying:

            - Age
            - Ethnicity
            - Gender
            - Religion
            - Not Cyberbullying  
            """
        )
        st.write("[Learn More about Cyberbullying](https://www.unicef.org/end-violence/how-to-stop-cyberbullying)")
        st.write('---')
        

    with right_column:









        option = st.radio('**Select Input Type**',('Single Text', 'Twitter Username'))
        
        if option == 'Single Text':
            input_text = st.text_area('**Text to detect**', '''Love Conquers All.''')
            if st.button('Predict'):
                if input_text != '':
                    input_text = emoji.demojize(input_text)
                    input_text = GoogleTranslator(source='auto', target='en').translate(input_text)
                    input_text = preprocess_text(input_text)
                    st.write('Keywords :',  ', '.join(input_text.split())  )

                    
                    input_text1 = vec.transform([input_text]).toarray()

                    # predict
                    y_pred = model.predict(input_text1)

                    # display
                    st.write('**:red[Cyberbullying Type :]**')
                    st.subheader(fit_lenc.inverse_transform(y_pred)[0])
                    st.caption('This is only a Prediction based on Machine Learning, it may not be accurate.')
                    
                else:
                    st.subheader('Please enter a text!')
        
        



        
        if option == 'Twitter Username':


            num = st.slider('Number of Tweets', 100, 1000, 25)
            id = st.text_input('**Username without @**', '''imessi''')

            if st.button('Predict'):
                st.write(':red[Scroll Down to see Result] :arrow_double_down:')
                if id != '':
                    tweets_list1 = []
                    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:'+str(id)).get_items()):
                        if i>num:
                            break
                        tweets_list1.append([tweet.content, tweet.lang])                       
                    df = pd.DataFrame(tweets_list1, columns=['tweet', 'lang'])
                    df.dropna(inplace=True)
                    if df.shape[0] == 0:
                        st.write('**:red[Can Not Scrape Twitter Account !!]**')
                        st.write('**:red[Either Wrong Username Given or Secuirity Protected Account !!! ]**')
                    else:
                        convert_emoji()
                        for i in range(len(df.tweet)):
                            if df.lang.iloc[i] != 'en':
                                df.tweet.iloc[i] = GoogleTranslator(source='auto', target='en').translate(df.tweet.iloc[i])
                        df.tweet =  [preprocess_text(text) for text in df.tweet.values]
                        st.write('Translated Tweet Keywords :')
                        df.dropna(inplace=True)
                        st.dataframe(df.tweet)


                        list_input = vec.transform(df.tweet.values).toarray()
                        y_pred = model.predict(list_input)
                        pred_list = fit_lenc.inverse_transform(y_pred).tolist()

                        fig, ax = plt.subplots()
                        ax.pie(Counter(pred_list).values() , labels = Counter(pred_list).keys(), autopct='%1.1f%%',pctdistance=0.8)

                        st.pyplot(fig)
                        st.caption('This is only a Prediction based on Machine Learning, it may not be accurate.')
                    
                else:
                    st.subheader('Please enter a text!')



