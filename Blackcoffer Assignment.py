#!/usr/bin/env python
# coding: utf-8

# In[31]:


# installing required libraries
get_ipython().system('pip install requests')
get_ipython().system('pip install beautifulsoup4')
get_ipython().system('pip install numpy')
get_ipython().system('pip install io')
get_ipython().system('pip install json')
get_ipython().system('pip install numpy')
get_ipython().system('pip install requests')
get_ipython().system('pip install syllables')
get_ipython().system('pip install nltk')


# In[32]:


#importing library
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
from io import BytesIO
import json
import numpy as np
from requests.models import MissingSchema 
import syllables
import nltk


# In[33]:


# The omw-1.4 resource in NLTK refers to the Open Multilingual Wordnet version 1.4. 
# WordNet is a lexical database of the English language, and it's widely used in natural language processing tasks.
nltk.download('omw-1.4')


# In[34]:


# load input data file 
df=pd.read_excel('/Users/hemantgarg/Downloads/URL.xlsx')[['URL_ID','URL']]


# In[35]:


df


# ## DATA EXTRACTION
# 
# ## For each of the articles, given in the input.xlsx file, extract the article text and save the extracted article in a text file 
# ## with URL_ID as its file name. While extracting text, please make sure your program extracts only the article title and the article text. 
# ## It should not extract the website header, footer, or anything other than the article text. 

# In[36]:


for i, row in df.iterrows():
    # Extract the URL and URL ID from the row
    url = row['URL']
    url_id = row['URL_ID']

    # Send a request to the webpage
    response = requests.get(url)

    # Parse the HTML content of the webpage using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the textual content of the article
    article_text = ""
    for p in soup.find_all('p'):
        if p.find('img'):
            # Ignore if an image is nested within a  tag
            continue
        else:
            article_text += p.get_text()

    # Save the extracted article text to a text file
    output_file = f"{url_id}.txt"
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(article_text)

    print(f"Extracted article from {url} and saved to {output_file}.")
     


# ## DATA ANALYSIS
# 
# 
# ## 1. Personal pronouns 

# In[54]:


# The "regex" package provides an alternative implementation of regular expressions compared to the built-in "re" module.
# using regex to find the counts of the words - “I,” “we,” “my,” “ours,” and “us”. 
# Special care is taken so that the country name US is not included in the list.
def count_personal_pronouns(text):
  personal_pronouns = ['I', 'we', 'my', 'ours', 'us']
  # Define a regular expression pattern to match the personal pronouns
  pattern = r'\b(' + '|'.join(personal_pronouns) + r')\b'
  # Compile the regular expression pattern
  regex = re.compile(pattern, flags=re.IGNORECASE)
  # Count the number of personal pronouns in the text
  count = len(regex.findall(text))
  return count


# In[38]:


import re
personalpronouns=[]
for i, row in df.iterrows():
    # Extract the URL and URL ID from the row
    url_id = row['URL_ID']
    with open(f'{url_id}.txt', 'r', encoding='utf-8') as file:
        text = file.read()
        personalpronouns.append(count_personal_pronouns(text))
     


# ## 2. Stop Words 

# In[39]:


def load_stop_words(file_path='/Users/hemantgarg/Downloads/StopWords'):
    with open(file_path, 'r', encoding='latin-1') as f:
        stop_words = [line.strip() for line in f]
    return set(stop_words)
    
stop_words_file = '/Users/hemantgarg/Downloads/StopWords/StopWords_Auditor.txt'
stop_words_Auditor = load_stop_words(stop_words_file)

stop_words_file = '/Users/hemantgarg/Downloads/StopWords/StopWords_Geographic.txt'
stop_words_Geographic = load_stop_words(stop_words_file)

stop_words_file = '/Users/hemantgarg/Downloads/StopWords/StopWords_Currencies.txt'
stop_words_Currencies = load_stop_words(stop_words_file)

stop_words_file = '/Users/hemantgarg/Downloads/StopWords/StopWords_DatesandNumbers.txt'
stop_words_DatesandNumbers = load_stop_words(stop_words_file)

stop_words_file = '/Users/hemantgarg/Downloads/StopWords/StopWords_Generic.txt'
stop_words_generic = load_stop_words(stop_words_file)

stop_words_file = '/Users/hemantgarg/Downloads/StopWords/StopWords_GenericLong.txt'
stop_words_GenericLong = load_stop_words(stop_words_file)

stop_words_file = '/Users/hemantgarg/Downloads/StopWords/StopWords_Names.txt'
stop_words_Names = load_stop_words(stop_words_file)


# In[40]:


import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Set up NLTK
nltk.download('punkt')

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Define a function for text normalization
def normalize_text(text):
    # Remove all non-word characters and convert to lowercase
    text = re.sub(r'[^\w.]', ' ', text)
    text = text.lower().strip()
    text = text.replace('.', ' FULL_STOP_TOKEN ')
    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stop words
    words = [word for word in words if word not in stop_words_Auditor]
    words = [word for word in words if word not in stop_words_Geographic]
    words = [word for word in words if word not in stop_words_Currencies]
    words = [word for word in words if word not in stop_words_DatesandNumbers]
    words = [word for word in words if word not in stop_words_generic]
    words = [word for word in words if word not in stop_words_GenericLong]
    words = [word for word in words if word not in stop_words_Names]
    # Lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join the words back into a single string
    text = ' '.join(words)

    return text


# In[41]:


for i, row in df.iterrows():
    # Extract the URL and URL ID from the row
    url_id = row['URL_ID']
    url = row['URL']

    # Read the text from the file
    with open(f'{url_id}.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    # Clean and normalize the text
    
    normalized_text = normalize_text(text)
  
    # Write the normalized text to the file
    with open(f'{url_id}.txt', 'w', encoding='utf-8') as file:
        file.write(normalized_text)


# ## 3. Positive Score 

# In[42]:


positive_words = set()

# Load positive words from file
with open('/Users/hemantgarg/Downloads/MasterDictionary/positive-words.txt', 'r') as f:
    positive_words = set(word.strip() for word in f.readlines())

def get_positive_score(text):
    # Split the text into words
    words = text.split()

    # Count the number of positive words in the text
    positive_count = sum(1 for word in words if word in positive_words)

    total_words = len(words)
    if total_words == 0:
        return 0
    positive_score = positive_count / total_words

    return positive_score


# ## 4. Negative Score 

# In[43]:


negative_words = set()

# Load positive words from file
with open('/Users/hemantgarg/Downloads/MasterDictionary/negative-words.txt', 'r', encoding='latin-1') as f:
    negative_words = set(word.strip() for word in f.readlines())

def get_negative_score(text):
    # Split the text into words
    words = text.split()

    # Count the number of positive words in the text
    negative_count = sum(1 for word in words if word in negative_words)

    total_words = len(words)
    if total_words == 0:
        return 0
    negative_score = negative_count / total_words

    return negative_score


# ## 5. Syllable Per Word

# In[44]:


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize, pos_tag

def count_syllables(word):
    vowels = 'aeiouy'
    num_vowels = 0
    for i in range(len(word)):
        if word[i].lower() in vowels:
            num_vowels += 1
            if i > 0 and word[i-1].lower() in vowels:
                num_vowels -= 1
    if word.endswith('e'):
        num_vowels -= 1
    if num_vowels == 0:
        num_vowels = 1
    return num_vowels
     


# ## 6. Percentage of Complex words 

# In[45]:


def count_complex_words(text):
    # Split the text into words
    words = text.split()

    # Count the number of complex words
    count = 0
    for word in words:
        # Get the number of syllables in the word
        syllable_count = syllables.estimate(word)

        # Count the word as complex if it has more than two syllables
        if syllable_count > 2:
            count += 1

    return count


def calculate_percentage_of_complex_words(text):
    words = word_tokenize(text)
    total_words = len(text)
    if total_words == 0:
        return 0
    num_complex_words = count_complex_words(text)
    percentage_of_complex_words = (num_complex_words / total_words)
    return percentage_of_complex_words


# In[46]:


def count_syllables(word):
    """
    Count the number of syllables in a word.

    Args:
        word (str): The word to count syllables for.

    Returns:
        int: The number of syllables in the word.
    """
    # Define a list of vowels
    vowels = ['a', 'e', 'i', 'o', 'u', 'y']

    # Remove trailing "es" and "ed" from the word
    word = word.rstrip('es')
    word = word.rstrip('ed')

    # Count the number of vowels in the word
    syllables = 0
    prev_char = None
    for char in word:
        if char in vowels and (prev_char is None or prev_char not in vowels):
            syllables += 1
        prev_char = char

    # Handle special cases where the word ends in "le"
    if word.endswith('le') and word[-3] not in vowels:
        syllables += 1

    # Handle special cases where the word has no vowels
    if syllables == 0:
        syllables = 1

    return syllables


# In[47]:


def count(word):
    words = text.split()
    count = 0
    for word in words:
        if word != 'FULL_STOP_TOKEN':
            count += len(word)
    return count


# In[48]:


positive=[]
negative=[]
polarity=[]
subjectivity=[]
fogIndex=[]
avgnum=[]
complexwords=[]
syllable=[]
wordcount=[]
avgwordlength=[]
for i, row in df.iterrows():
    # Extract the URL and URL ID from the row
    url_id = row['URL_ID']
    with open(f'{url_id}.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    
    sentences = text.split('FULL_STOP_TOKEN')
    num_sentences = len(sentences)
    
    # Remove the FULL_STOP_TOKEN from the text
    text = text.replace('FULL_STOP_TOKEN', "")
    # Split the text into words
    words = text.split()
    # Count the words
    num_words = len(words)
    
    positive_score = get_positive_score(text)
    positive.append(positive_score)
    
    negative_score = get_negative_score(text)
    negative.append(negative_score)
    
    temp=((positive_score - negative_score)/ ((positive_score + negative_score) + 0.000001))
    polarity.append(temp)
    
    temp2=(positive_score + negative_score)/ ((num_words) + 0.000001)
    subjectivity.append(temp2)
    
    if num_sentences==0:
        Average_Sentence_Length=0
    else:
        Average_Sentence_Length= num_words/num_sentences
    Percentage_of_Complex_words=calculate_percentage_of_complex_words(text)
    fogIndex.append(0.4 * (Average_Sentence_Length + Percentage_of_Complex_words))
    
    avgnum.append(Average_Sentence_Length)
    
    complexwords.append(count_complex_words(text))
    
    wordcount.append(num_words)
    
    syllable.append(count_syllables(text))
    
    
    if num_words==0:
        Average=0
    else:
        Average= count(text)/num_words
    
    avgwordlength.append(Average)


# In[49]:


df["POSITIVE SCORE"]=positive
df["NEGATIVE SCORE"]=negative
df["POLARITY SCORE"]=polarity
df["SUBJECTIVITY SCORE"]=subjectivity
df["FOG INDEX"]=fogIndex
df["AVG NUMBER OF WORDS PER SENTENCE"]=avgnum
df["COMPLEX WORD COUNT"]=complexwords
df["WORD COUNT"]=wordcount
df["SYLLABLE PER WORD"]=syllable
df["PERSONAL PRONOUNS"]=personalpronouns
df["AVG WORD LENGTH"]=avgwordlength  


# In[53]:


## avg sentence length is same as avg number of words per sentences 


# In[55]:


df.to_excel('Output Data Structure.xlsx', index=False)


# In[56]:


df


# In[57]:


df.head(5)


# In[ ]:




