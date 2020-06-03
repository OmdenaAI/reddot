#to import the data into a dataframe
import json 
import pandas as pd 
from pandas import json_normalize #package for flattening json in pandas df
#to normalize the data
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet') 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
#nltk.download('popular', quiet=True)
#to create a wordcloud
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
#Machine Learning
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix
#bi-grams and tri-grams
import seaborn as sns


#load json object
with open('./data/abuse.json') as f:
    d = json.load(f)

df = json_normalize(data = d['articles']) 

print(df.head(5))
print("\nSchema:\n",df.dtypes)
print("\nNb of articles, columns = ",df.shape)

def pre_process(text):
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("</?.*?>'"," <> ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    #Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', text)

    return text

df['text'] = df['title'] + df['description'] + df['content'] 
#df['text'] = df['text'].apply(lambda x:pre_process(x))

#show the first 'text'
#print("\nText content : \n" + df['text'][2])

#Fetch wordcount for each abstract
df['word_count'] = df['text'].apply(lambda x: len(str(x).split(" ")))
print(df[['text','word_count']].head())

#Descriptive statistics of word counts
print("\n")
print(df.word_count.describe())

#print(pre_process('text'))

#Creating a list of stop words 
#print(stopwords.words("english"))
stop_words = set(stopwords.words("english"))
##Creating a list of custom stopwords
new_words = ["published", "chars", "pm", "file"]
stop_words = stop_words.union(new_words)

corpus = []
for i in range(0, 20): 

	text = pre_process(df['text'][i])

	#Convert to list from string
	text = text.split()

	#Stemming
	ps = PorterStemmer()

	#Lemmatisation
	lem = WordNetLemmatizer()
	text = [lem.lemmatize(word) for word in text if not word in stop_words] 
	text = " ".join(text)
	corpus.append(text)

#print(corpus[2])

#Word cloud
wordcloud = WordCloud(background_color='white',stopwords=stop_words,max_words=100,max_font_size=50, random_state=42).generate(str(corpus))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
#plt.show()
fig.savefig("word1.png", dpi=900)

cv=CountVectorizer(max_df=0.9,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
X=cv.fit_transform(corpus)

print(list(cv.vocabulary_.keys())[:10])

#Most frequently occuring words
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    return words_freq[:n]

#Convert most freq words to dataframe for plotting bar plot
top_words = get_top_n_words(corpus, n=20)
top_df = pd.DataFrame(top_words)
top_df.columns=["Word", "Freq"]

#Barplot of most freq words
sns.set(rc={'figure.figsize':(13,8)})
g = sns.barplot(x="Word", y="Freq", data=top_df)
g.set_xticklabels(g.get_xticklabels(), rotation=30)

#Most frequently occuring Bi-grams
def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2), max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

top2_words = get_top_n2_words(corpus, n=20)
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
print(top2_df)

#Barplot of most freq Bi-grams
sns.set(rc={'figure.figsize':(13,8)})
h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
h.set_xticklabels(h.get_xticklabels(), rotation=45)

#Most frequently occuring Tri-grams
def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

top3_words = get_top_n3_words(corpus, n=20)
top3_df = pd.DataFrame(top3_words)
top3_df.columns=["Tri-gram", "Freq"]
print(top3_df)

#Barplot of most freq Tri-grams
sns.set(rc={'figure.figsize':(13,8)})
j=sns.barplot(x="Tri-gram", y="Freq", data=top3_df)
j.set_xticklabels(j.get_xticklabels(), rotation=45)

#TF-IDF
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(X)# get feature names
feature_names=cv.get_feature_names()

#Function for sorting tf_idf in descending order
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

doc = ""
for i in range(0, 20): 

	# fetch document for which keywords needs to be extracted
	doc=doc + corpus[i]

#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())

#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,5)
 
# now print the results
print("\nText analyzed:")
print(doc)
print("\nKeywords:")
for k in keywords:
    print(k,keywords[k])


