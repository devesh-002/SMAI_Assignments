# # -*- coding: utf-8 -*-
# """2021121003_A1_Q1.ipynb

# Automatically generated by Colaboratory.

# Original file is located at
#     https://colab.research.google.com/drive/1mbXQVnXZzVh7145dfD9WLm3GUum2j7mr

# # Assignment 1
# ## Question `1` (K-Nearest Neighbour)

# | | |
# |-|-|
# | Course | Statistical Methods in AI |
# | Release Date | `19.01.2023` |
# | Due Date | `29.01.2023` |

# ### Instructions:
# 1.   Assignment must be implemented using python notebook only (Colab , VsCode , Jupyter etc.)
# 2.   You are allowed to use libraries for data preprocessing (numpy, pandas, nltk etc) and for algorithms as well (sklearn etc). You are not however allowed to directly use classifier models.
# 3.   The performance of the model will hold weightage but you will also be graded largely for data preprocessing steps , explanations , feature selection for vectors etc.
# 4.   Strict plagiarism checking will be done. An F will be awarded for plagiarism.

# ### The Dataset
# The dataset is avaible in the zip file which is a collection of *11099 tweets*. The data will be in the form of a csv file. The ground truth is also given in the zip file which corresponds to whether a tweet was popular or not. Since the task involves selecting features yourself to vectorize a tweet , we suggest some data analysis of the columns you consider important.
# <br><br>

# ### The Task
# You have to build a classifier which can predict the popularity of the tweet, i.e , if the tweet was popular or not. You are required to use **KNN** algorithm to build the classifier and cannot use any inbuilt classifier. All columns are supposed to be analyzed , filtered and preprocessed to determine its importance as a feature in the vector for every tweet (Not every column will be useful).<br>
# The Data contains the **raw text of the tweet**(in the text column) as well as other **meta data** like likes count , user followers count. Note that it might be useful to **create new columns** with useful information. For example, *number of hashtags* might be useful but is not directly present as a column.<br>
# There are 3 main sub parts:
# 1. *Vectorize tweets using only meta data* - likes , user followers count , and other created data
# 2. *Vectorize tweets using only it's text*. This segment will require NLP techniques to clean the text and extract a vector using a BoW model. Here is a useful link for the same - [Tf-Idf](https://towardsdatascience.com/text-vectorization-term-frequency-inverse-document-frequency-tfidf-5a3f9604da6d). Since these vectors will be very large , we recommend reducing their dimensinality (~10 - 25). Hint: [Dimentionality Reduction](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491). Please note that for this also you are allowed to use libraries.

# 3. *Combining the vectors from above two techinques to create one bigger vector*
# <br>


# Using KNN on these vectors build a classifier to predict the popularity of the tweet and report accuracies on each of the three methods as well as analysis. You can use sklearn's Nearest Neighbors and need not write KNN from scratch. (However you cannot use the classifier directly). You are expected to try the classifier for different number of neighbors and identify the optimal K value.

# ### Announcements

# 1. You are expected to only use the sklearn.neighbors.NearestNeighbors function and not the KNeighboursClassifier function.
# 2. You are free to choose the dimensionality The range mentioned is just a recommendation based on testing.
# 3. You can use sklearn or any other library to vectorize for tf-idf

# ### Resources

# 1. https://towardsdatascience.com/data-normalization-with-pandas-and-scikit-learn-7c1cc6ed6475
# 2. https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
# 3. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# 4. https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a

# ## Import necessary libraries
# """

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from google.colab import drive
drive.mount('/content/drive')

"""## Load and display the data"""

#your code here
df = pd.read_csv('drive/MyDrive/Tweets.csv')
df2 = pd.read_csv('drive/MyDrive/ground_truth.csv', header=None)
df2.columns = ['GT']
df2['GT'] = df2['GT'].astype('int')
# df2
print(df.dtypes)
print(df2)
# for i in df.columns:
#     print(type(df[i]))

df.truncated.unique()
# huh
# how many?
item_counts = df['truncated'].value_counts()
print(item_counts)
# well, not that it matters.
    # one, i cannot untruncate the tweets
    # two, i don't care about the semantic meaning in this case

# pd.set_option('display.max_colwidth', None)
print(df.metadata)
# nothing much in metadata
'''
notable features
    # favorite_count
    # retweet_count
    # user_followers_count
    # user_listed_count
    # user_friends_count
    # user_favourites_count
    # user_statuses_count
    # hashtags from entities
        # we take number of hashtags as a feature
    # user_verified    
    # is_quote_status
    # lang
''' 
    
# retweet count should matter?
# follower count matter over friend count?
# text: BoW Tf-Idf
# favorite_count = like

# print(df.entities)
num_of_hashtags = []
#     print(res['hashtags'], len(res['hashtags']))
for i in df.entities:
    res = eval(i)
    num_of_hashtags.append(len(res['hashtags']))
# print(num_of_hashtags)
df['hashtag_count'] = num_of_hashtags
df

df = df.drop(['entities', 'created_at', 'id', 'id_str', 'text', 'truncated', 'metadata', 'source', 'user_name', 'user_screen_name', 'user_created_at'], axis=1)
df.columns

df

"""## Exploratory Data Analysis
*This is an ungraded section but is recommended to get a good grasp on the dataset*
"""

# your code here
print(df.info())
print(df.describe())

# fixing the columns to create 11d graph
df["is_quote_status"] = df["is_quote_status"].astype(int)
df["user_verified"] = df["user_verified"].astype(int)
# df['is_quote_status'] = df['is_quote_status'].replace('False','0')
# df['is_quote_status'] = df['is_quote_status'].replace('True','1')
df

item_counts = df["lang"].value_counts()
item_counts=(item_counts.to_dict())
pd.options.mode.chained_assignment = None  # default='warn'

# df['lang']=item_counts[df['lang']]
# print(item_counts)
for i in df.index:
    x=df['lang'][i]
    df['lang'][i]=item_counts[x]
    
df

"""## Part-1
*Vectorize tweets using only meta data*
"""

def get_features(df, df2):
    arr = df.to_numpy()
    ground_truth = df2.to_numpy()
    return arr, ground_truth
    
    
x, y = get_features(df, df2)
print(x[0], y[2])
df2

# """Perform KNN using the vector obtained from get_features() function. Following are the steps to be followed:
# 1. Normalise the vectors
# 2. Split the data into training and test to estimate the performance.
# 3. Fit the Nearest Neughbiurs module to the training data and obtain the predicted class by getting the nearest neighbours on the test data.
# 4. Report the accuracy, chosen k-value and method used to obtain the predicted class. Hint: Plot accuracies for a range of k-values. 
# """

# vectors obtained are x, y
# normalizing using numpy

vector_norm_x = np.linalg.norm(x)
print(vector_norm_x)
normalized_vector = x / np.linalg.norm(x)
print(normalized_vector)

# normalization using scikit
normalized_x = normalize(x)
normalized_x
x_train, x_test, y_train, y_test = train_test_split(normalized_x, y, test_size=0.2, random_state=37)

# huh getting different normalized values
# gonna use scikit's implementation here
from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(n_neighbors=3, algorithm='auto', metric='minkowski', p=2)
knn.fit(x_train)
val = knn.kneighbors(x_test, return_distance=False)

def classification(val, x_train, y_train):
    predicted_test = []
    for i in val:
        counter=0
        for j in i:
            if y_train[j] == 0:
                counter = counter - 1
            else:
                counter = counter + 1
        if(counter<=0):
            predicted_test.append([0])
        else:
            predicted_test.append([1])
    return np.array(predicted_test)
    
l = classification(val, x_train, y_train)
acc = accuracy_score(y_test, l)
acc

# experimenting for different values of k
accuracy_scores = []
for i in range(1, 100):
    knn = NearestNeighbors(n_neighbors = i, algorithm='auto', metric='minkowski', p=2)
    knn.fit(x_train)
    val = knn.kneighbors(x_test, return_distance=False)
    l = classification(val, x_train, y_train)
    acc = accuracy_score(y_test, l)
    accuracy_scores.append(acc)
accuracy_scores
xacc = max(accuracy_scores)
print(xacc)

# matplotlib karde bhai

# """## Part-2
# Vectorize tweets based on the text. More details and reference links can be checked on the Tasks list in the start of the notebook
# """

# !pip install PyStemmer

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from Stemmer import Stemmer
import re

# ps = PorterStemmer()
ps = Stemmer('porter')

stop_words = set(stopwords.words("english"))
stop_words.add('rt')

def tokenise(data):
    data = data.lower()
    # removing {|}
    data = re.sub(r"{\|(.*?)\|}", " ", data, flags=re.DOTALL)
    # removing html stuff
    data = re.sub(r"&nbsp;|&lt;|&gt;|&amp;|&quot;|&apos;", r" ", data)
    # substituting hyperlinks with " "
    data = re.sub(r"http\S*[\s | \t | \n]", r" ", data)
    # removing tags and hashtags
    re.sub("(#[A-Za-z0-9]+)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", data)
    # tokenization
    tokens = re.split(r'[^A-Za-z0-9]+', data)
    return tokens

def stemming_and_stopping(data):
    StemmedUp = []
    StemmedUp = [ps.stemWord(i) for i in data if i not in stop_words if len(i) < 35 if len(i) >=2]
    return StemmedUp

def preprocessing(data):
    for idx, x in enumerate(data):
      x = tokenise(x)
      x = stemming_and_stopping(x)
      x = " ".join(x)
      data[idx] = x
    return data

new_df = pd.read_csv('drive/MyDrive/Tweets.csv')
# print(preprocessing(new_df.text.to_list()))
preprocessed = preprocessing(new_df.text.to_list())
print(preprocessed)

from sklearn.feature_extraction.text import TfidfVectorizer

def tweet_vectoriser(data):
#   """
#   Funtion to return a matrix of dimensions (number of tweets, number of features extracted per tweet)
#   Following are the steps for be followed:
#     1. Remove links, tags and hashtags from each tweet.
#     DONE IN THE CELL ABOVE
#     2. Apply TF-IDF on the tweets to extract a vector. 
#     3. Perform dimensionality reduction on the obtained vector. 
#   Input parameters to this funcion are to be chosen as per requirement (Example: Array of tweets) 
#   """
  # your code here
  vectorizer = TfidfVectorizer(analyzer = 'word')

  # convert the documents into a matrix
  tfidf_wm = vectorizer.fit_transform(data)
  
  tfidf_tokens = vectorizer.get_feature_names()
  size_of_index = len(data)+1
  df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(), index = list(range(1,size_of_index)), columns = tfidf_tokens)
  return df_tfidfvect

# print(tweet_vectoriser(preprocessed))
# print(len(tweet_vectoriser(preprocessed)))
vectorized_tweet_text = tweet_vectoriser(preprocessed)

# time for dimensionality reduction
# PCA transforms data linearly into new properties that are not correlated with each other.
# '''
#   SVD gives you the whole nine-yard of diagonalizing a matrix into special matrices 
#   that are easy to manipulate and to analyze. It lay down the foundation to untangle 
#   data into independent components. PCA skips less significant components. 
#   Obviously, we can use SVD to find PCA by truncating the less important basis 
#   vectors in the original SVD matrix.
# '''

from sklearn.decomposition import PCA

def get_features(df, df2):
    arr = df.to_numpy()
    ground_truth = df2.to_numpy()
    return arr, ground_truth

tweet_x, tweet_y = get_features(vectorized_tweet_text, df2)
pca = PCA(n_components = 20)
pca.fit(tweet_x)
tweet_x = pca.transform(tweet_x)
print(tweet_x.shape)


normalized_tweet_x = normalize(tweet_x)
normalized_tweet_x



x_train, x_test, y_train, y_test = train_test_split(normalized_tweet_x, tweet_y, test_size=0.2, random_state=37)

knn = NearestNeighbors(n_neighbors=3, algorithm='auto', metric='minkowski', p=2)
knn.fit(x_train)
val = knn.kneighbors(x_test, return_distance=False)

def text_classification(val, x_train, y_train):
    predicted_test = []
    for i in val:
        counter=0
        for j in i:
            if y_train[j] == 0:
                counter = counter - 1
            else:
                counter = counter + 1
        if(counter<=0):
            predicted_test.append([0])
        else:
            predicted_test.append([1])
    return np.array(predicted_test)
    
l = text_classification(val, x_train, y_train)
acc = accuracy_score(y_test, l)
acc

# experimenting for different values of k
text_accuracy_scores = []
for i in range(1, 100):
    knn = NearestNeighbors(n_neighbors = i, algorithm='auto', metric='minkowski', p=2)
    knn.fit(x_train)
    val = knn.kneighbors(x_test, return_distance=False)
    l = text_classification(val, x_train, y_train)
    acc = accuracy_score(y_test, l)
    text_accuracy_scores.append(acc)
text_accuracy_scores
xacc_text = max(text_accuracy_scores)
print(xacc_text)

# matplotlib karde bhai

# """Perform KNN using the vector obtained from tweet_vectoriser() function. Following are the steps to be followed:

# 1. Normalise the vectors
# 2. Split the data into training and test to estimate the performance.
# 3. Fit the Nearest Neughbiurs module to the training data and obtain the predicted class by getting the nearest neighbours on the test data.
# 4. Report the accuracy, chosen k-value and method used to obtain the predicted class. Hint: Plot accuracies for a range of k-values.
# """



"""## Part-3
### Subpart-1

Combine both the vectors obtained from the tweet_vectoriser() and get_features()
"""

# your code here
x
len(x)
x.shape

tweet_x
len(tweet_x)
tweet_x.shape

new_vec = np.concatenate((x, tweet_x), axis = 1)
new_vec.shape

# """Perform KNN using the vector obtained in the previous step. Following are the steps to be followed:

# 1. Normalise the vectors
# 2. Split the data into training and test to estimate the performance.
# 3. Fit the Nearest Neughbiurs module to the training data and obtain the predicted class by getting the nearest neighbours on the test data.
# 4. Report the accuracy, chosen k-value and method used to obtain the predicted class. Hint: Plot accuracies for a range of k-values.
# """

normalized_x = normalize(new_vec)
normalized_x
x_train, x_test, y_train, y_test = train_test_split(normalized_x, y, test_size=0.2, random_state=37)

knn = NearestNeighbors(n_neighbors=3, algorithm='auto', metric='minkowski', p=2)
knn.fit(x_train)
val = knn.kneighbors(x_test, return_distance=False)

def concat_classification(val, x_train, y_train):
    predicted_test = []
    for i in val:
        counter=0
        for j in i:
            if y_train[j] == 0:
                counter = counter - 1
            else:
                counter = counter + 1
        if(counter<=0):
            predicted_test.append([0])
        else:
            predicted_test.append([1])
    return np.array(predicted_test)
    
l = concat_classification(val, x_train, y_train)
acc = accuracy_score(y_test, l)
acc

# experimenting for different values of k
concat_accuracy_scores = []
for i in range(1, 50):
    knn = NearestNeighbors(n_neighbors = i, algorithm='auto', metric='minkowski', p=2)
    knn.fit(x_train)
    val = knn.kneighbors(x_test, return_distance=False)
    l = concat_classification(val, x_train, y_train)
    acc = accuracy_score(y_test, l)
    concat_accuracy_scores.append(acc)
concat_accuracy_scores
xacc_concat = max(concat_accuracy_scores)
print(xacc_concat)

# matplotlib karde bhai

# """### Subpart-2

# Explain the differences between the accuracies obtained in each part above based on the features used.
# """

# text gives slightly better accuracy than other features