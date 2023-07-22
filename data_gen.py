import statistics
import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from nltk.stem import PorterStemmer, WordNetLemmatizer
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')



data_vec =pd.read_csv('./articles.csv', encoding='ISO-8859-1')

feat = []
for i in range(data_vec.shape[0]):
    if not (data_vec['Heading'][i].isspace() or data_vec['Article.Description'][i].isspace() or data_vec['Full_Article'][i].isspace()):
        if data_vec['Heading'][i] != "" and data_vec['Article.Description'][i] != "" and data_vec['Full_Article'][i] != "":
            feat.append(([data_vec['Heading'][i], data_vec['Article.Description'][i], data_vec['Full_Article'][i]]))

flatList = sum(feat, [])
# create the transform [data['Title'][i], data['Abstract'][i], data['Conclusion'][i]]
vectorizer = CountVectorizer()
vectorizer.fit(flatList)


def stop_word(words):
    # get list of english stop words
    stop_words = stopwords.words('english')

    # remove stop words
    filtered_words = [word for word in words if word not in stop_words]

    return filtered_words

def stemming(words):

    # create stemmer
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]

    return stemmed_words

def lemmatization(words):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    return lemmatized_words

def tokenization(str):
    # create list of words from text
    words = nltk.word_tokenize(str)

    return words

def tf_idf(text_ds):
    # NMF is able to use tf-idf
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_vectorizer.fit(flatList)

    # encode document
    tfidf = tfidf_vectorizer.transform([text_ds])


    return tfidf.toarray()

def latent_sematic(str):

    # encode document
    td_matrix = vectorizer.transform([str])

    # create the model
    lsa = TruncatedSVD(n_components=2)

    # fit and transform the term-document matrix
    td_matrix_2d = lsa.fit_transform(td_matrix)

    return td_matrix_2d

def bag_of_words(str):
    # encode document
    vector = vectorizer.transform([str])

    return np.sum(vector.toarray())

def part_of_speech(words):
    # perform POS tagging
    tags1 = nltk.pos_tag(words)
    tags=[]
    for i in tags1:
        tags.append(i[1])
    # create list of unique POS tags
    unique_tags = list(set(tags))

    # create one-hot encoding of POS tags
    tag_features = []
    for tag in tags:
        feature = np.zeros(len(unique_tags))
        feature[unique_tags.index(tag)] = 1
        tag_features.append(feature)
    tag_features=np.array(tag_features)

    return np.sum(tag_features)

def pointwise_mutual_information(tokens):
    # create a list of all word pairs
    pairs = []
    for i in range(len(tokens) - 1):
        pairs.append((tokens[i], tokens[i + 1]))

    # calculate the frequency of each word
    freq = nltk.FreqDist(tokens)

    # calculate the PMI of each word pair
    pmi_values = {}
    for pair in pairs:
        x = freq[pair[0]]
        y = freq[pair[1]]
        xy = freq[pair]+1
        n = len(tokens)
        pmi = math.log2(xy / (x * y))
        pmi_values[pair] = pmi

    return sum(list(pmi_values.values()))

def feat_techn(str):
    # tokenization
    words = tokenization(str)

    # Stop word removal
    words = stop_word(words)

    # Stemming
    words = stemming(words)

    #lemmatization
    words = lemmatization(words)

    str1 = ' '.join(words)

    # Bag of words
    feat1 = bag_of_words(str1)

    #part_of_speech
    feat2 = part_of_speech(words)

    # pointwise_mutual_information
    feat3 = pointwise_mutual_information(words)

    # tf_idf
    feat4 = tf_idf(str1)

    # latent_sematic
    feat5 = latent_sematic(str1)
    feat = [[feat1], [feat2], [feat3], feat4.tolist()[0], feat5.tolist()[0]]
    feat = sum(feat, [])

    return feat


def feat_extract(data):
    feat=[]
    for i in data:
        feat.append(feat_techn(i))
    feat = sum(feat, [])
    return np.array(feat)


def datagen():
    data =pd.read_csv('./articles.csv', encoding='ISO-8859-1')
    feat, label_str, label_uniq, label=[], [],{},[]
    n=0
    for i in range(data.shape[0]):
        if not (data_vec['Heading'][i].isspace() or data_vec['Article.Description'][i].isspace() or data_vec['Full_Article'][i].isspace()):
            if data['Heading'][i] != "" and data['Article.Description'][i] != "" and data['Full_Article'][i] != "":
                feat.append(feat_extract([data['Heading'][i], data['Article.Description'][i], data['Full_Article'][i]]))
                label_str.append(data['Article_Type'][i])
                if not data['Article_Type'][i] in label_uniq:
                    label_uniq.update({data['Article_Type'][i]:n})
                    n+=1
                label.append(label_uniq[data['Article_Type'][i]])

    feat=np.array(feat)
    # Normalization
    feat = feat / np.max(feat, axis=0)
    # Data Cleaning and dummy variable for nan
    feat = np.nan_to_num(feat)


    return feat, label