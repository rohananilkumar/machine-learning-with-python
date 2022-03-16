import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
# nltk.download_shell()
# Use this to download stop words. The stop words are the words that are most common in a language which can be ignored while processing a given text

messages = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])

# print(messages.describe())

# print(messages.groupby('label').describe())

messages['length'] = messages['message'].apply(len)

# print(messages.head())

# messages['length'].plot.hist(bins=50)

# messages.hist(column='length', by='label', bins=60) 
# This shows that length is a good feature to distinguish between spam and ham

def  text_process(mess):
    """
    1. Remove punc words
    2. Remove stop words
    3. Return list of clean text words
    """

    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# print(messages['message'].head(5).apply(text_process))

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
print(len(bow_transformer.vocabulary_))

mess4 = messages['message'][3]

print(mess4)

bow4 = bow_transformer.transform([mess4])
# This returns a frequency count and the index of the word in the bow

messages_bow = bow_transformer.transform(messages['message'])

print("Shape of sparse matrix:",  messages_bow.shape)

sparsity = (100.0*messages_bow.nnz/(messages_bow.shape[0]*messages_bow.shape[1]))

print('sparsity :{}'.format(round(sparsity)))

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)

print(tfidf4)

print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

messages_tfidf = tfidf_transformer.transform(messages_bow)

#using naive_bayes algorithm
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])
print(spam_detect_model.predict(tfidf4)[0])

#Now doing all of the above using train test split 
msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=.3)

#creating a pipeline to do all the steps in one single step
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(msg_train, label_train)

predictions = pipeline.predict(msg_test)

print(classification_report(label_test,predictions))

# print(bow4)
# plt.show()
