import cnn_re.cnn as cnn
import cnn_re.vectorizer.relation_vectorizer as relation_vectorizer
import re_preprocessor as preprocessor
import pickle as pk
import os
import numpy as np
###########################################
# Preprocessing:
# Convert into Inputs into the relation_mention_vectorizer:
# X: array(dict) [{sentence_id:, id:, segments:[], segment_labels:[], ent1:int, ent2:int}]
# y: array(string) : labels of realations

#######################################
if not os.path.exists("./dataset.p"):
    data_dir = './data/'
    word2vec_model_path = "./data/word2vec/wiki.en.text.model"

    corpse = preprocessor.RelationPreprocessor(data_dir)
    sentence_vectorizer = relation_vectorizer.RelationVectorizer(word2vec_model_path)
    sentence_vectorizer.init_size(corpse.lines_info)
    X = sentence_vectorizer.transform(corpse.lines_info)
    y = corpse.labels
    pk.dump((X, y), file=open("./dataset.p", 'w'))
else:
    X, y = pk.load(open("./dataset.p", 'r'))


y = np.array(y)

'''
x_train = X[0:1100]
x_test = X[1100:]

y_train = y[0:1100]
y_test = y[1100:]
'''
n_sample = X.shape[0]
n_word = X.shape[1]
n_feature = X.shape[2]
train_ratio = 0.8
x_train = X[0: np.floor(n_sample * train_ratio)]
y_train = y[0: np.floor(n_sample * train_ratio)]
y_train = y[np.ceil(n_sample * train_ratio):]
y_test = y[np.ceil(n_sample * train_ratio)]

n_channel = 1
batch_size = 2

'''
x_train = np.reshape(x_train, [-1, n_word, n_feature, n_channel])
cnn_model = cnn.CNN(input_shape=[n_word, n_feature, n_channel], classes=np.unique(y), conv_shape=[5, 25])
'''

x_train = np.reshape(x_train, [-1, n_word, n_feature, n_channel])
x_train = x_train[-1, :20, :320, 0]
x_train = np.reshape(x_train, [-1, 20, 320, n_channel])
cnn_model = cnn.CNN(input_shape=[x_train.shape[1], x_train.shape[2], 1], classes=np.unique(y), conv_shape=[5, 25])
cnn_model.fit(x_train, y_train)

####################
# Training the CNN #
####################
'''
x_train = np.reshape(x_train, [-1, 8, 389, 320])
cnn_model = cnn(input_shape=[8, 389, 320], classes=np.unique(y), conv_shape=[5, 25])
cnn_model.fit(x_train, y_train)
'''

'''
# seqwidth x veclength x channels
x_train = np.reshape(x_train, [-1, 20, 320, 1])
cnn = CNN(input_shape=[20, 320, 1], classes=np.unique(y), conv_shape=[5, 25])
cnn.fit(x_train, y_train)
'''


# Testing :
###########
x_test = np.reshape(x_test, [-1, 20, 320, 1])
y_pred = cnn_model.predict(x_test)

# y_true = [list(i).index(1) for i in y_true]

# print classification_report(y_true, y_pred)




