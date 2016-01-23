import cnn_re.cnn as cnn
import cnn_re.vectorizer.relation_vectorizer as relation_vectorizer
import cnn_re.vectorizer.preprocessor as preprocessor
import pickle as pk
import os
import numpy as np
import time
import datetime


def timestamp():
    ts = time.time()
    stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
    return stamp


#n_channel = 1
batch_size = 50
test_interval = 20
snapshot = 10000
iterations = 50000
word2vec_model_path = "./data/word2vec_cn/wiki.zh.text.model"
source_file = './data/d-s-ann.0.5.txt'
jieba_dict = './data/user-2.dict'

stamp = timestamp()
exp_root = './_expdata'
save_path = os.path.join(exp_root, stamp)
os.mkdir(save_path)
log_dir = os.path.join(save_path, 'log')
print 'exp data is put into %s' % save_path


def init_exp_root():
    if not os.path.exists(exp_root):
        os.mkdir(exp_root)


# TODO: preprocessor is not a good name, so as to name as X, y
# TODO: X: should be 4-d array
def load_relation(source_file):
    # Convert into Inputs into the relation_mention_vectorizer:
    # X: array(dict) [{sentence_id:, id:, segments:[], segment_labels:[], ent1:int, ent2:int}]
    # y: array(string) : labels of realations
    if not os.path.exists("./dataset.p"):
        proc_util = preprocessor.RelationPreprocessor(max_token_size=20, chinese_dict=jieba_dict)
        infos, labels = proc_util.load_data(source_file)
        sentence_vectorizer = relation_vectorizer.RelationVectorizer(word2vec_model_path, max_tokens_length=20)
        X, y = sentence_vectorizer.transform(infos, labels)
        pk.dump((X, y), file=open("./dataset.p", 'w'))
    else:
        X, y = pk.load(open("./dataset.p", 'r'))

    y = np.array(y)
    return X, y


def split_train_test(X, y, split_ratio=0.8):
    n_sample = X.shape[0]
    n_train_size = np.floor(n_sample * split_ratio)
    x_train = X[0: n_train_size]
    y_train = y[0: n_train_size]
    x_test = X[n_train_size:]
    y_test = y[n_train_size:]
    return x_train, y_train, x_test, y_test


def create_cnn_model(input_shape, classes):
    cnn_model = cnn.CNN(input_shape=input_shape,
                        classes=np.unique(classes),
                        iterations=iterations,
                        batch_size=batch_size, 
                        log_dir=log_dir)

    return cnn_model


def train(cnn_model, x_train, y_train):
    '''n_train_size = x_train.shape[0]
    n_word = x_train.shape[1]
    n_feature = x_train.shape[2]
    x_train = x_train[:, :, :320, 0]
    '''
    x_train = x_train[:, :, :, np.newaxis]
    cnn_model.fit(x_train, y_train, test_interval, save_path=save_path, snapshot=snapshot)


def test(cnn_model, x_test, y_test):
    '''n_test_size = y_test.shape[0]
    n_word = y_test.shape[1]
    n_feature = y_test.shape[2]
    n_channel = 1
    x_test = np.reshape(x_test, [n_test_size, n_word, n_feature, n_channel])
    x_test = x_test[:, :20, :320, 0]
    x_test = np.reshape(x_test, [-1, 20, 320, n_channel])
    '''
    y_pred = cnn_model.predict(x_test)
    print y_pred


def train_main():
    init_exp_root();
    X, y = load_relation(source_file)
    n_channel = 1
    x_train, y_train, x_test, y_test = split_train_test(X, y, split_ratio=0.8)
    x_train = x_train[:, :, :320]
    cnn_model = create_cnn_model(input_shape=[x_train.shape[1], x_train.shape[2], n_channel], classes=y)
    train(cnn_model, x_train, y_train)


def predict(x):
    model_path = ''
    cnn_model = create_cnn_model(input_shape=[x.shape[1], x.shape[2], n_channel], classes=y)
    cnn_model.restore(model_path)
    pred = cnn_model.predict(x_train)


def predict_main():
    predict(x)


if __name__ == '__main__':
    train_main()
    #predict_main()


