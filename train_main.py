# -*- coding: utf-8 -*-
import cnn_re.cnn as cnn
import cnn_re.vectorizer.relation_vectorizer as relation_vectorizer
import cnn_re.vectorizer.preprocessor as preprocessor
import pickle as pk
import os
import numpy as np
import time
import datetime
import sklearn.metrics as skmetric


def timestamp():
    ts = time.time()
    stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
    return stamp


#n_channel = 1
batch_size = 50
test_interval = 50
snapshot = 10000
iterations = 5000
word2vec_model_path = "./data/word2vec_cn/wiki.zh.text.model"
source_file = './data/d-s-ann.0.5.txt'
jieba_dict = './data/user-2.dict'
nchannel = 1

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


def init_cnn_model(input_shape, classes):
    cnn_model = cnn.CNN()
    cnn_model.net_work_diy(input_shape=input_shape,
                           classes=np.unique(classes))

    return cnn_model


def train(cnn_model, x_train, y_train):
    x_train = x_train[:, :, :, np.newaxis]
    cnn_model.fit(x_train,
                    y_train,
                    test_interval=test_interval,
                    save_path=save_path,
                    snapshot=snapshot,
                    iterations=iterations,
                    batch_size=batch_size, 
                    timestamp=stamp)


def test(cnn_model, x_test, y_test):
    y_pred = cnn_model.predict(x_test)
    print y_pred


def predict(model, x):
    pred = model.predict(x)
    return pred


def train_main():
    init_exp_root();
    X, y = load_relation(source_file)
    n_channel = 1
    x_train, y_train, x_test, y_test = split_train_test(X, y, split_ratio=0.8)
    x_train = x_train[:, :, :320]
    cnn_model = init_cnn_model(input_shape=[x_train.shape[1], x_train.shape[2], n_channel], classes=y)
    train(cnn_model, x_train, y_train)
    pred = predict(cnn_model, x_test)
    skmetric.accuracy_score(y_test, pred)
    print skmetric.confusion_matrix(y_test, pred)


def predict_main():
    X, y = load_relation(source_file)
    X = X[:, :, :320, np.newaxis]
    cnn_model = init_cnn_model(input_shape=[X.shape[1], X.shape[2], nchannel], classes=y)
    cnn_model.restore('./_expdata/20160123212258/iters-4999.model')
    y_pred = predict(cnn_model, X)
    y_pred += 1 
    y = [ int(i) for i in y]
    print  'accuracy: %f' % skmetric.accuracy_score(y, y_pred)
    print 'confusion matrix'
    print skmetric.confusion_matrix(y, y_pred)
    print 'precision score :' % skmetric.precision_score(y, y_pred)
    print 'recall score :' % skmetric.recall_score(y, y_pred) 
    print 'f1 score :' % skmetric.f1_score(y, y_pred) 


def predict_test():
    sentence = u'1\t<e1>抑郁症</e1>症状<e2>情绪低落</e2>就是高兴不起来，总是忧愁伤感，甚至悲观绝望，《红楼梦》中整天皱眉叹气，动不动就流眼泪的林黛玉就是典型的例子。'
    proc_util = preprocessor.RelationPreprocessor(max_token_size=20, chinese_dict=jieba_dict)
    infos, labels = proc_util.get_relation_line(sentence)
    sentence_vectorizer = relation_vectorizer.RelationVectorizer(word2vec_model_path, max_tokens_length=20)
    X, y = sentence_vectorizer.transform([infos], [labels])   
    X = X[:,:,:320, np.newaxis]
    cnn_model = init_cnn_model(input_shape=[X.shape[1], X.shape[2], nchannel], classes=[1,2,3,4,5])
    cnn_model.restore('./_expdata/20160123212258/iters-4999.model')
    y_pred = predict(cnn_model, X) + 1
    
    print  u'sentence : {}. prediction relation label is {}'.format(sentence, y_pred)


if __name__ == '__main__':
    #train_main()
    #test()
    #predict_main()
    predict_test()


