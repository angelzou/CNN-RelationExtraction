__author__ = 'hadyelsahar'

import numpy as np
from sklearn.base import TransformerMixin
from word_vectorizer import WordVectorizer

# inputs :
# segments            : array of strings of max length m (padding smaller sizes sequences with zeros)
# segments labels     : array of strings
# ent1,ent2 : position of entity1 and entity2 in segments    0 <= ent1, ent2 < m
# outputs :
# vector representation of each segment : mxn martix  m = len(segments+padding), n = len(Wvec + position_vec + features)


class RelationVectorizer(TransformerMixin):

    def __init__(self, word2vec_model_path, position_vector=True, word_position_size=10, ner=False, pos=False, dependency=False):

        self.position_vector = position_vector

        # word_position vectors will be filled when calling init_size function
        self.word_position = None
        self.word_position_size = word_position_size
        self.sentence_vectorizer = WordVectorizer(word2vec_model_path, ner=ner, pos=pos, dependency=dependency)

        # sizes of the output sequence matrix m is number of words in the sequence
        # n is the size of the vector representation of each word in the sequence
        self.m = None
        self.n = self.sentence_vectorizer.model.vector_size + 2 * self.word_position_size
        # self.n = 380 + 2*self.word_position_size

    def transform(self, sentence_matrix):
        """
        :param X: array(dict) [{segments:[],segment_labels:[],ent1:int, ent2:int}]
        :param transform_params:
        :return:
        """
        sentence_matrix_out = np.zeros([0, self.m, self.n], np.float32)

        for sentence_elements in sentence_matrix:
            # padding with zeros
            sentence_vec = [self.sentence_vectorizer.word2vec(elements[-1])
                            for elements in sentence_elements["segments"]]
            sentence_vec = np.array(sentence_vec, dtype=np.float32)

            pad_size = self.m - sentence_vec.shape[0]

            if pad_size > 0:
                temp = np.zeros((pad_size, self.sentence_vectorizer.model.vector_size))
                sentence_vec = np.vstack([sentence_vec, temp])
            else:
                # TODO trim sentence when it's length is longer than m
                pass

            # position with respect to ent1
            entity1_vec = self.lookup_word_pos(sentence_elements["ent1_pos"])  # dimension m x _
            # position with respect to ent2
            entity2_vec = self.lookup_word_pos(sentence_elements["ent2_pos"])  # dimension m x _
            #  merging different parts of vector representation of words
            print sentence_vec.shape, entity1_vec.shape, entity2_vec.shape
            print sentence_matrix_out.shape
            sentence_matrix_vec = np.hstack([sentence_vec, entity1_vec, entity2_vec])
            sentence_matrix_out = np.append(sentence_matrix_out, [sentence_matrix_vec], axis=0)
        return sentence_matrix_out

    def init_size(self, sentence_matrix):
        """
        :param X: array(dict) [{segments:[], segment_labels:[], ent1:int, ent2:int}]
        :param y:
        :param fit_params:
        :return:
        """
        max_sentence_length = max([len(sentence["segments"]) for sentence in sentence_matrix])
        self.m = max_sentence_length
        # original index = -l+1,....,0,...l-1
        # array index    = 0,.......,(l-1),...(2xl)-1
        self.word_position = np.random.rand((2 * max_sentence_length) - 1, self.word_position_size)
        return self

    def lookup_word_pos(self, p):
        """
        :param p: position of entity
        :return: array of dimension self.m x self.word_position_size

        example : if ent1 = 2 self.m = 10   i.e. : (w0, w1, w2(e1), w3, w4, w5, w6, w7, w8, w9)
                  return: word_position[-2:8]   ===
                  add (l-1) to get indices between (0,2l-1) ===>  word_position[7:17]
        """
        start = -p + self.m - 1
        end = start + self.m
        return self.word_position[start:end]










