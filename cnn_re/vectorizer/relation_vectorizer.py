__author__ = 'hadyelsahar'
__author__ = 'olenet'


import numpy as np
from word_vectorizer import WordVectorizer
from tqdm import tqdm


class RelationVectorizer():

    def __init__(self, word2vec_model_path, max_tokens_length=20, position_vector=True, word_position_size=10, ner=False, pos=False, dependency=False):

        self.position_vector = position_vector

        # word_position vectors will be filled when calling init_size function
        self.word_position = None
        self.word_position_size = word_position_size
        self.sentence_vectorizer = WordVectorizer(word2vec_model_path, ner=ner, pos=pos, dependency=dependency)

        # sizes of the output sequence matrix m is number of words in the sequence
        # n is the size of the vector representation of each word in the sequence
        self.m = max_tokens_length
        self.n = self.sentence_vectorizer.model.vector_size + 2 * self.word_position_size
        # original index = -l+1,....,0,...l-1
        # array index    = 0,.......,(l-1),...(2xl)-1
        self.word_position = np.random.rand((2 * max_tokens_length) - 1, self.word_position_size)

    # write words which do not have mapping vector to file
    def word2vec_test(self, tokens):
	no_vec_words = []
	for token in tokens:
	    vec = self.sentence_vectorizer.word2vec(token)
	    if len(vec.shape) == 0:
		print token.encode('utf-8'), 'length vec == 0'
		no_vec_words.append(token)
	with open('../../../data/no_vec_words.txt', 'w') as f:
	    f.writelines(' '.join(no_vec_words).encode('utf-8'))

    def tokens_to_vec(self, tokens):
        sentence_vec = []
        for token in tokens:
            vec = self.sentence_vectorizer.word2vec(token)
            if len(vec.shape) == 0:
                print token.encode('utf-8'), 'length vec == 0'
                return []
            sentence_vec.append(vec)

        sentence_vec = np.array(sentence_vec, dtype=np.float32)
        return sentence_vec

    def transform(self, sentence_matrix, labels):
        sentence_matrix_out = np.zeros([0, self.m, self.n], np.float32)
        valid_label = []

        count = 0
        for sentence_elements, label in tqdm(zip(sentence_matrix, labels)):
            count += 1
            sentence_vec = self.tokens_to_vec(sentence_elements["tokens"])
            if sentence_vec == []:
                 print 'line %d sentence vector is null' % count
                 continue
            entity1_vec = self.lookup_word_pos(sentence_elements["ent1_pos"], self.m)  # dimension m x _
            entity2_vec = self.lookup_word_pos(sentence_elements["ent2_pos"], self.m)  # dimension m x _

            pad_size = self.m - sentence_vec.shape[0]
            if pad_size > 0:
                temp = np.zeros((pad_size, self.sentence_vectorizer.model.vector_size))
                sentence_vec = np.vstack([sentence_vec, temp])

            #  merging different parts of vector representation of words
            sentence_matrix_vec = np.hstack([sentence_vec, entity1_vec, entity2_vec])
            sentence_matrix_out = np.append(sentence_matrix_out, [sentence_matrix_vec], axis=0)
            valid_label.append(label)
        return sentence_matrix_out, valid_label

    def lookup_word_pos(self, p, sentence_length):
        """
        :param p: position of entity
        :return: array of dimension self.m x self.word_position_size

        example : if ent1 = 2 self.m = 10   i.e. : (w0, w1, w2(e1), w3, w4, w5, w6, w7, w8, w9)
                  return: word_position[-2:8]   ===
                  add (l-1) to get indices between (0,2l-1) ===>  word_position[7:17]
        """
        start = -p + sentence_length - 1
        end = start + sentence_length
        return self.word_position[start:end]


def trim_long_sentence(sentence, max_length, ent1_pos, ent2_pos):
    if len(sentence) < max_length or \
                    abs(ent1_pos - ent2_pos) >= max_length:
        return -1

    diff = {}
    for i in range(len(sentence)-max_length+1):
        start = i
        end = i + max_length - 1
        if ent1_pos >= start and ent2_pos <= end:
            left_margin = abs(ent1_pos - start)
            right_margin = - abs(ent2_pos - end)
            diff[i] = abs(left_margin - right_margin)
    ind = min(diff, key=lambda x: x)
    return sentence[ind: ind+max_length]


def test_trim_long_sentence():
    sentence1 = ['a', 'b', 'ent1', 'c', 'd', 'ent2', 'c']
    sentence2 = ['a', 'b', 'ent1', 'c', 'ent2']
    sentence3 = ['ent1', 'c', 'd', 'ent2']
    print trim_long_sentence(sentence1, 4, 2, 5)
    print trim_long_sentence(sentence2, 4, 2, 4)
    print trim_long_sentence(sentence2, 4, 2, 4)

    print trim_long_sentence(sentence1, 6, 2, 5)
    print trim_long_sentence(sentence2, 5, 2, 4)
    print trim_long_sentence(sentence3, 5, 0, 3)
    print trim_long_sentence(sentence3, 5, 0, 3)

    sentence1 = ['ent1', 'c', 'd', 'a', 'b', 'ent2', 'c']
    print trim_long_sentence(sentence1, 3, 0, 3)


def check_word_vector(word2vec_model_path, file_path):
    import preprocessor
    rp = preprocessor.RelationPreprocessor()
    rv = RelationVectorizer(word2vec_model_path)
    with open(file_path, "r") as f:
	text = f.readlines()
    for line in text:
	if not line:
	    contnue
        #line_clean, e1, e2, label = preprocessor.extract_info(line)
	tokens, tokens_pos = rp.tokenize(line)
        rv.word2vec_test(tokens)	

    

if __name__ == '__main__':
    # test_trim_long_sentence()
    file_path = '/home/angelzou/CNN-RelationExtraction/data/d-s-ann.0.5.txt'
    word2vec_model_path ='/home/angelzou/word2vec/wiki.medicine.zh.text.model'
    check_word_vector(word2vec_model_path, file_path)
