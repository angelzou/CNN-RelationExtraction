__author__ = 'hadyelsahar'

import re
import numpy as np
from nltk.tokenize import TreebankWordTokenizer


class RelationPreprocessor:

    def __init__(self, max_token_size):
        self.out_label = "OUT"
        self.max_token_size = max_token_size

    def load_data(self, file_path):
        print 'processing file: {}'.format(file_path)
        file_text = file(file_path, 'r')
        text = file_text.read()
        rels, rel_labels = self.get_relations(text)
        file_text.close()
        lines_info = np.array(rels)
        labels = np.array(rel_labels)
        return lines_info, labels

    def get_relations(self, raw_txt):
        txt_lines = raw_txt.split("\n")
        y = []
        X = []
        for line in txt_lines:
            if not line:
                continue
            line_clean, e1, e2, label = extract_info(line)
            tokens, tokens_pos = self.tokenize(line_clean)
            print line_clean
            id_e1, id_e2 = tokens.index(e1), tokens.index(e2)
            if len(tokens) > self.max_token_size:
                tokens, tokens_pos = trim_token(tokens, tokens_pos, self.max_token_size, id_e1, id_e2)
            if not tokens:
                continue

            x = {"tokens": tokens, "token_pos": tokens_pos,
                 "ent1": e1, "ent2": e2,
                 "ent1_pos": tokens.index(e1), "ent2_pos": tokens.index(e2)}
            X.append(x)
            y.append(label)
        return X, y

    def tokenize(self, text, returnids=True):
        """
        adaptation of Treebanktokenizer to allow start and end positions of each token of sentences
        :param s: seting sentence
        :param returnids: if true return a tuple of array of tokens and array of tuples containing start and
                        end positions of each tokens [tokens],ids[(start,end)])
                        eg.  sentence : "hello hi a" ["hello","hi","a"] [(0,5),(6,8),(9,10)]
        :return:
        """
        if returnids:
            tokens = TreebankWordTokenizer().tokenize(text)
            positions = []
            start = 0
            for token in tokens:
                positions.append((start, start+len(token)))
                start = start+len(token)+1
            return tokens, positions
        else:
            TreebankWordTokenizer().tokenize(text)


def trim_token(tokens, tokens_pos, max_token_size, id_e1, id_e2):
    trimed_tokens, trimed_ind = \
        trim_long_sentence(tokens, max_token_size, id_e1, id_e2)
    if not trimed_tokens:
        return [], []

    trimed_pos = [tokens_pos[ind] for ind in trimed_ind]
    tokens_pos = let_pos_from_zero(trimed_pos)
    tokens = trimed_tokens
    return tokens, tokens_pos


def let_pos_from_zero(trimed_pos):
    ind_start = trimed_pos[0][0]
    if ind_start != 0:
        trimed_pos = [(x-ind_start, y-ind_start) for x, y in trimed_pos]
    return trimed_pos


def extract_entity(sentence):
    # TODO make re compile global or private to get better performance
    entity_a_raw = re.compile(r"(?<=<e1>)(.*?)(?=</e1>)").findall(sentence)[0]
    entity_b_raw = re.compile(r"(?<=<e2>)(.*?)(?=</e2>)").findall(sentence)[0]

    sentence = sentence.replace('<e1>', ' ')
    sentence = sentence.replace('<e2>', ' ')
    sentence = sentence.replace('</e1>', ' ')
    sentence = sentence.replace('</e2>', ' ')
    entity_a = entity_a_raw.replace(' ', '_')
    entity_b = entity_b_raw.replace(' ', '_')
    sentence = sentence.replace(entity_a_raw, entity_a)
    sentence = sentence.replace(entity_b_raw, entity_b)
    return sentence, entity_a, entity_b


def extract_info(line):
    label, sentence = line.split('\t')
    sentence, entity_a, entity_b = extract_entity(sentence)
    return sentence, entity_a, entity_b, label


def trim_long_sentence(sentence, max_length, ent1_pos, ent2_pos):#, token_pos):
    if len(sentence) < max_length or \
                    abs(ent1_pos - ent2_pos) >= max_length:
        return [], []

    diff = {}
    for i in range(len(sentence)-max_length+1):
        start = i
        end = i + max_length - 1
        if ent1_pos >= start and ent2_pos <= end:
            left_margin = abs(ent1_pos - start)
            right_margin = abs(ent2_pos - end)
            diff[i] = abs(left_margin - right_margin)
    ind = min(diff, key=diff.get)
    return sentence[ind: ind+max_length], range(ind, ind+max_length)


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
    sentence4 = ['ent1', 'ent2', 'c', 'd', 'a', 'b', 'c']
    print trim_long_sentence(sentence4, 4, 0, 1)


def test_get_ralations():
    rp = RelationPreprocessor(max_token_size=20)
    rp.load_data('/Users/liujiaxiang/code/nlp/CNN-RelationExtraction/example/data/sentence.txt')

if __name__ == '__main__':
    #test_trim_long_sentence()
    test_get_ralations()




'''
def get_relations(self, raw_txt, annotation_txt, label_dict_txt, file_name):
    txt_lines = raw_txt.split("\n")

    relation_lines = annotation_txt.split("\n")
    relations = [line.split("\t") for line in relation_lines]

    label_dict_lines = label_dict_txt.split("\n")
    label_dict = {}
    for label_dict_line in label_dict_lines:
        label_name, label_no = label_dict_line.split('\t')
        label_dict[label_name] = label_no

    if len(txt_lines) != len(relation_lines):
        print 'source txt lines{} not equal to annotation lines{}.'\
            .format(len(txt_lines), len(relation_lines))
        return -1

    y = []
    X = []
    for (line, ann) in zip(txt_lines, relations):
        segment_of_lines = []
        labels = []
        # TODO make re compile global or private to get better performance
        entity_a_raw = re.compile(r"(?<=<e1>)(.*?)(?=</e1>)").findall(line)[0]
        entity_b_raw = re.compile(r"(?<=<e2>)(.*?)(?=</e2>)").findall(line)[0]

        line = line.replace('<e1>', '')
        line = line.replace('<e2>', '')
        line = line.replace('</e1>', '')
        line = line.replace('</e2>', '')
        entity_a = entity_a_raw.replace(' ', '_')
        entity_b = entity_b_raw.replace(' ', '_')
        line = line.replace(entity_a_raw, entity_a)
        line = line.replace(entity_b_raw, entity_b)

        tokens, tokens_pos = self.tokenize(line)

        entity_a_ind = -1
        entity_b_ind = -1
        for i, (start, end) in enumerate(tokens_pos):
            # if start and end of a token not in the tagged range add token to segment_of_lines with out tag
            # if len(set(range(start,end)) & set(tagged_range)) == 0:
            seg_id = "T%s" % str(i)
            segment_of_lines.append([seg_id, self.out_label, start, end, tokens[i]])
            labels.append(self.out_label)
            if tokens[i] == entity_a:
                entity_a_ind = i
            if tokens[i] == entity_b:
                entity_b_ind = i

        label_str = ann[0].strip()
        if label_str not in label_dict:
            print 'unrecognized label {}. line: '.format(label_str, line)
            continue
        label_no = label_dict[label_str]
        y.append(label_no)
        x = {"sentence_id": file_name,
             "segments": segment_of_lines, "segment_labels": labels,
             "ent1": entity_a, "ent2": entity_b,
             "ent1_pos": entity_a_ind, "ent2_pos": entity_b_ind}
        X.append(x)

    return X, y
    '''
