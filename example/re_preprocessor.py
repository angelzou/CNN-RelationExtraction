__author__ = 'hadyelsahar'

import os
import sys
import re
import numpy as np
from nltk.tokenize import TreebankWordTokenizer


class RelationPreprocessor:
    """
    RelationPreprocessor class is the class responsible of converting the brat annotation
    files format of relation mentions into standard format to be processed by the RelationMentionVectorizer class
    with availability to save the preprocessed dataset into a pickle file in the 'data' folder
    """
    def __init__(self, input_dir):
        """
        :param input_dir: directory where all .ann and all .txt files exists
                    - for each sentence in brat annotaiton tool there are two files .txt and .ann
        :return: self
        """

        self.out_label = "OUT"

        if input_dir is not None:
            print "reading data..."
            # S     :   dict {id:sentence}  the raw sentences
            # X: array(dict) [{id:, segments:[], segment_labels:[], ent1:int, ent2:int}]
            # y     :   the correct labels                           - shape: nnx1
            self.raw_text, self.lines_info, self.labels = self.read_data(input_dir)
            print "done"

    def read_data(self, input_dir):
        """
        :param input_dir: directory contains brat annotation files
        Two files per sentence ending with ".txt" or ".ann"

        .txt : text file contains one sentence per files
        .ann : annotation file contains one Tag or relation per line
        for the scope of phrase extraction we wil concentrate only on lines starting with T
        which are the labeling for segment_of_lines

        :return: (S,Xid,y)
         S     :   dict {id:sentence}  the raw sentences
         X: array(dict) [{sentence_id:, id:, segments:[], segment_labels:[], ent1:int, ent2:int}]
         y:   array containing all labels for X in order
        """

        files = os.listdir(input_dir)

        # select only files with annotation existing
        # get the basename without the file type
        # add .txt or .ann later
        file_names = [x.replace(".ann", "") for x in files if ".ann" in x]
        file_names.sort()

        raw_text = {}
        lines_info = []
        labels = []

        for file_name in file_names:
            print 'processing file: {}'.format(file_name)
            # collect text sentences tokens
            file_text = file("%s/%s.txt" % (input_dir, file_name), 'r')
            text = file_text.read()

            file_ann = file("%s/%s.ann" % (input_dir, file_name), 'r')
            annotation = file_ann.read()
            raw_text[file_name] = text

            # reading segments and their labels
            #segment_of_lines = self.get_segments(text)
            # reading relations and Generation training data
            rels, labels = self.get_relations(text, annotation, file_name)

            lines_info += rels
            labels += labels

            file_text.close()
            file_ann.close()

        raw_text = np.array(raw_text)
        lines_info = np.array(lines_info)
        labels = np.array(labels)

        return raw_text, lines_info, labels

    def get_segments(self, txt):
        """
        return array of sorted segment_of_lines with (OUT) segment_of_lines in between for untagged segments
        :param txt: string of .txt file
        :return: array of segment_of_lines in order [["T1", "Subject", "0", "8", "Michalka wonka"], .... ]
        """

        # collect segment_of_lines in the annotation by selecting lines only that contain segment_of_lines
        segment_of_lines = []
        txt_lines = txt.split("\n")
        for line in txt_lines:
            tokens, tokens_pos = self.tokenize(line)
            for i, (start, end) in enumerate(tokens_pos):
                # if start and end of a token not in the tagged range add token to segment_of_lines with out tag
                # if len(set(range(start,end)) & set(tagged_range)) == 0:
                seg_id = "T%s" % str(i)
                segment_of_lines.append([seg_id, self.out_label, start, end, tokens[i]])

        # eg. ["T1", "Subject", "0", "8", "Michalka wonka"]
        # segment_of_lines = sorted(segment_of_lines, key=lambda l: l[2])
        return segment_of_lines

    def get_relations_dep(self, ann, segment_of_lines, file_name):
        """
        :param ann: text from brat .ann (annotation file)
        :param segment_of_lines: segment_of_lines extracted from self.get_segment_of_lines method
                    ["T1", "Subject", "0", "8", "Michalka wonka"]
        :return: [{sentence_id:, id:, segments:[], segment_labels:[], ent1:int, ent2:int}], [labels]
        """
        X = []
        y = []
        # collect relations in the annotation file by selecting lines only that contain relations
        relation_lines = ann.split("\n")
        relations = [line.split("\t") for line in relation_lines]

        for relation in relations:
            y.append(relation[0].strip())

            entity_a = relation[1].strip()
            entity_b = relation[2].strip()

            segments = [i[-1] for i in segment_of_lines]
            segment_labels = [i[1] for i in segment_of_lines]

            for index, tag in enumerate(segment_of_lines):
                if tag[-1] == entity_a:
                    ent1 = index
                if tag[-1] == entity_b:
                    ent2 = index
            x = {"sentence_id": file_name,
                 "segments": segments, "segment_labels": segment_labels,
                 "ent1": ent1, "ent2": ent2}
            X.append(x)

        return X, y

    def get_relations(self, raw_txt, annotation, file_name):
        '''
        :return: array of segment_of_lines in order [["T1", "Subject", "0", "8", "Michalka wonka"], .... ]
        :return: [{sentence_id:, id:, segments:[], segment_labels:[], ent1:int, ent2:int}], [labels]
         S     :   dict {id:sentence}  the raw sentences
         X: array(dict) [{sentence_id:, id:, segments:[], segment_labels:[], ent1:int, ent2:int}]
         y:   array containing all labels for X in order
        '''
        # collect relations in the annotation file by selecting lines only that contain relations
        txt_lines = raw_txt.split("\n")
        relation_lines = annotation.split("\n")
        relations = [line.split("\t") for line in relation_lines]

        if len(txt_lines) != len(relation_lines):
            sys.stderr.write('source txt lines{} not equal to annotation lines{}.'
                             .format(len(txt_lines), len(relation_lines)))
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

            y.append(ann[0].strip())
            x = {"sentence_id": file_name,
                 "segments": segment_of_lines, "segment_labels": labels,
                 "ent1": entity_a, "ent2": entity_b,
                 "ent1_pos": entity_a_ind, "ent2_pos": entity_b_ind}
            X.append(x)

        # eg. ["T1", "Subject", "0", "8", "Michalka wonka"]
        # segment_of_lines = sorted(segment_of_lines, key=lambda l: l[2])

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
            TreebankWordTokenizer().tokenize(s)

#p = RelationPreprocessor()