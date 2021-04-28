import re
import os
import numpy as np
from collections import defaultdict


class DataHandler():
    _sent_pattern = '<S>(.*?)</S>'
    _term_pattern = '<term class="tech">(.*?)</term>'
    _other_pattern = '<term class="other">(.*?)</term>'
    clear_text = []
    all_terms = []
    _file_path_array = []
    _annotated_text = []
    _text_with_ann = []
    word_to_index = dict()
    index_to_embedding = np.array([])
    tagged_text = []

    tok_pat = '<token.*?>(.*?)</token>'
    term_pat = '<term.*?="tech".*?>(.*?)</term>'
    other_pat = '<term.*?="other".*?>(.*?)</term>'
    token_files = []
    sep_tokens = []

    # Load word embedding file from Glove folder
    def load_word_embedding(self, glove_path):
        self.word_to_index, self.index_to_embedding = self.load_embedding_from_file(glove_path, with_indexes=True)
        print('Load glove embedding finished')

    # Go through the source data folder to find all target filenames.
    def _set_file_path(self, path):
        for root, dirs, files in os.walk(path):
            file_path = [root + '\\' + f for f in files]
            if file_path:
                self._file_path_array.append(file_path)

    def _read_file(self, path):
        with open(path) as f:
            return f.readlines()

    # read tokenized file from dictionary
    def set_tokens(self, path):
        self._set_file_path(path)
        for dirs in self._file_path_array:
            for file in dirs:
                text = self._read_file(file)
                self.token_files.append(text)
                sentence = str(text)
                term_token = re.findall(self.term_pat, sentence)
                tokens = re.findall(self.tok_pat, sentence)
                all_tok = ' '.join(tokens)
                for s in term_token:
                    p = re.findall(self.tok_pat, s)
                    term = ' '.join(p)
                    tmp = '+'.join(p)
                    if term in all_tok:
                        all_tok = all_tok.replace(term, tmp)
                other_pat = re.findall(self.other_pat, sentence)
                for op in other_pat:
                    o = re.findall(self.tok_pat, op)
                    o_term = ' '.join(o)
                    o_tmp = '='.join(o)
                    if o_term in all_tok:
                        all_tok = all_tok.replace(o_term, o_tmp)
                self.sep_tokens.append(all_tok.split(' '))

    # read annotated source file from dictionary
    def set_source_text(self):
        for dirs in self._file_path_array:
            for file in dirs:
                text = self._read_file(file)
                self._annotated_text.append(text)

        for str in self._annotated_text:
            cleaned = re.findall(self._sent_pattern, str[0])
            self._text_with_ann.append('\n'.join(cleaned).strip())

        reg = re.compile('<[^>]*>')

        for text in self._text_with_ann:
            plain = reg.sub('', text)
            self.clear_text.append(plain)
            r1 = re.findall(self._term_pattern, text)
            r2 = re.findall(self._other_pattern, text)
            self.all_terms = self.all_terms + r1
            for term in r1:
                tag_term = term.replace(' ', '_')
                plain = plain.replace(term, tag_term)
            for term in r2:
                oth_term = term.replace(' ', '+')
                plain = plain.replace(term, oth_term)
            self.tagged_text.append(plain)

    # write all preprocessed data to the file and used later
    def write_tokens_to_file(self, output):
        token_count = 0
        with open(output, 'w', encoding='utf-8') as out:
            for file in self.sep_tokens:
                for tok in file:
                    if '+' in tok:
                        tmp = tok.replace('+', ' ')
                        out.write(tmp + '\t' + 'T' + '\n')
                        token_count = token_count + 1
                    elif '=' in tok:
                        tmp = tok.replace('+', ' ')
                        out.write(tmp + '\t' + 'O' + '\n')
                        token_count = token_count + 1
                    else:
                        out.write(tok + '\t' + 'N' + '\n')
                        token_count = token_count + 1
                out.write('\n')

    # Read the embedding file
    def load_embedding_from_file(self, glove_filename, with_indexes=True):
        if with_indexes:
            word_to_index_dict = dict()
            index_to_embedding_array = []
        else:
            word_to_embedding_dict = dict()

        with open(glove_filename, 'r', encoding='utf-8') as glove_file:
            for (i, line) in enumerate(glove_file):

                split = line.split(' ')

                word = split[0]

                representation = split[1:]
                representation = np.array(
                    [float(val) for val in representation]
                )

                if with_indexes:
                    word_to_index_dict[word] = i
                    index_to_embedding_array.append(representation)
                else:
                    word_to_embedding_dict[word] = representation

        _WORD_NOT_FOUND = [0.0] * len(representation)  # Empty representation for unknown words.
        if with_indexes:
            _LAST_INDEX = i + 1
            word_to_index_dict = defaultdict(lambda: _LAST_INDEX, word_to_index_dict)
            index_to_embedding_array = np.array(index_to_embedding_array + [_WORD_NOT_FOUND])
            return word_to_index_dict, index_to_embedding_array
        else:
            word_to_embedding_dict = defaultdict(lambda: _WORD_NOT_FOUND)
            return word_to_embedding_dict


if __name__ == '__main__':
    dh = DataHandler()
    dh.set_tokens('tokenized')
