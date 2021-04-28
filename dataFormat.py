import re


class DataFormat():
    punc_pattern = "[\.\!\/_,;$%(^*+\"\']+|[+——！，。？、~@#￥%……&*]+|\s[\'\"]+|[).]"
    uni_word = set()
    tags = {'T', 'N'}
    sep_data = []

    # load tokenized data from file, in this function,
    # there is a optional choice to give different labels to a multi-word terminology
    def load_token_data(self, filename):
        sent_count = 0
        token_count = 0
        print('loading from file :{}'.format(filename))
        token_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if line != '\n':
                    line = line.strip().split('\t')
                    word = line[0].lower().split(' ')
                    if line[1] == 'T':
                        if len(word) > 1:
                            for i, s in enumerate(word):
                                token_count = token_count + 1
                                s = s.lower()
                                if i == 0:
                                    self.uni_word.add(s)
                                    token_data.append([s, 'T'])
                                elif i == len(word) - 1:
                                    self.uni_word.add(s)
                                    token_data.append([s, 'T'])
                                else:
                                    self.uni_word.add(s)
                                    token_data.append([s, 'T'])
                        else:
                            token_count = token_count + 1
                            self.uni_word.add(word[0].lower())
                            token_data.append([word[0].lower(), 'T'])
                    elif line[1] == "O":
                        if len(word) > 1:
                            for i, s in enumerate(word):
                                token_count = token_count + 1
                                s = s.lower()
                                if i == 0:
                                    self.uni_word.add(s)

                                    token_data.append([s, 'N'])
                                elif i == len(word) - 1:
                                    self.uni_word.add(s)

                                    token_data.append([s, 'N'])
                                else:
                                    self.uni_word.add(s)

                                    token_data.append([s, 'N'])
                        else:
                            token_count = token_count + 1
                            self.uni_word.add(word[0].lower())
                            token_data.append([word[0].lower(), 'N'])
                    else:
                        token_count = token_count + 1
                        self.uni_word.add(word[0].lower())
                        token_data.append([word[0].lower(), 'N'])
                        if word[0] == '.' or word[0] == '?':
                            sent_count = sent_count + 1
                            self.sep_data.append(token_data)
                            token_data = []

        return self.sep_data

    # Load another version of data from ACL-rd-tech version 1
    def load_version_1(self,filename):
        tech_word = set()
        text_data = []
        term_pat = '<term.*?ann="[1-2]">.*?</term>'
        word_pat = '<term.*?ann="[1-2]">(.*?)</term>'
        ecp_pat = '(\\\\)*\s-[L,R]RB-\s'
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                sent = line.strip('\n').split('\t')[1]
                sent = re.sub(ecp_pat, '', sent).replace('\/', ' ').replace('\\', '')
                word = re.findall(word_pat, sent)[0].split(' ')
                if len(word) > 1:
                    for sw in word:
                        if sw:
                            tech_word.add(sw)
                    word = '_'.join(word)
                else:
                    tech_word.add(word[0])
                    word = word[0] + '_'
                if word:
                    clr = re.sub(term_pat, word, sent)
                    text_data.append(clr)
        all_pre = []

        for sent in text_data:
            tag = []
            tmp = sent.split(' ')
            for word in tmp:
                if word.__contains__('_'):
                    word = word.replace('_', ' ').split(' ')
                    if len(word) > 1:
                        if word[1]:
                            for i, s in enumerate(word):
                                if i == 0:
                                    self.uni_word.add(s)

                                    tag.append([s, "T"])
                                else:
                                    self.uni_word.add(s)

                                    tag.append([s, "T"])
                        else:
                            self.uni_word.add(word[0])

                            tag.append([word[0], 'T'])
                else:
                    if word in tech_word:

                        tag.append([word, "T"])
                    else:
                        self.uni_word.add(word)
                        tag.append([word, "N"])
            all_pre.append(tag)
        return all_pre


'''
    def load_acl_from_file(self):
        ann_text = []
        sent = []
        start_time = datetime.datetime.now()

        for line in fileinput.input('ann_acl_1.txt'):
            line = line.strip('\n')
            if len(line) > 1:
                tmp = line.split('\t')
                self.uni_word.add(tmp[0])
                sent.append(tmp)
            else:
                ann_text.append(sent)
                sent = []

        end_time = datetime.datetime.now()
        print((end_time - start_time).seconds)
        print("load acl 1 dataset finished!")
        print(len(self.uni_word))

        return ann_text
'''

if __name__ == '__main__':
    df = DataFormat()
