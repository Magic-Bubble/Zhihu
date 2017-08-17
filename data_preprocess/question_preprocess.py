import numpy as np

class Alphabet(dict):
    def __init__(self, start_id=1):
        self.fid = start_id
    
    def add(self, item):
        idx = self.get(item, None)
        if idx is None:
            idx = self.fid
            self[item] = idx
            self.fid += 1
        return idx
    
    def dump(self, fname):
        with open(fname, 'w') as out:
            for k in sorted(self.keys()):
                out.write("{}\t{}\n".format(k, self[k]))

def load_embed(fname):
    f = open(fname)
    cnt, vocab_size, embed_dim = 0, 0, 0
    embed_dict = {}
    print 'Load embedding file start!'
    for line in f:
        cnt += 1
        if cnt % 10000 == 0:
            print cnt
        terms = line.strip().split(' ')
        if len(terms) == 2:
            vocab_size = int(terms[0])
            embed_dim = int(terms[1])
        if len(terms) == embed_dim + 1:
            embed_dict[terms[0]] = np.array([float(ii) for ii in terms[1:]])
    print 'Load embedding file finish!'
    return embed_dict, vocab_size, embed_dim

def parse_embed(embed_file):
    alphabet, embed_mat = Alphabet(), []
    embed_dict, vocab_size, embed_dim = load_embed(embed_file)
    unknown_word_idx = 0
    embed_mat.append(np.random.uniform(-0.25, 0.25, embed_dim))
    for word in embed_dict:
        alphabet.add(word)
        embed_mat.append(embed_dict[word])
    dummy_word_idx = alphabet.fid
    embed_mat.append(np.zeros(embed_dim))
    return alphabet, embed_mat, unknown_word_idx, dummy_word_idx

char_embed_file = '../ieee_zhihu_cup/char_embedding.txt'
word_embed_file = '../ieee_zhihu_cup/word_embedding.txt'
char_alphabet, char_embed_mat, unknown_char_idx, dummy_char_idx = parse_embed(char_embed_file)
word_alphabet, word_embed_mat, unknown_word_idx, dummy_word_idx = parse_embed(word_embed_file)

def load_question(fname, split=False, rate=None):
    f = open(fname)
    cnt, idx, title_char, title_word, desc_char, desc_word = 0, [], [], [], [], []
    print 'Load question file start!'
    for line in f:
        cnt += 1
        if cnt % 10000 == 0:
            print cnt
        terms = line.strip().split('\t')
        idx.append(terms[0])
        if len(terms) == 5:
            title_char.append(terms[1])
            title_word.append(terms[2])
            desc_char.append(terms[3])
            desc_word.append(terms[4])
        elif len(terms) == 4:
            title_char.append(terms[1])
            title_word.append(terms[2])
            desc_char.append(terms[3])
            desc_word.append('')
        elif len(terms) == 3:
            title_char.append(terms[1])
            title_word.append(terms[2])
            desc_char.append('')
            desc_word.append('')
        elif len(terms) == 1:
            title_char.append('')
            title_word.append('')
            desc_char.append('')
            desc_word.append('')
    print 'Load question file finish!'
    if split:
        ids = np.arange(cnt)
        np.random.seed(1024)
        np.random.shuffle(ids)
        train_id = ids[:int(cnt*rate)]
        val_id = ids[int(cnt*rate):]
        print "Finished"
        return [idx[i] for i in train_id], [title_char[i] for i in train_id], \
                [title_word[i] for i in train_id], [desc_char[i] for i in train_id], \
                [desc_word[i] for i in train_id], \
                [idx[i] for i in val_id], [title_char[i] for i in val_id], \
                [title_word[i] for i in val_id], [desc_char[i] for i in val_id], \
                [desc_word[i] for i in val_id]
    print "Finished"
    return idx, title_char, title_word, desc_char, desc_word

def convert2indices(data, alphabet, unknown_word_idx, dummy_word_idx, max_length):
    data_idx = []
    for item in data:
        item_arr = [ii for ii in item.split(',') if ii != '']
        ex = np.ones(max_length) * dummy_word_idx
        for i, word in enumerate(item_arr):
            if i >= max_length:
                break
            idx = alphabet.get(word, unknown_word_idx)
            ex[i] = idx
        data_idx.append(ex)
    data_idx = np.array(data_idx).astype('int32')
    return data_idx

question_train_file = '../ieee_zhihu_cup/question_train_set.txt'
question_test_file = '../ieee_zhihu_cup/question_eval_set.txt'
train_idx, train_title_char, train_title_word, train_desc_char, train_desc_word, \
val_idx, val_title_char, val_title_word, val_desc_char, val_desc_word \
            = load_question(question_train_file, split=True, rate=0.9)
train_idx_all, train_title_char_all, train_title_word_all, train_desc_char_all, \
                train_desc_word_all = load_question(question_train_file)
test_idx, test_title_char, test_title_word, test_desc_char, test_desc_word \
                           = load_question(question_test_file)

title_char_max_length = 85#22
title_word_max_length = 50#30#13
desc_char_max_length = 300#117
desc_word_max_length = 150#120#58

train_title_char_indices = convert2indices(train_title_char, char_alphabet, \
                                           unknown_char_idx, dummy_char_idx, \
                                           title_char_max_length)
train_title_word_indices = convert2indices(train_title_word, word_alphabet, \
                                          unknown_word_idx, dummy_word_idx, \
                                          title_word_max_length)
train_desc_char_indices = convert2indices(train_desc_char, char_alphabet, \
                                           unknown_char_idx, dummy_char_idx, \
                                           desc_char_max_length)
train_desc_word_indices = convert2indices(train_desc_word, word_alphabet, \
                                          unknown_word_idx, dummy_word_idx, \
                                          desc_word_max_length)

val_title_char_indices = convert2indices(val_title_char, char_alphabet, \
                                           unknown_char_idx, dummy_char_idx, \
                                           title_char_max_length)
val_title_word_indices = convert2indices(val_title_word, word_alphabet, \
                                          unknown_word_idx, dummy_word_idx, \
                                          title_word_max_length)
val_desc_char_indices = convert2indices(val_desc_char, char_alphabet, \
                                           unknown_char_idx, dummy_char_idx, \
                                           desc_char_max_length)
val_desc_word_indices = convert2indices(val_desc_word, word_alphabet, \
                                          unknown_word_idx, dummy_word_idx, \
                                          desc_word_max_length)

basedir = './embed'
np.save('{}/char_embed_mat.npy'.format(basedir), char_embed_mat)
del char_embed_mat
np.save('{}/word_embed_mat.npy'.format(basedir), word_embed_mat)
del word_embed_mat
print 'Save embedding finished!'
basedir = './train'
np.save('{}/train_idx.npy'.format(basedir), train_idx)
del train_idx
np.save('{}/train_title_char_indices_2.npy'.format(basedir), train_title_char_indices)
del train_title_char_indices
np.save('{}/train_title_word_indices_2.npy'.format(basedir), train_title_word_indices)
del train_title_word_indices
np.save('{}/train_desc_char_indices_2.npy'.format(basedir), train_desc_char_indices)
del train_desc_char_indices
np.save('{}/train_desc_word_indices_2.npy'.format(basedir), train_desc_word_indices)
del train_desc_word_indices
print 'Save train data finished!'
basedir = './val'
np.save('{}/val_idx.npy'.format(basedir), val_idx)
del val_idx
np.save('{}/val_title_char_indices_2.npy'.format(basedir), val_title_char_indices)
del val_title_char_indices
np.save('{}/val_title_word_indices_2.npy'.format(basedir), val_title_word_indices)
del val_title_word_indices
np.save('{}/val_desc_char_indices_2.npy'.format(basedir), val_desc_char_indices)
del val_desc_char_indices
np.save('{}/val_desc_word_indices_2.npy'.format(basedir), val_desc_word_indices)
del val_desc_word_indices
print 'Save val data finished!'

train_title_char_indices_all = convert2indices(train_title_char_all, char_alphabet, \
                                           unknown_char_idx, dummy_char_idx, \
                                           title_char_max_length)
train_title_word_indices_all = convert2indices(train_title_word_all, word_alphabet, \
                                          unknown_word_idx, dummy_word_idx, \
                                          title_word_max_length)
train_desc_char_indices_all = convert2indices(train_desc_char_all, char_alphabet, \
                                           unknown_char_idx, dummy_char_idx, \
                                           desc_char_max_length)
train_desc_word_indices_all = convert2indices(train_desc_word_all, word_alphabet, \
                                          unknown_word_idx, dummy_word_idx, \
                                          desc_word_max_length)

test_title_char_indices = convert2indices(test_title_char, char_alphabet, \
                                           unknown_char_idx, dummy_char_idx, \
                                           title_char_max_length)
test_title_word_indices = convert2indices(test_title_word, word_alphabet, \
                                          unknown_word_idx, dummy_word_idx, \
                                          title_word_max_length)
test_desc_char_indices = convert2indices(test_desc_char, char_alphabet, \
                                           unknown_char_idx, dummy_char_idx, \
                                           desc_char_max_length)
test_desc_word_indices = convert2indices(test_desc_word, word_alphabet, \
                                          unknown_word_idx, dummy_word_idx, \
                                          desc_word_max_length)

basedir = './train_all'
np.save('{}/train_idx_all.npy'.format(basedir), train_idx_all)
np.save('{}/train_title_char_indices_all_2.npy'.format(basedir), train_title_char_indices_all)
np.save('{}/train_title_word_indices_all_2.npy'.format(basedir), train_title_word_indices_all)
np.save('{}/train_desc_char_indices_all_2.npy'.format(basedir), train_desc_char_indices_all)
np.save('{}/train_desc_word_indices_all_2.npy'.format(basedir), train_desc_word_indices_all)
print 'Save train all data finished!'
basedir = './test'
np.save('{}/test_idx.npy'.format(basedir), test_idx)
np.save('{}/test_title_char_indices_2.npy'.format(basedir), test_title_char_indices)
np.save('{}/test_title_word_indices_2.npy'.format(basedir), test_title_word_indices)
np.save('{}/test_desc_char_indices_2.npy'.format(basedir), test_desc_char_indices)
np.save('{}/test_desc_word_indices_2.npy'.format(basedir), test_desc_word_indices)
print 'Save test data finished!'