import numpy as np

# class Alphabet(dict):
#     def __init__(self, start_id=1):
#         self.fid = start_id
    
#     def add(self, item):
#         idx = self.get(item, None)
#         if idx is None:
#             idx = self.fid
#             self[item] = idx
#             self.fid += 1
#         return idx
    
#     def dump(self, fname):
#         with open(fname, 'w') as out:
#             for k in sorted(self.keys()):
#                 out.write("{}\t{}\n".format(k, self[k]))

# def load_embed(fname):
#     f = open(fname)
#     cnt, vocab_size, embed_dim = 0, 0, 0
#     embed_dict = {}
#     print 'Load embedding file start!'
#     for line in f:
#         cnt += 1
#         if cnt % 10000 == 0:
#             print cnt
#         terms = line.strip().split(' ')
#         if len(terms) == 2:
#             vocab_size = int(terms[0])
#             embed_dim = int(terms[1])
#         if len(terms) == embed_dim + 1:
#             embed_dict[terms[0]] = np.array([float(ii) for ii in terms[1:]])
#     print 'Load embedding file finish!'
#     return embed_dict, vocab_size, embed_dim

# def parse_embed(embed_file):
#     alphabet, embed_mat = Alphabet(), []
#     embed_dict, vocab_size, embed_dim = load_embed(embed_file)
#     unknown_word_idx = 0
#     embed_mat.append(np.random.uniform(-0.25, 0.25, embed_dim))
#     for word in embed_dict:
#         alphabet.add(word)
#         embed_mat.append(embed_dict[word])
#     dummy_word_idx = alphabet.fid
#     embed_mat.append(np.zeros(embed_dim))
#     return alphabet, embed_mat, unknown_word_idx, dummy_word_idx

# char_embed_file = '../ieee_zhihu_cup/char_embedding.txt'
# word_embed_file = '../ieee_zhihu_cup/word_embedding.txt'
# char_alphabet, char_embed_mat, unknown_char_idx, dummy_char_idx = parse_embed(char_embed_file)
# word_alphabet, word_embed_mat, unknown_word_idx, dummy_word_idx = parse_embed(word_embed_file)

def load_topic(fname):
    f = open(fname)
    cnt, idx, pidx, title_char, title_word, desc_char, desc_word = 0, [], [], [], [], [], []
    print "Load topic start!"
    for line in f:
        cnt += 1
        if cnt % 100 == 0:
            print cnt
        terms = line.strip().split('\t')
        idx.append(terms[0])
        if len(terms) == 6:
            pidx.append(terms[1])
            title_char.append(terms[2])
            title_word.append(terms[3])
            desc_char.append(terms[4])
            desc_word.append(terms[5])
        elif len(terms) == 5:
            pidx.append(terms[1])
            title_char.append(terms[2])
            title_word.append(terms[3])
            desc_char.append(terms[4])
            desc_word.append('')
        elif len(terms) == 4:
            pidx.append(terms[1])
            title_char.append(terms[2])
            title_word.append(terms[3])
            desc_char.append('')
            desc_word.append('')
    print "Load topic finish!"
    return idx, pidx, title_char, title_word, desc_char, desc_word

# def getTopicid(data, idx_dict):
#     data_idx = []
#     for item in data:
#         item_arr = [ii for ii in item.split(',') if ii != '']
#         ex = np.zeros(len(item_arr))
#         for i, topic in enumerate(item_arr):
#             idx = idx_dict.get(topic)
#             ex[i] = idx
#         data_idx.append(ex.astype('int32'))
#     return data_idx

# def convert2indices(data, alphabet, unknown_word_idx, dummy_word_idx, max_length):
#     data_idx = []
#     for item in data:
#         item_arr = [ii for ii in item.split(',') if ii != '']
#         ex = np.ones(max_length) * dummy_word_idx
#         for i, word in enumerate(item_arr):
#             if i >= max_length:
#                 break
#             idx = alphabet.get(word, unknown_word_idx)
#             ex[i] = idx
#         data_idx.append(ex)
#     data_idx = np.array(data_idx).astype('int32')
#     return data_idx

topic_file = '../ieee_zhihu_cup/topic_info.txt'
topic_idx, topic_pidx, topic_title_char, topic_title_word, topic_desc_char, \
                                topic_desc_word = load_topic(topic_file)

# topic_idx_dict = {tid:i for i, tid in enumerate(topic_idx)}
# topic_pidx_indices = getTopicid(topic_pidx, topic_idx_dict)

# topic_title_char_max_length = 4
# topic_title_word_max_length = 1
# topic_desc_char_max_length = 64
# topic_desc_word_max_length = 31

# topic_title_char_indices = convert2indices(topic_title_char, char_alphabet, \
                                           # unknown_char_idx, dummy_char_idx, \
#                                            topic_title_char_max_length)
# topic_title_word_indices = convert2indices(topic_title_word, word_alphabet, \
#                                            unknown_word_idx, dummy_word_idx, \
#                                            topic_title_word_max_length)
# topic_desc_char_indices = convert2indices(topic_desc_char, char_alphabet, \
#                                            unknown_char_idx, dummy_char_idx, \
#                                            topic_desc_char_max_length)
# topic_desc_word_indices = convert2indices(topic_desc_word, word_alphabet, \
#                                            unknown_word_idx, dummy_word_idx, \
#                                            topic_desc_word_max_length)

basedir = './topic'
np.save('{}/topic_idx.npy'.format(basedir), topic_idx)
# np.save('{}/topic_pidx_indices.npy'.format(basedir), topic_pidx_indices)
# np.save('{}/topic_title_char_indices.npy'.format(basedir), topic_title_char_indices)
# np.save('{}/topic_title_word_indices.npy'.format(basedir), topic_title_word_indices)
# np.save('{}/topic_desc_char_indices.npy'.format(basedir), topic_desc_char_indices)
# np.save('{}/topic_desc_word_indices.npy'.format(basedir), topic_desc_word_indices)