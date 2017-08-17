import numpy as np

def load_label(fname, split=False, rate=None):
    f = open(fname)
    cnt, idx, tidx = 0, [], []
    for line in f:
        cnt += 1
        if cnt % 10000 == 0:
            print cnt
        terms = line.strip().split('\t')
        idx.append(terms[0])
        if len(terms) == 2:
            tidx.append(terms[1])
    if split:
        ids = np.arange(cnt)
        np.random.seed(1024)
        np.random.shuffle(ids)
        train_id = ids[:int(cnt*rate)]
        val_id = ids[int(cnt*rate):]
        return [tidx[i] for i in train_id], [tidx[i] for i in val_id]
    return tidx

def getLabelid(data, topic_dict):
    data_idx = []
    for item in data:
        item_arr = [ii for ii in item.split(',') if ii != '']
        ex = np.zeros(len(item_arr))
        for i, topic in enumerate(item_arr):
            idx = topic_dict.get(topic)
            ex[i] = idx
        data_idx.append(ex.astype('int32'))
    return data_idx

label_file = '../ieee_zhihu_cup/question_topic_train_set.txt'
train_label_idx, val_label_idx = load_label(label_file, split=True, rate=0.9)
train_label_idx_all = load_label(label_file)

topic_idx = list(np.load('../data_preprocess/topic/topic_idx.npy'))
topic_idx_dict = {tid:i for i, tid in enumerate(topic_idx)}

train_label = getLabelid(train_label_idx, topic_idx_dict)
print 'Get train label finished!'
val_label = getLabelid(val_label_idx, topic_idx_dict)
print 'Get val label finished!'
train_label_all = getLabelid(train_label_idx_all, topic_idx_dict)
print 'Get train label all finished!'

basedir = './train'
np.save('{}/train_label_indices.npy'.format(basedir), train_label)
print 'Save train label finished!'
basedir = './val'
np.save('{}/val_label_indices.npy'.format(basedir), val_label)
print 'Save val label finished!'
basedir = './train_all'
np.save('{}/train_label_indices_all.npy'.format(basedir), train_label_all)
print 'Save train label all finished!'