import matplotlib.pyplot as plt
import numpy as np

def getLabelInfo(fpath):
    f = open(fpath)
    cnt = 0
    idx, tidx = set(), []
    for line in f:
        cnt += 1
        if cnt % 10000 == 0:
            print cnt
        terms = line.strip().split('\t')
        idx.add(terms[0])
        if len(terms) == 2:
            tidx.append(terms[1])
    f.close()
    return idx, tidx

def getLenInfo(arr):
    return [len([ii for ii in item.split(',') if ii != '']) for item in arr]

def plotHist(lenArray, save_name):
    save_path = './' + save_name + '.png'
    plt.figure()
    plt.title(save_name)
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.xlim(-2,10)
    plt.hist(lenArray, bins=100, normed=1, alpha=.5)
    plt.savefig(save_path)

def analysis(fpath):
    idx, tidx = getLabelInfo(fpath)
    label_len = np.array(getLenInfo(tidx))
    print np.std(label_len)
    # plotHist(getLenInfo(tidx), "question_topic")

label_file = "../ieee_zhihu_cup/question_topic_train_set.txt"
analysis(label_file)
