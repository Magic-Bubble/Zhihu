import matplotlib.pyplot as plt
import numpy as np

def getTopicInfo(fpath):
    f = open(fpath)
    cnt = 0
    idx, pidx, title_char, title_word, desc_char, desc_word = [], [], [], [], [], []
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
    f.close()
    return idx, pidx, title_char, title_word, desc_char, desc_word

def getLenInfo(arr):
    return [len([ii for ii in item.split(',') if ii != '']) for item in arr]

def plotHist(lenArray, save_name):
    save_path = './' + save_name + '.png'
    plt.figure()
    plt.title(save_name)
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.hist(lenArray, bins=100, normed=1, alpha=.2)
    plt.savefig(save_path)

def analysis(fpath):
    idx, pidx, title_char, title_word, desc_char, desc_word = getTopicInfo(fpath)
    pidx_len = np.array(getLenInfo(pidx))
    print np.average(pidx_len)
    title_char_len = np.array(getLenInfo(title_char))
    print np.average(title_char_len)
    title_word_len = np.array(getLenInfo(title_word))
    print np.average(title_word_len)
    desc_char_len = np.array(getLenInfo(desc_char))
    print np.average(desc_char_len)
    desc_word_len = np.array(getLenInfo(desc_word))
    print np.average(desc_word_len)
    #plotHist(getLenInfo(pidx), "topic_parent_topic")
    #plotHist(getLenInfo(title_char), "topic_title_char")
    #plotHist(getLenInfo(title_word), "topic_title_word")
    #plotHist(getLenInfo(desc_char), "topic_desc_char")
    #plotHist(getLenInfo(desc_word), "topic_desc_word")

if __name__ == '__main__':
    topic_info_file = "../ieee_zhihu_cup/topic_info.txt"
    analysis(topic_info_file)
