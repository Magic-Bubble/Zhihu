import matplotlib.pyplot as plt
import numpy as np

def getQuesInfo(fpath):
    f = open(fpath)
    cnt = 0
    idx = []
    idx, title_char, title_word, desc_char, desc_word = [], [], [], [], []
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
            desc_char.append('')
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
    f.close()
    return idx, title_char, title_word, desc_char, desc_word

def getLenInfo(arr):
    return [len([ii for ii in item.split(',') if ii != '']) for item in arr if len([ii for ii in item.split(',') if ii != '']) != 0]

def plotHist(train_len, eval_len, save_name):
    save_path = './' + save_name + '.png'
    plt.figure()
    plt.title(save_name)
    plt.xlabel("Length")
    plt.ylabel("Count")
    _, _, train = plt.hist(train_len, bins=100, normed=1, alpha=.2, color='b')
    _, _, evalu = plt.hist(eval_len, bins=100, normed=1, alpha=.2, color='r')
    plt.legend([train[0], evalu[0]], ['train', 'eval'], loc = 'upper right')
    plt.savefig(save_path)

def analysis(train_path, eval_path):
    train_idx, train_title_char, train_title_word, train_desc_char, train_desc_word = getQuesInfo(train_path)
    eval_idx, eval_title_char, eval_title_word, eval_desc_char, eval_desc_word = getQuesInfo(eval_path)
    #train_title_char_len = np.array(getLenInfo(train_title_char))
    #print np.std(train_title_char_len)
    #train_title_word_len = np.array(getLenInfo(train_title_word))
    #print np.std(train_title_word_len)
    #train_desc_char_len = np.array(getLenInfo(train_desc_char))
    #print np.std(train_desc_char_len)
    #train_desc_word_len = np.array(getLenInfo(train_desc_word))
    #print np.std(train_desc_word_len)
    #eval_title_char_len = np.array(getLenInfo(eval_title_char))
    #print np.std(eval_title_char_len)
    #eval_title_word_len = np.array(getLenInfo(eval_title_word))
    #print np.std(eval_title_word_len)
    #eval_desc_char_len = np.array(getLenInfo(eval_desc_char))
    #print np.std(eval_desc_char_len)
    #eval_desc_word_len = np.array(getLenInfo(eval_desc_word))
    #print np.std(eval_desc_word_len)
    # plotHist(getLenInfo(train_title_char), getLenInfo(eval_title_char), "question_title_char")
    # plotHist(getLenInfo(train_title_word), getLenInfo(eval_title_word), "question_title_word")
    # plotHist(getLenInfo(train_desc_char), getLenInfo(eval_desc_char), "question_desc_char")
    # plotHist(getLenInfo(train_desc_word), getLenInfo(eval_desc_word), "question_desc_word")

if __name__ == '__main__':
    train_question_file = "../ieee_zhihu_cup/question_train_set.txt"
    eval_question_file = "../ieee_zhihu_cup/question_eval_set.txt"
    #train_question_file = "../test/question_train_set.txt"
    #eval_question_file = "../test/question_eval_set.txt"
    analysis(train_question_file, eval_question_file)
