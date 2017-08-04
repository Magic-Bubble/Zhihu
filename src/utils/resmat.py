import torch
import numpy as np
from writeResult import write_result
from torch.nn import functional as F
from torch.autograd import Variable

def emsemble_eval():
    result_dir = '/home/dyj/'
    label = torch.load(result_dir + 'label.pt')
    cnn1 = torch.load(result_dir + 'TextCNN1_2017-07-27#10:15:20_res.pt')
    cnn2 = torch.load(result_dir + 'TextCNN2_2017-07-27#10:32:21_res.pt')
    rnn1 = torch.load(result_dir + 'RNN1_2017-07-27#10:48:05_res.pt')
    rnn2 = torch.load(result_dir + 'RNN2_2017-07-27#10:41:03_res.pt')
    rcnn1 = torch.load(result_dir + 'RCNN1_2017-07-27#11:01:07_res.pt')
    rcnncha = torch.load(result_dir + 'RCNNcha_2017-07-27#16:19:23_res.pt')
    fasttext4 = torch.load(result_dir + 'FastText4_2017-07-28#15:14:47_res.pt')
    fasttext1 = torch.load(result_dir + 'FastText1_2017-07-29#10:31:43_res.pt')
    fasttext7 = torch.load(result_dir + 'FastText7_2017-07-30#21:07:18_res.pt')
    rnn3 = torch.load(result_dir + 'RNN3_2017-07-31#07:07:41_res.pt')
    rnn4 = torch.load(result_dir + 'RNN4_2017-07-31#12:53:56_res.pt')
    logit = (cnn1 + rnn1 + rcnn1 + rcnncha + fasttext1) / 4 + (cnn2 + rnn2) / 2 + fasttext4 / 4 + fasttext7 / 7 + rnn3 / 3
    print get_score(logit, label)
    
def emsemble_test():
    test_idx = np.load('../../data_preprocess/test/test_idx.npy')
    topic_idx = np.load('../../data_preprocess/topic/topic_idx.npy')
    
    result_dir = '/home/dyj/'
    cnn1 = torch.load(result_dir + 'TextCNN1_2017-07-27#12:30:16_test_res.pt')
    cnn1_loss_weight = torch.load(result_dir + 'TextCNN1_loss_weight.pt')
    rnn1 = torch.load(result_dir+'RNN1_2017-07-27#12:35:51_test_res.pt')
    rnn1_loss_weight = torch.load(result_dir + 'RNN1_loss_weight.pt')
    rcnn1 = torch.load(result_dir + 'RCNN1_2017-07-27#11:30:42_test_res.pt')
    rcnn1_loss_weight = torch.load(result_dir + 'RCNN1_loss_weight.pt')
    rcnncha = torch.load(result_dir + 'RCNNcha_2017-07-27#16:00:33_test_res.pt')
    rcnncha_loss_weight = torch.load(result_dir + 'RCNNcha_loss_weight.pt')
    # fasttext4 = torch.load(result_dir + 'FastText4_2017-07-28#17:20:21_test_res.pt')
    # fasttext4_loss_weight = torch.load(result_dir + 'FastText4_loss_weight.pt')
    fasttext1_char = torch.load('../results/FastText1_char_test_res.pt')
    fasttext1_loss_weight = torch.load('../snapshots/FastText/layer_2_loss_weight_char.pt')
    fasttext10 = torch.load(result_dir + 'FastText10_test_res.pt')
    fasttext10_loss_weight = torch.load(result_dir + 'FastText10_loss_weight.pt')
    # rnn10 = torch.load(result_dir + 'RNN10_test_res.pt')
    # rnn10_loss_weight = torch.load(result_dir + 'RNN10_loss_weight.pt')
    cnn3 = torch.load(result_dir + 'TextCNN3_test_res.pt')
    cnn3_loss_weight = torch.load(result_dir + 'TextCNN3_loss_weight.pt')
    # cnn7 = torch.from_numpy(np.load('/home/cuidesheng/ncnnp7.npy')).float()
    rnn10_cnn2 = torch.load('../results/RNN10_CNN2_test_res.pt')
    rnn10_cnn2_loss_weight = torch.load('../snapshots/TextCNN/layer_13_loss_weight_char.pt')
    logit = sigmoid(cnn1) * torch.sqrt(1-cnn1_loss_weight+cnn1_loss_weight.mean()).expand_as(cnn1) * 0.0066 + sigmoid(rnn1) * torch.sqrt(1-rnn1_loss_weight+rnn1_loss_weight.mean()).expand_as(rnn1) * 0.0187 + sigmoid(rcnn1) * torch.sqrt(1-rcnn1_loss_weight+rcnn1_loss_weight.mean()).expand_as(rcnn1) * 0.0058 +  sigmoid(rcnncha) * torch.sqrt(1-rcnncha_loss_weight+rcnncha_loss_weight.mean()).expand_as(rcnncha) * 0.0164 + sigmoid(fasttext1_char) * torch.sqrt(1-fasttext1_loss_weight+fasttext1_loss_weight.mean()).expand_as(fasttext1_char) * 0.0458 + sigmoid(fasttext10/10) * torch.sqrt(1-fasttext10_loss_weight+fasttext10_loss_weight.mean()).expand_as(fasttext10) * 0.0535 + sigmoid(rnn10_cnn2/12) * torch.sqrt(1-rnn10_cnn2_loss_weight+rnn10_cnn2_loss_weight.mean()).expand_as(rnn10_cnn2) * 0.5887 + sigmoid(cnn3/3) * torch.sqrt(1-cnn3_loss_weight+cnn3_loss_weight.mean()).expand_as(cnn3) * 0.1563
    predict_label_list = [list(ii) for ii in logit.topk(5, 1)[1]]
    lines = []
    for qid, top5 in zip(test_idx, predict_label_list):
        topic_ids = [topic_idx[i] for i in top5]
        lines.append('{},{}'.format(qid, ','.join(topic_ids)))

    write_result(lines, model_name='Emsemble', result_dir=result_dir)
    
def get_score(logit, label):
    predict_label_list = [list(ii) for ii in logit.topk(5, 1)[1]]
    marked_label_list = [list(np.where(ii.numpy()==1)[0]) for ii in label]
    
    right_label_num = 0
    right_label_at_pos_num = [0, 0, 0, 0, 0]
    sample_num = 0
    all_marked_label_num = 0
    for predict_labels, marked_labels in zip(predict_label_list, marked_label_list):
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
            if label in marked_label_set:
                right_label_num += 1
                right_label_at_pos_num[pos] += 1
                
    precision = 0.0
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / np.log(2.0 + pos)
    recall = float(right_label_num) / all_marked_label_num
    score = (precision * recall) / (precision + recall)

    return precision, recall, score

def get_loss_weight(logit, label):
    class_num = logit.size(1)
    predict_label_list = [list(ii) for ii in logit.topk(5, 1)[1]]
    marked_label_list = [list(np.where(ii.numpy()==1)[0]) for ii in label]
    sample_per_class = torch.zeros(class_num)
    error_per_class = torch.zeros(class_num)
    for predict_labels, marked_labels in zip(predict_label_list, marked_label_list):
        for true_label in marked_labels:
            sample_per_class[true_label] += 1
            if true_label not in predict_labels:
                error_per_class[true_label] += 1
    return error_per_class / sample_per_class

def normalize(logit):
    logit = torch.sigmoid(logit)
    logit = logit / logit.sum(1).expand_as(logit)
    return logit

def sigmoid(logit):
    return torch.sigmoid(logit)

def search_class_weight(logit, label, init, iter_num, step, save_name):
    '''
        logit: n * class_num
    '''
    class_num = logit.size(1)
    print get_score(logit, label)
    best_weight = init
    best_score = get_score(logit * best_weight.expand_as(logit), label)[2]
    print 'init', best_score
    for i in range(iter_num):
        for j in range(class_num):
            cur_weight = best_weight.clone()
            cur_weight[j] += step
            cur_score = get_score(logit * cur_weight.expand_as(logit), label)[2]
            if cur_score > best_score:
                best_weight = cur_weight.clone()
                best_score = cur_score
                torch.save(best_weight, save_name)
                print i, j, best_score
            cur_weight[j] -= 2*step
            cur_score = get_score(logit * cur_weight.expand_as(logit), label)[2]
            if cur_score > best_score:
                best_weight = cur_weight.clone()
                best_score = cur_score
                torch.save(best_weight, save_name)
                print i, j, best_score
                
def search_model_weight(logit, label, init, iter_num, step, save_name):
    '''
        logit: n * class_num * model_num
    '''
    class_num = logit.size(1)
    model_num = logit.size(2)
    best_weight = init
    best_score = get_score(torch.mm(logit.view(-1, model_num), best_weight.unsqueeze(1)).squeeze(1).view(-1, class_num), label)[2]
    print 'init', best_score
    for i in range(iter_num):
        for j in range(model_num):
            cur_weight = best_weight.clone()
            cur_weight[j] += step
            cur_score = get_score(torch.mm(logit.view(-1, model_num), cur_weight.unsqueeze(1)).squeeze(1).view(-1, class_num), label)[2]
            if cur_score > best_score:
                best_weight = cur_weight.clone()
                best_score = cur_score
                torch.save(best_weight, save_name)
                print i, j, best_score
            cur_weight[j] -= 2*step
            cur_score = get_score(torch.mm(logit.view(-1, model_num), cur_weight.unsqueeze(1)).squeeze(1).view(-1, class_num), label)[2]
            if cur_score > best_score:
                best_weight = cur_weight.clone()
                best_score = cur_score
                torch.save(best_weight, save_name)
                print i, j, best_score
                
if __name__ == '__main__':
    emsemble_test()