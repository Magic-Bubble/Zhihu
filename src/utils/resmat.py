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
    cnn2 = torch.load(result_dir + 'TextCNN2_2017-07-27#12:22:42_test_res.pt')
    rnn1 = torch.load(result_dir + 'RNN1_2017-07-27#12:35:51_test_res.pt')
    rnn2 = torch.load(result_dir + 'RNN2_2017-07-27#11:33:24_test_res.pt')
    rcnn1 = torch.load(result_dir + 'RCNN1_2017-07-27#11:30:42_test_res.pt')
    rcnncha = torch.load(result_dir + 'RCNNcha_2017-07-27#16:00:33_test_res.pt')
    fasttext4 = torch.load(result_dir + 'FastText4_2017-07-28#17:20:21_test_res.pt')
    fasttext1 = torch.load(result_dir + 'FastText1_2017-07-29#10:47:46_test_res.pt')
    fasttext7 = torch.load(result_dir + 'FastText7_2017-07-31#09:52:37_test_res.pt')
    rnn5 = torch.load(result_dir + 'RNN5_2017-07-31#19:18:53_test_res.pt')
    logit = sigmoid(cnn1) * 0.0610 + sigmoid(cnn2) * 0.1218 + sigmoid(rnn1) * 0.0727 + sigmoid(rnn2) * 0.0594 + sigmoid(rcnn1) * 0.0515 + sigmoid(rcnncha) * 0.0613 + sigmoid(fasttext4/4) * 0.0295 + sigmoid(fasttext1) * 0.0398 + sigmoid(fasttext7/7) * 0.0875 + sigmoid(rnn5/5) * 0.3186
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

if __name__ == '__main__':
   # emsemble_eval() 
   emsemble_test() 
