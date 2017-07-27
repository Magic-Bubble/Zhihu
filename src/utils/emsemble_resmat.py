import torch
import numpy as np
from writeResult import write_result

def emsemble_eval():
    result_dir = '/home/dyj/'
    label = torch.load(result_dir + 'label.pt')
    cnn1 = torch.load(result_dir + 'TextCNN_2017-07-27#10:15:20_res.pt')
    cnn2 = torch.load(result_dir + 'TextCNN_2017-07-27#10:32:21_res.pt')
    rnn1 = torch.load(result_dir + 'RNN_2017-07-27#10:48:05_res.pt')
    rnn2 = torch.load(result_dir + 'RNN_2017-07-27#10:41:03_res.pt')
    rcnn1 = torch.load(result_dir + 'RCNN_2017-07-27#11:01:07_res.pt')
    rcnncha = torch.load(result_dir + 'RCNNcha_2017-07-27#16:19:23_res.pt')
    logit = (cnn1 + rnn1 + rcnn1 + rcnncha) / 4 + (cnn2 + rnn2) / 2
    print get_score(logit, label)
    
def emsemble_test():
    test_idx = np.load('../../data_preprocess/test/test_idx.npy')
    topic_idx = np.load('../../data_preprocess/topic/topic_idx.npy')
    
    result_dir = '../results/'
    cnn1 = torch.load(result_dir + 'TextCNN1_2017-07-27#12:30:16_test_res.pt')
    cnn2 = torch.load(result_dir + 'TextCNN2_2017-07-27#12:22:42_test_res.pt')
    rnn1 = torch.load(result_dir + 'RNN1_2017-07-27#12:35:51_test_res.pt')
    rnn2 = torch.load(result_dir + 'RNN2_2017-07-27#11:33:24_test_res.pt')
    rcnn1 = torch.load(result_dir + 'RCNN1_2017-07-27#11:30:42_test_res.pt')
    rcnncha = torch.load(result_dir + 'RCNNcha_2017-07-27#16:00:33_test_res.pt')
    logit = (cnn1 + rnn1 + rcnn1 + rcnncha) / 4 + (cnn2 + rnn2) / 2
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

if __name__ == '__main__':
   #emsemble_eval() 
   emsemble_test() 
