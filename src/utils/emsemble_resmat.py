import torch

def emsemble_eval():
    result_dir = '../results/'
    label = torch.load(result_dir + 'label.pt')
    cnn1 = torch.load(result_dir + 'TextCNN_2017-07-26#11:13:38_res.pt')
    cnn2 = torch.load(result_dir + 'TextCNN_2017-07-26#11:21:47_res.pt')
    rnn1 = torch.load(result_dir + 'RNN_2017-07-26#10:36:17_res.pt')
    rnn2 = torch.load(result_dir + 'RNN_2017-07-26#19:48:56_res.pt')
    rcnn1 = torch.load(result_dir + 'RCNN_2017-07-26#10:51:11_res.pt')
    logit = (cnn1 + rnn1 + rcnn1) / 3 + (cnn2 + rnn2) / 2
    print get_score(logit, label)
    
def emsemble_test():
    test_idx = np.load('../../data_preprocess/test/test_idx.npy')
    topic_idx = np.load('../../data_preprocess/topic/topic_idx.npy')
    
    cnn1 = torch.load()
    cnn2 = torch.load()
    rnn1 = torch.load()
    rnn2 = torch.load()
    rcnn1 = torch.load()
    logit = (cnn1 + rnn1 + rcnn1) / 3 + (cnn2 + rnn2) / 2
    predict_label_list = [list(ii) for ii in logit.topk(5, 1)[1]]
    lines = []
    for qid, top5 in zip(test_idx, predict_label_list):
        topic_ids = [topic_idx[i] for i in top5]
        lines.append('{},{}'.format(qid, ','.join(topic_ids)))

    write_result(lines, model_name='Emsemble', result_dir='../results')
    
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