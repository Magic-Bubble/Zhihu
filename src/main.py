import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
import numpy as np
import fire

from config import DefaultConfig
from dataset import Dataset
import models
from utils import load_model, save_model, Visualizer, Logger

def train(**kwargs):
    opt = DefaultConfig()
    opt.update(**kwargs)

    vis = Visualizer(opt['model'])
    logger = Logger()

    logger.info('Load {} data starting...'.format('char' if opt['use_char'] else 'word'))
    if opt['use_char']:
        opt['embed_num'] = opt['char_embed_num']
        embed_mat = np.load(opt['char_embed'])
        train_title = np.load(opt['train_title_char'])
        train_desc = np.load(opt['train_desc_char'])
        train_label = np.load(opt['train_label'])
        val_title = np.load(opt['val_title_char'])
        val_desc = np.load(opt['val_desc_char'])
        val_label = np.load(opt['val_label'])
    else:
        opt['embed_num'] = opt['word_embed_num']
        embed_mat = np.load(opt['word_embed'])
        train_title = np.load(opt['train_title_word'])
        train_desc = np.load(opt['train_desc_word'])
        train_label = np.load(opt['train_label'])
        val_title = np.load(opt['val_title_word'])
        val_desc = np.load(opt['val_desc_word'])
        val_label = np.load(opt['val_label'])
    logger.info('Load {} data finished!'.format('char' if opt['use_char'] else 'word'))
    				
    train_dataset = Dataset(title=train_title, desc=train_desc, label=train_label, class_num=opt['class_num'])
    train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=opt['batch_size'])
    val_dataset = Dataset(title=val_title, desc=val_desc, label=val_label, class_num=opt['class_num'])
    val_loader = data.DataLoader(val_dataset, shuffle=True, batch_size=opt['batch_size'])
    
    logger.info('Using model {}'.format(opt['model']))
    Model = getattr(models, opt['model'])
    model = Model(embed_mat, opt)
    print model

    if opt['use_self_loss']:
        Loss = getattr(models, opt['loss_function'])
    else:
        Loss = getattr(nn, opt['loss_function'])
    loss_function = Loss()
    
    if opt['load']:
        if opt.get('load_name', None) is None:
            model = load_model(model, model_dir=opt['model_dir'], model_name=opt['model'])
        else:
            model = load_model(model, model_dir=opt['model_dir'], model_name=opt['model'], \
                              name=opt['load_name'])
    
    if opt['device'] != None:
        torch.cuda.set_device(opt['device'])

    if opt['cuda']:
        model.cuda()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])
    
    logger.info('Start running...')

    steps = 0
    model.train()
    base_epoch = opt['base_epoch']
    for epoch in range(1, opt['epochs']+1):
        for i, batch in enumerate(train_loader, 0):
            text, label = batch
            text, label = Variable(text), Variable(label).float()
            # title, desc, label = batch
            # title, desc, label = Variable(title), Variable(desc), Variable(label).float()
            if opt['cuda']:
                text, label = text.cuda(), label.cuda()
                # title, desc, label = title.cuda(), desc.cuda(), label.cuda()
                
            optimizer.zero_grad()
            logit = model(text)
            # logit = model(title, desc)
            
            loss = loss_function(logit, label)
            loss.backward()
            optimizer.step()
            
            steps +=1 
            if steps % opt['log_interval'] == 0:
                corrects = ((logit.data > opt['threshold']) == label.data.byte()).sum()
                accuracy = 100.0 * corrects / (opt['batch_size'] * opt['class_num'])
                log_info = 'Steps[{:>8}] (epoch[{:>2}] / batch[{:>5}]) - loss: {:.6f}, acc: {:.4f} % ({} / {})'.format( \
                                steps, epoch + base_epoch, (i+1), loss.data[0], accuracy, \
                                corrects, opt['batch_size'] * opt['class_num'])
                logger.info(log_info)
                vis.plot('loss', loss.data[0])
        logger.info('Training epoch {} finished!'.format(epoch + base_epoch))
        precision, recall, score = eval(val_loader, model, opt)
        log_info = 'Epoch[{}] - score: {:.6f} (precision: {:.4f}, recall: {:.4f})'.format( \
                            epoch + base_epoch, score, precision, recall)
        vis.log(log_info)
        save_model(model, model_dir=opt['model_dir'], model_name=opt['model'], \
                    epoch=epoch+base_epoch, score=score)
								
def eval(val_loader, model, opt):
    model.eval()
    predict_label_list, marked_label_list = [], []
    for i, batch in enumerate(val_loader, 0):
        text, label = batch
        text, label = Variable(text), Variable(label)
        # title, desc, label = batch
        # title, desc, label = Variable(title), Variable(desc), Variable(label)
        if opt['cuda']:
            text, label = text.cuda(), label.cuda()
            # title, desc, label = title.cuda(), desc.cuda(), label.cuda()
        logit = model(text)
        # logit = model(title, desc)
        predict_label_list += [list(ii) for ii in logit.topk(5, 1)[1].data]
        marked_label_list += [list(np.where(ii.cpu().numpy()==1)[0]) for ii in label.data]

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
								
def train_all(**args):
    opt = DefaultConfig()
    opt.update(**args)

    vis = Visualizer(opt['model'])
    logger = Logger()
    
    logger.info('Load {} data starting...'.format('char' if opt['use_char'] else 'word'))
    if opt['use_char']:
        opt['embed_num'] = opt['char_embed_num']
    	embed_mat = np.load(opt['char_embed'])
    	train_title = np.load(opt['train_title_char_all'])
    	train_desc = np.load(opt['train_desc_char_all'])
    	train_label = np.load(opt['train_label_all'])
    else:
        opt['embed_num'] = opt['word_embed_num']
    	embed_mat = np.load(opt['word_embed'])
    	train_title = np.load(opt['train_title_word_all'])
    	train_desc = np.load(opt['train_desc_word_all'])
    	train_label = np.load(opt['train_label_all'])
    logger.info('Load {} data finished!'.format('char' if opt['use_char'] else 'word'))
    		
    train_dataset = Dataset(title=train_title, desc=train_desc, label=train_label, class_num=opt['class_num'])
    train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=opt['batch_size'])

    logger.info('Using model {}'.format(opt['model']))
    Model = getattr(models, opt['model'])
    model = Model(embed_mat, opt)

    if opt['use_self_loss']:
        Loss = getattr(models, opt['self_loss'])
    else:
        Loss = getattr(nn, opt['loss_function'])
    loss_function = Loss()
    
    if opt['load']:
        model = load_model(model, model_dir=opt['model_dir'], model_name=opt['model'])
    
    if opt['cuda']:
        model.cuda()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])

    logger.info('Start running...')

    steps = 0
    model.train()
    for epoch in range(1, opt['epochs']+1):
        for i, batch in enumerate(train_loader, 0):
            text, label = batch
            text, label = Variable(text), Variable(label)
            if opt['cuda']:
                text, label = text.cuda(), label.cuda()
                
            optimizer.zero_grad()
            logit = model(text)
            
            loss = loss_function(logit, label)
            loss.backward()
            optimizer.step()
            
            steps +=1 
            if steps % opt['log_interval'] == 0:
                corrects = ((logit.data > opt['threshold']) == label.data.byte()).sum()
                accuracy = 100.0 * corrects / (opt['batch_size'] * opt['class_num'])
                log_info = 'Steps[{:>8}] (epoch[{:>2}] / batch[{:>5}]) - loss: {:.6f}, acc: {:.4f} % ({} / {})'.format( \
                                steps, epoch, (i+1), loss.data[0], accuracy, \
                                corrects, opt['batch_size'] * opt['class_num'])
                logger.info(log_info)
                vis.plot('loss', loss.data[0])
        save_model(model, model_dir=opt['model_dir'], model_name=opt['model'], \
                    epoch=epoch)
		
def test(**args):
    opt = DefaultConfig()
    opt.update(**args)

    logger = Logger()

    logger.info('Load {} data starting...'.format('char' if opt['use_char'] else 'word'))
    if opt['use_char']:
        opt['embed_num'] = opt['char_embed_num']
        embed_mat = np.load(opt['char_embed'])
        test_title = np.load(opt['test_title_char'])
        test_desc = np.load(opt['test_desc_char'])
    else:
        opt['embed_num'] = opt['word_embed_num']
        embed_mat = np.load(opt['word_embed'])
        test_title = np.load(opt['test_title_word'])
        test_desc = np.load(opt['test_desc_word'])
    logger.info('Load {} data finished!'.format('char' if opt['use_char'] else 'word'))

    test_idx = np.load(opt['test_idx'])
    topic_idx = np.load(opt['topic_idx'])
            
    test_dataset = Dataset(test=True, title=test_title, desc=test_desc)
    test_loader = data.DataLoader(test_dataset, shuffle=False, batch_size=opt['batch_size'])

    logger.info('Using model {}'.format(opt['model']))
    Model = getattr(models, opt['model'])
    model = Model(embed_mat, opt)
    
    if opt['load']:
        model = load_model(model, model_dir=opt['model_dir'], model_name=opt['model'])
    
    if opt['cuda']:
        model.cuda()

    logger.info('Start testing...')

    model.eval()
    predict_label_list = []
    for i, batch in enumerate(test_loader, 0):
        text = batch
        text = Variable(text)
        if opt['cuda']:
            text = text.cuda()
        logit = model(text)
        predict_label_list += [list(ii) for ii in logit.topk(5, 1)[1].data]

    lines = []
    for qid, top5 in zip(test_idx, predict_label_list):
        topic_ids = [topic_idx[i] for i in top5]
        lines.append('{},{}'.format(qid, ','.join(topic_ids)))

    with open(opt['result'], 'w') as output:
        output.write('\n'.join(lines))
                     
if __name__ == '__main__':
    fire.Fire()