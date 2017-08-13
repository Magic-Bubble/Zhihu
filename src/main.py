import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
import numpy as np
import fire
import datetime

from config import DefaultConfig
from dataset import Dataset, Stack_Dataset
import models
from utils import load_model, save_model, Visualizer, Logger, write_result, get_loss_weight, get_score

def train(**kwargs):
    opt = DefaultConfig()
    opt.update(**kwargs)

    vis = Visualizer(opt['model'])
    logger = Logger()

    prefix = ''
    if opt['use_double_length']: prefix += '_2'
    if opt['data_shuffle']: prefix += '_shuffle'
    print prefix
    if opt['use_char']:
        logger.info('Load char data starting...')
        opt['embed_num'] = opt['char_embed_num']
        embed_mat = np.load(opt['char_embed'])
        train_title = np.load(opt['train_title_char'+prefix])
        train_desc = np.load(opt['train_desc_char'+prefix])
        train_label = np.load(opt['train_label'])
        val_title = np.load(opt['val_title_char'+prefix])
        val_desc = np.load(opt['val_desc_char'+prefix])
        val_label = np.load(opt['val_label'])
        logger.info('Load char data finished!')
    elif opt['use_word']:
        logger.info('Load word data starting...')
        opt['embed_num'] = opt['word_embed_num']
        embed_mat = np.load(opt['word_embed'])
        train_title = np.load(opt['train_title_word'+prefix])
        train_desc = np.load(opt['train_desc_word'+prefix])
        train_label = np.load(opt['train_label'])
        val_title = np.load(opt['val_title_word'+prefix])
        val_desc = np.load(opt['val_desc_word'+prefix])
        val_label = np.load(opt['val_label'])
        logger.info('Load word data finished!')
    elif opt['use_char_word']:
        logger.info('Load char-word data starting...')
        embed_mat_char = np.load(opt['char_embed'])
        embed_mat_word = np.load(opt['word_embed'])
        embed_mat = np.vstack((embed_mat_char, embed_mat_word))
        train_title = np.load(opt['train_title_char'+prefix])
        train_desc = np.load(opt['train_desc_word'+prefix])
        train_label = np.load(opt['train_label'])
        val_title = np.load(opt['val_title_char'+prefix])
        val_desc = np.load(opt['val_desc_word'+prefix])
        val_label = np.load(opt['val_label'])
        logger.info('Load char-word data finished!')
    elif opt['use_word_char']:
        logger.info('Load word-char data starting...')
        embed_mat_char = np.load(opt['char_embed'])
        embed_mat_word = np.load(opt['word_embed'])
        embed_mat = np.vstack((embed_mat_char, embed_mat_word))
        train_title = np.load(opt['train_title_word'+prefix])
        train_desc = np.load(opt['train_desc_char'+prefix])
        train_label = np.load(opt['train_label'])
        val_title = np.load(opt['val_title_word'+prefix])
        val_desc = np.load(opt['val_desc_char'+prefix])
        val_label = np.load(opt['val_label'])
        logger.info('Load word-char data finished!')
    
    train_dataset = Dataset(title=train_title, desc=train_desc, label=train_label, class_num=opt['class_num'])
    train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=opt['batch_size'])
    val_dataset = Dataset(title=val_title, desc=val_desc, label=val_label, class_num=opt['class_num'])
    val_loader = data.DataLoader(val_dataset, shuffle=False, batch_size=opt['batch_size'])
    
    logger.info('Using model {}'.format(opt['model']))
    Model = getattr(models, opt['model'])
    model = Model(embed_mat, opt)
    print model

    loss_weight = torch.ones(opt['class_num'])
    if opt['boost']:
        if opt['base_layer'] != 0:
            cal_res = torch.load('{}/{}/layer_{}_cal_res_top1.pt'.format(opt['model_dir'], opt['model'], opt['base_layer']), map_location=lambda storage, loc: storage)
            logger.info('Load cal_res successful!')
            loss_weight = torch.load('{}/{}/layer_{}_loss_weight_top1.pt'.format(opt['model_dir'], opt['model'], opt['base_layer']+1), map_location=lambda storage, loc: storage)
        else:
            cal_res = torch.zeros(opt['val_num'], opt['class_num'])
        print 'cur_layer:', opt['base_layer'] + 1, \
              'loss_weight:', loss_weight.mean(), loss_weight.max(), loss_weight.min(), loss_weight.std()

    if opt['use_self_loss']:
        Loss = getattr(models, opt['loss_function'])
    else:
        Loss = getattr(nn, opt['loss_function'])
    
    if opt['load']:
        if opt.get('load_name', None) is None:
            model = load_model(model, model_dir=opt['model_dir'], model_name=opt['model'])
        else:
            model = load_model(model, model_dir=opt['model_dir'], model_name=opt['model'], \
                              name=opt['load_name'])

    if opt['cuda'] and opt['device'] != None:
        torch.cuda.set_device(opt['device'])

    if opt['cuda']:
        model.cuda()
        loss_weight = loss_weight.cuda()
        
    # import sys
    # precision, recall, score = eval(val_loader, model, opt, save_res=True)
    # print precision, recall, score
    # sys.exit()
        
    loss_function = Loss(weight=loss_weight+1-loss_weight.mean())
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])
    
    logger.info('Start running...')

    steps = 0
    model.train()
    base_epoch = opt['base_epoch']
    for epoch in range(1, opt['epochs']+1):
        for i, batch in enumerate(train_loader, 0):
            title, desc, label = batch
            title, desc, label = Variable(title), Variable(desc), Variable(label).float()
            if opt['cuda']:
                title, desc, label = title.cuda(), desc.cuda(), label.cuda()
                
            optimizer.zero_grad()
            logit = model(title, desc)
            
            loss = loss_function(logit, label)
            loss.backward()
            optimizer.step()
            
            steps +=1 
            if steps % opt['log_interval'] == 0:
                corrects = ((logit.data > opt['threshold']) == (label.data).byte()).sum()
                accuracy = 100.0 * corrects / (opt['batch_size'] * opt['class_num'])
                log_info = 'Steps[{:>8}] (epoch[{:>2}] / batch[{:>5}]) - loss: {:.6f}, acc: {:.4f} % ({} / {})'.format( \
                                steps, epoch + base_epoch, (i+1), loss.data[0], accuracy, \
                                corrects, opt['batch_size'] * opt['class_num'])
                logger.info(log_info)
                vis.plot('loss', loss.data[0])
                precision, recall, score = eval(batch, model, opt, isBatch=True)
                vis.plot('score', score)
        logger.info('Training epoch {} finished!'.format(epoch + base_epoch))
        precision, recall, score = eval(val_loader, model, opt)
        log_info = 'Epoch[{}] - score: {:.6f} (precision: {:.4f}, recall: {:.4f})'.format( \
                            epoch + base_epoch, score, precision, recall)
        vis.log(log_info)
        save_model(model, model_dir=opt['model_dir'], model_name=opt['model'], \
                    epoch=epoch+base_epoch, score=score)
        if epoch + base_epoch == 2:
            model.opt['static'] = False
        elif epoch + base_epoch == 4:
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt['lr']*opt['lr_decay']
        elif epoch + base_epoch >= 5:
            if opt['boost']:
                res, truth = eval(val_loader, model, opt, return_res=True)
                ori_score = get_score(cal_res, truth)
                cal_res += res
                cur_score = get_score(cal_res, truth)
                logger.info('Layer {}: {}, Layer {}: {}'.format(opt['base_layer'], ori_score, opt['base_layer']+1, cur_score))
                loss_weight = get_loss_weight(cal_res, truth)
                torch.save(cal_res, '{}/{}/layer_{}_cal_res_top1.pt'.format(opt['model_dir'], opt['model'], opt['base_layer']+1))
                logger.info('Save cal_res successful!')
                torch.save(loss_weight, '{}/{}/layer_{}_loss_weight_top1.pt'.format(opt['model_dir'], opt['model'], opt['base_layer']+2))
            break
								
def eval(val_loader, model, opt, isBatch=False, return_err=False, save_res=False, return_res=False):
    model.eval()
    predict_label_list, marked_label_list = [], []
    if isBatch:
        title, desc, label = val_loader
        title, desc, label = Variable(title), Variable(desc), Variable(label)
        if opt['cuda']:
            title, desc, label = title.cuda(), desc.cuda(), label.cuda()
        logit = model(title, desc)

        predict_label_list += [list(ii) for ii in logit.topk(5, 1)[1].data]
        marked_label_list += [list(np.where(ii.cpu().numpy()==1)[0]) for ii in label.data]
    else:
        if save_res or return_res:
            res = torch.Tensor(opt['val_num'], opt['class_num'])
            truth = torch.Tensor(opt['val_num'], opt['class_num'])
        for i, batch in enumerate(val_loader, 0):
            batch_size = batch[0].size(0)
            title, desc, label = batch
            title, desc, label = Variable(title), Variable(desc), Variable(label)
            if opt['cuda']:
                title, desc, label = title.cuda(), desc.cuda(), label.cuda()
            logit = model(title, desc)
            if save_res or return_res:
                res[i*opt['batch_size']:i*opt['batch_size']+batch_size] = logit.data.cpu()
                truth[i*opt['batch_size']:i*opt['batch_size']+batch_size] = label.data.cpu()
            predict_label_list += [list(ii) for ii in logit.topk(5, 1)[1].data]
            marked_label_list += [list(np.where(ii.cpu().numpy()==1)[0]) for ii in label.data]
    model.train()

    if save_res:
        torch.save(res, '{}/{}_{}_res.pt'.format(opt['result_dir'], opt['model'], datetime.datetime.now().strftime('%Y-%m-%d#%H:%M:%S')))
        #torch.save(truth, '{}/{}_{}_label.pt'.format(opt['result_dir'], opt['model'], datetime.datetime.now().strftime('%Y-%m-%d#%H:%M:%S')))

    if return_res:
        return res, truth
            
    if return_err:
        sample_per_class = torch.zeros(opt['class_num'])
        error_per_class = torch.zeros(opt['class_num'])
        if opt['cuda']:
            sample_per_class = sample_per_class.cuda()
            error_per_class = error_per_class.cuda()
        for predict_labels, marked_labels in zip(predict_label_list, marked_label_list):
            for true_label in marked_labels:
                sample_per_class[true_label] += 1
                if true_label not in predict_labels:
                    error_per_class[true_label] += 1
        return error_per_class, sample_per_class

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

def finetune_all(**kwargs):
    opt = DefaultConfig()
    opt.update(**kwargs)

    vis = Visualizer(opt['model'])
    logger = Logger()

    prefix = ''
    if opt['use_double_length']: prefix += '_2'
    if opt['data_shuffle']: prefix += '_shuffle'
    print prefix
    if opt['use_char']:
        logger.info('Load char data starting...')
        opt['embed_num'] = opt['char_embed_num']
        embed_mat = np.load(opt['char_embed'])
        train_title = np.load(opt['train_title_char_all'+prefix])
        train_desc = np.load(opt['train_desc_char_all'+prefix])
        train_label = np.load(opt['train_label_all'])
        logger.info('Load char data finished!')
    elif opt['use_word']:
        logger.info('Load word data starting...')
        opt['embed_num'] = opt['word_embed_num']
        embed_mat = np.load(opt['word_embed'])
        train_title = np.load(opt['train_title_word_all'+prefix])
        train_desc = np.load(opt['train_desc_word_all'+prefix])
        train_label = np.load(opt['train_label_all'])
        logger.info('Load word data finished!')
    elif opt['use_char_word']:
        logger.info('Load char-word data starting...')
        embed_mat_char = np.load(opt['char_embed'])
        embed_mat_word = np.load(opt['word_embed'])
        embed_mat = np.vstack((embed_mat_char, embed_mat_word))
        train_title = np.load(opt['train_title_char_all'+prefix])
        train_desc = np.load(opt['train_desc_word_all'+prefix])
        train_label = np.load(opt['train_label_all'])
        logger.info('Load char-word data finished!')
    elif opt['use_word_char']:
        logger.info('Load word-char data starting...')
        embed_mat_char = np.load(opt['char_embed'])
        embed_mat_word = np.load(opt['word_embed'])
        embed_mat = np.vstack((embed_mat_char, embed_mat_word))
        train_title = np.load(opt['train_title_word_all'+prefix])
        train_desc = np.load(opt['train_desc_char_all'+prefix])
        train_label = np.load(opt['train_label_all'])
        logger.info('Load word-char data finished!')
    
    train_dataset = Dataset(title=train_title, desc=train_desc, label=train_label, class_num=opt['class_num'])
    train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=opt['batch_size'])
    
    logger.info('Using model {}'.format(opt['model']))
    Model = getattr(models, opt['model'])
    model = Model(embed_mat, opt)
    print model

    loss_weight = torch.ones(opt['class_num'])
    if opt['boost']:
        if opt['base_layer'] != 0:
            if opt['use_char']:
                loss_weight = torch.load('{}/{}/layer_{}_loss_weight_char.pt'.format(opt['model_dir'], opt['model'], opt['base_layer']+1), map_location=lambda storage, loc: storage)
            elif opt['use_word']:
                loss_weight = torch.load('{}/{}/layer_{}_loss_weight_3.pt'.format(opt['model_dir'], opt['model'], opt['base_layer']+1), map_location=lambda storage, loc: storage)
        print 'cur_layer:', opt['base_layer'] + 1, \
              'loss_weight:', loss_weight.mean(), loss_weight.max(), loss_weight.min(), loss_weight.std()

    if opt['use_self_loss']:
        Loss = getattr(models, opt['loss_function'])
    else:
        Loss = getattr(nn, opt['loss_function'])
    
    if opt['load']:
        if opt.get('load_name', None) is None:
            model = load_model(model, model_dir=opt['model_dir'], model_name=opt['model'])
        else:
            model = load_model(model, model_dir=opt['model_dir'], model_name=opt['model'], \
                              name=opt['load_name'])

    if opt['cuda'] and opt['device'] != None:
        torch.cuda.set_device(opt['device'])

    if opt['cuda']:
        model.cuda()
        loss_weight = loss_weight.cuda()
        
    loss_function = Loss(weight=loss_weight+1-loss_weight.mean())
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])
    
    logger.info('Start running...')

    steps = 0
    model.train()
    base_epoch = opt['base_epoch']
    for epoch in range(base_epoch+1, opt['epochs']+1):
        for i, batch in enumerate(train_loader, 1):
            title, desc, label = batch
            title, desc, label = Variable(title), Variable(desc), Variable(label).float()
            if opt['cuda']:
                title, desc, label = title.cuda(), desc.cuda(), label.cuda()
                
            optimizer.zero_grad()
            logit = model(title, desc)
            
            loss = loss_function(logit, label)
            loss.backward()
            optimizer.step()
            
            steps +=1 
            if steps % opt['log_interval'] == 0:
                corrects = ((logit.data > opt['threshold']) == (label.data).byte()).sum()
                accuracy = 100.0 * corrects / (opt['batch_size'] * opt['class_num'])
                log_info = 'Steps[{:>8}] (epoch[{:>2}] / batch[{:>5}]) - loss: {:.6f}, acc: {:.4f} % ({} / {})'.format( \
                                steps, epoch, i, loss.data[0], accuracy, \
                                corrects, opt['batch_size'] * opt['class_num'])
                logger.info(log_info)
                # vis.plot('loss', loss.data[0])
                # precision, recall, score = eval(batch, model, opt, isBatch=True)
                # vis.plot('score', score)
        logger.info('Training epoch {} finished!'.format(epoch))
        save_model(model, model_dir=opt['model_dir'], model_name=opt['model'], \
                    epoch=epoch)
        if epoch == 6:
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt['lr']*opt['lr_decay']
		
def test(**kwargs):
    opt = DefaultConfig()
    opt.update(**kwargs)

    logger = Logger()

    prefix = ''
    if opt['use_double_length']: prefix += '_2'
    if opt['data_shuffle']: prefix += '_shuffle'
    print prefix
    if opt['use_char']:
        logger.info('Load char data starting...')
        opt['embed_num'] = opt['char_embed_num']
        embed_mat = np.load(opt['char_embed'])
        test_title = np.load(opt['test_title_char'+prefix])
        test_desc = np.load(opt['test_desc_char'+prefix])
        logger.info('Load char data finished!')
    elif opt['use_word']:
        logger.info('Load word data starting...')
        opt['embed_num'] = opt['word_embed_num']
        embed_mat = np.load(opt['word_embed'])
        test_title = np.load(opt['test_title_word'+prefix])
        test_desc = np.load(opt['test_desc_word'+prefix])
        logger.info('Load word data finished!')
    elif opt['use_char_word']:
        logger.info('Load char-word data starting...')
        embed_mat_char = np.load(opt['char_embed'])
        embed_mat_word = np.load(opt['word_embed'])
        embed_mat = np.vstack((embed_mat_char, embed_mat_word))
        test_title = np.load(opt['test_title_char'+prefix])
        test_desc = np.load(opt['test_desc_word'+prefix])
        logger.info('Load char-word data finished!')
    elif opt['use_word_char']:
        logger.info('Load word-char data starting...')
        embed_mat_char = np.load(opt['char_embed'])
        embed_mat_word = np.load(opt['word_embed'])
        embed_mat = np.vstack((embed_mat_char, embed_mat_word))
        test_title = np.load(opt['test_title_word'+prefix])
        test_desc = np.load(opt['test_desc_char'+prefix])
        logger.info('Load word-char data finished!')

    test_idx = np.load(opt['test_idx'])
    topic_idx = np.load(opt['topic_idx'])
            
    test_dataset = Dataset(test=True, title=test_title, desc=test_desc)
    test_loader = data.DataLoader(test_dataset, shuffle=False, batch_size=opt['batch_size'])

    logger.info('Using model {}'.format(opt['model']))
    Model = getattr(models, opt['model'])
    model = Model(embed_mat, opt)
    
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

    logger.info('Start testing...')

    model.eval()
    predict_label_list = []
    res = torch.Tensor(opt['test_num'], opt['class_num'])
    for i, batch in enumerate(test_loader, 0):
        batch_size = batch[0].size(0)
        title, desc = batch
        title, desc = Variable(title), Variable(desc)
        if opt['cuda']:
            title, desc = title.cuda(), desc.cuda()
        logit = model(title, desc)
        if opt.get('save_resmat', False):
            res[i*opt['batch_size']:i*opt['batch_size']+batch_size] = logit.data.cpu()
        predict_label_list += [list(ii) for ii in logit.topk(5, 1)[1].data]

    if opt.get('save_resmat', False):
        torch.save(res, '{}/{}_{}_test_res.pt'.format(opt['result_dir'], opt['model'], datetime.datetime.now().strftime('%Y-%m-%d#%H:%M:%S')))
	return

    lines = []
    for qid, top5 in zip(test_idx, predict_label_list):
        topic_ids = [topic_idx[i] for i in top5]
        lines.append('{},{}'.format(qid, ','.join(topic_ids)))

    if opt.get('load_name', None) is None:
        write_result(lines, model_dir=opt['model_dir'], model_name=opt['model'], result_dir=opt['result_dir'])
    else:
        write_result(lines, model_dir=opt['model_dir'], model_name=opt['model'], \
                          name=opt['load_name'], result_dir=opt['result_dir'])
        
def train_stack(**kwargs):
    opt = DefaultConfig()
    opt.update(**kwargs)

    vis = Visualizer(opt['model'])
    logger = Logger()
    				
    result_dir = '/home/dyj/'
    resmat = [#(result_dir+'TextCNN1_2017-07-27#10:15:20_res.pt', 1),\
              #(result_dir+'RNN1_2017-07-27#10:48:05_res.pt', 1),\
              #(result_dir+'RCNN1_2017-07-27#11:01:07_res.pt', 1),\
              #(result_dir+'RCNNcha_2017-07-27#16:19:23_res.pt', 1),\
              #('snapshots/FastText/layer_1_cal_res_char.pt', 1),\
              #(result_dir+'FastText4_2017-07-28#15:14:47_res.pt', 4),\
              #('snapshots/TextCNN/layer_17_cal_res_3.pt', 17),\
              (result_dir + 'RNN10_cal_res.pt', 10),\
              ('snapshots/TextCNN/layer_10_cal_res_char.pt', 10),\
              ('snapshots/TextCNN/layer_10_cal_res_top1.pt', 10),\
              ('snapshots/TextCNN/layer_8_cal_res_top1_char.pt', 8),\
              (result_dir + 'TextCNN4_cal_res.pt', 4),\
              (result_dir + 'FastText10_res.pt', 10),\
              ('snapshots/TextCNN/layer_4_cal_res_shuffle.pt', 4)
              ]
    label = result_dir+'label.pt'
    opt['stack_num'] = len(resmat)
    
    train_dataset = Stack_Dataset(resmat=resmat, label=label)
    train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=opt['batch_size'])
    
    logger.info('Using model {}'.format(opt['model']))
    Model = getattr(models, opt['model'])
    model = Model(opt)
    print model

    if opt['use_self_loss']:
        Loss = getattr(models, opt['loss_function'])
    else:
        Loss = getattr(nn, opt['loss_function'])
    
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
        
    loss_function = Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])
    
    logger.info('Start running...')

    steps = 0
    model.train()
    for epoch in range(opt['base_epoch']+1, opt['epochs']+1):
        for i, batch in enumerate(train_loader, 1):
            resmat, label = batch[0:-1], batch[-1]
            resmat, label = [Variable(ii) for ii in resmat], Variable(label)
            if opt['cuda']:
                resmat, label = [ii.cuda() for ii in resmat], label.cuda()
                
            optimizer.zero_grad()
            logit = model(resmat)
            
            loss = loss_function(logit, label)
            loss.backward()
            optimizer.step()
            
            steps +=1 
            if steps % opt['log_interval'] == 0:
                corrects = ((logit.data > opt['threshold']) == (label.data).byte()).sum()
                accuracy = 100.0 * corrects / (opt['batch_size'] * opt['class_num'])
                log_info = 'Steps[{:>8}] (epoch[{:>2}] / batch[{:>5}]) - loss: {:.6f}, acc: {:.4f} % ({} / {})'.format( \
                                steps, epoch, i, loss.data[0], accuracy, \
                                corrects, opt['batch_size'] * opt['class_num'])
                logger.info(log_info)
                vis.plot('loss', loss.data[0])
                precision, recall, score = get_score(logit.data.cpu(), label.data.cpu())
                logger.info('Precision {}, Recall {}, Score {}'.format(precision, recall, score))
                vis.plot('score', score)
        logger.info('Training epoch {} finished!'.format(epoch))
        #save_model(model, model_dir=opt['model_dir'], model_name=opt['model'], epoch=epoch)
        if epoch == 3:
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt['lr']*opt['lr_decay']
    save_model(model, model_dir=opt['model_dir'], model_name=opt['model'], epoch=epoch)
        
def test_stack(**kwargs):
    opt = DefaultConfig()
    opt.update(**kwargs)

    logger = Logger()
    				
    result_dir = '/home/dyj/'
    resmat = [result_dir+'TextCNN1_2017-07-27#12:30:16_test_res.pt',\
              result_dir+'TextCNN2_2017-07-27#12:22:42_test_res.pt', \
              result_dir+'RNN1_2017-07-27#12:35:51_test_res.pt',\
              result_dir+'RNN2_2017-07-27#11:33:24_test_res.pt',\
              result_dir+'RCNN1_2017-07-27#11:30:42_test_res.pt',\
              result_dir+'RCNNcha_2017-07-27#16:00:33_test_res.pt',\
              result_dir+'FastText4_2017-07-28#17:20:21_test_res.pt',\
              result_dir+'FastText1_2017-07-29#10:47:46_test_res.pt']
    opt['stack_num'] = len(resmat)
    
    test_dataset = Stack_Dataset(resmat=resmat, test=True)
    test_loader = data.DataLoader(test_dataset, shuffle=False, batch_size=opt['batch_size'])
    
    test_idx = np.load(opt['test_idx'])
    topic_idx = np.load(opt['topic_idx'])
    
    logger.info('Using model {}'.format(opt['model']))
    Model = getattr(models, opt['model'])
    model = Model(opt)
    print model
    
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

    logger.info('Start testing...')

    model.eval()
    predict_label_list = []
    res = torch.Tensor(opt['test_num'], opt['class_num'])
    for i, batch in enumerate(test_loader, 0):
        batch_size = batch[0].size(0)
        resmat = batch
        resmat = [Variable(ii) for ii in resmat]
        if opt['cuda']:
            resmat = [ii.cuda() for ii in resmat]
        logit = model(resmat)
        if opt.get('save_resmat', False):
            res[i*opt['batch_size']:i*opt['batch_size']+batch_size] = logit.data.cpu()
        predict_label_list += [list(ii) for ii in logit.topk(5, 1)[1].data]

    if opt.get('save_resmat', False):
        torch.save(res, '{}/{}_{}_test_res.pt'.format(opt['result_dir'], opt['model'], datetime.datetime.now().strftime('%Y-%m-%d#%H:%M:%S')))
	return

    lines = []
    for qid, top5 in zip(test_idx, predict_label_list):
        topic_ids = [topic_idx[i] for i in top5]
        lines.append('{},{}'.format(qid, ','.join(topic_ids)))

    if opt.get('load_name', None) is None:
        write_result(lines, model_dir=opt['model_dir'], model_name=opt['model'], result_dir=opt['result_dir'])
    else:
        write_result(lines, model_dir=opt['model_dir'], model_name=opt['model'], \
                          name=opt['load_name'], result_dir=opt['result_dir'])
                     
if __name__ == '__main__':
    fire.Fire()
