import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
import numpy as np
import fire
import datetime

from config import DefaultConfig
from dataset import Dataset
import models
from utils import load_model, save_model, Visualizer, Logger, write_result

def train(**kwargs):
    opt = DefaultConfig()
    opt.update(**kwargs)

    vis = Visualizer(opt['model'])
    logger = Logger()

    logger.info('Load {} data starting...'.format('char' if opt['use_char'] else 'word'))
    if opt['use_double_length']: prefix = '_2'
    else: prefix = ''
    if opt['use_char']:
        opt['embed_num'] = opt['char_embed_num']
        embed_mat = np.load(opt['char_embed'])
        train_title = np.load(opt['train_title_char'+prefix])
        train_desc = np.load(opt['train_desc_char'+prefix])
        train_label = np.load(opt['train_label'])
        val_title = np.load(opt['val_title_char'+prefix])
        val_desc = np.load(opt['val_desc_char'+prefix])
        val_label = np.load(opt['val_label'])
    else:
        opt['embed_num'] = opt['word_embed_num']
        embed_mat = np.load(opt['word_embed'])
        train_title = np.load(opt['train_title_word'+prefix])
        train_desc = np.load(opt['train_desc_word'+prefix])
        train_label = np.load(opt['train_label'])
        val_title = np.load(opt['val_title_word'+prefix])
        val_desc = np.load(opt['val_desc_word'+prefix])
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
        
    import sys
    precision, recall, score = eval(val_loader, model, opt, save=True)
    print precision, recall, score
    sys.exit()
        
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
                corrects = ((logit.data > opt['threshold']) == (label.data/5).byte()).sum()
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
            break
								
def eval(val_loader, model, opt, isBatch=False, return_error=False, save=False):
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
        if save:
            res = torch.Tensor(299997, 1999)
            truth = torch.Tensor(299997, 1999)
        for i, batch in enumerate(val_loader, 0):
            batch_size = batch[0].size(0)
            title, desc, label = batch
            title, desc, label = Variable(title), Variable(desc), Variable(label)
            if opt['cuda']:
                title, desc, label = title.cuda(), desc.cuda(), label.cuda()
            logit = model(title, desc)
            if save:
                res[i*opt['batch_size']:i*opt['batch_size']+batch_size] = logit.data.cpu()
                truth[i*opt['batch_size']:i*opt['batch_size']+batch_size] = label.data.cpu()
            predict_label_list += [list(ii) for ii in logit.topk(5, 1)[1].data]
            marked_label_list += [list(np.where(ii.cpu().numpy()==1)[0]) for ii in label.data]
    model.train()

    if save:
        torch.save(res, '{}/{}_{}_res.pt'.format(opt['result_dir'], opt['model'], datetime.datetime.now().strftime('%Y-%m-%d#%H:%M:%S')))
        torch.save(truth, '{}/{}_{}_label.pt'.format(opt['result_dir'], opt['model'], datetime.datetime.now().strftime('%Y-%m-%d#%H:%M:%S')))
            
    if return_error:
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
    
def train_all(**kwargs):
    opt = DefaultConfig()
    opt.update(**kwargs)

    vis = Visualizer(opt['model'])
    logger = Logger()
    
    logger.info('Load {} data starting...'.format('char' if opt['use_char'] else 'word'))
    if opt['use_double_length']: prefix = '_2'
    else: prefix = ''
    if opt['use_char']:
        opt['embed_num'] = opt['char_embed_num']
    	embed_mat = np.load(opt['char_embed'])
    	train_title = np.load(opt['train_title_char_all'+prefix])
    	train_desc = np.load(opt['train_desc_char_all'+prefix])
    	train_label = np.load(opt['train_label_all'])
    else:
        opt['embed_num'] = opt['word_embed_num']
    	embed_mat = np.load(opt['word_embed'])
    	train_title = np.load(opt['train_title_word_all'+prefix])
    	train_desc = np.load(opt['train_desc_word_all'+prefix])
    	train_label = np.load(opt['train_label_all'])
    logger.info('Load {} data finished!'.format('char' if opt['use_char'] else 'word'))

    logger.info('Using model {}'.format(opt['model']))
    Model = getattr(models, opt['model'])
    model = Model(embed_mat, opt)
    print model

    if opt['use_self_loss']:
        Loss = getattr(models, opt['loss_function'])
    else:
        Loss = getattr(nn, opt['loss_function'])

    if opt['load_loss_weight']:
        error_per_class = torch.load('{}/{}/folder_{}_error_per_class.pt'.format(opt['model_dir'], opt['model'], opt['base_folder']), map_location=lambda storage, loc: storage)
        sample_per_class = torch.load('{}/{}/folder_{}_sample_per_class.pt'.format(opt['model_dir'], opt['model'], opt['base_folder']), map_location=lambda storage, loc: storage)
        loss_weight = error_per_class / sample_per_class * 2
    else:
        error_per_class = torch.zeros(opt['class_num'])
        sample_per_class = torch.zeros(opt['class_num'])
        loss_weight = torch.ones(opt['class_num'])
        
    if opt['device'] != None:
        torch.cuda.set_device(opt['device'])    
    
    if opt['cuda']:
        error_per_class = error_per_class.cuda()
        sample_per_class = sample_per_class.cuda()
        loss_weight = loss_weight.cuda()
    loss_function = Loss(weight=loss_weight)
    
    print loss_weight
    
    if opt['load']:
        if opt.get('load_name', None) is None:
            model = load_model(model, model_dir=opt['model_dir'], model_name=opt['model'])
        else:
            model = load_model(model, model_dir=opt['model_dir'], model_name=opt['model'], \
                              name=opt['load_name'])

    if opt['cuda']:
        model.cuda()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])
    
    logger.info('Start running...')
    
    model.train()
    base_folder, base_epoch = opt['base_folder'], opt['base_epoch']
    for cv_num in range(base_folder+1, opt['folder_num']+1):
        if opt['folder_num'] == 1:    
            train_dataset = Dataset(title=train_title, desc=train_desc, label=train_label, class_num=opt['class_num'])
            train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=opt['batch_size'])
        if opt['folder_num'] > 1:
            train_dataset = Dataset(title=train_title, desc=train_desc, label=train_label, class_num=opt['class_num'], folder_num=opt['folder_num'], cv_num=cv_num, cv=False)
            train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=opt['batch_size'])
            val_dataset = Dataset(title=train_title, desc=train_desc, label=train_label, class_num=opt['class_num'], folder_num=opt['folder_num'], cv_num=cv_num, cv=True)
            val_loader = data.DataLoader(val_dataset, shuffle=True, batch_size=opt['batch_size'])
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
                
                if i % opt['log_interval'] == 0:
                    corrects = ((logit.data > opt['threshold']) == label.data.byte()).sum()
                    accuracy = 100.0 * corrects / (opt['batch_size'] * opt['class_num'])
                    log_info = 'Folder[{:>2}] (epoch[{:>2}] / batch[{:>5}]) - loss: {:.6f}, acc: {:.4f} % ({} / {})'.format( \
                                    cv_num, epoch, i, loss.data[0], accuracy, \
                                    corrects, opt['batch_size'] * opt['class_num'])
                    logger.info(log_info)
                    vis.plot('loss', loss.data[0])
                    precision, recall, score = eval(batch, model, opt, isBatch=True)
                    vis.plot('score', score)
            logger.info('Training epoch {} finished!'.format(epoch))
            if opt['folder_num'] > 1:
                precision, recall, score = eval(val_loader, model, opt)
                log_info = 'Folder[{}], Epoch[{}] - score: {:.6f} (precision: {:.4f}, recall: {:.4f})'.format( \
                                    cv_num, epoch, score, precision, recall)
                vis.log(log_info)
                save_model(model, model_dir=opt['model_dir'], model_name=opt['model'], \
                        epoch=epoch, score=score, folder=cv_num)
            else:
                save_model(model, model_dir=opt['model_dir'], model_name=opt['model'], epoch=epoch)
            if epoch == 2:
                model.opt['static'] = False
            elif epoch == 4:
                optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr']*opt['lr_decay'])
        logger.info('Training folder {} finished!'.format(cv_num))
        error_per_class_tmp, sample_per_class_tmp = eval(val_loader, model, opt, return_error=True)
        error_per_class += error_per_class_tmp
        sample_per_class += sample_per_class_tmp
        torch.save(error_per_class, '{}/{}/folder_{}_error_per_class.pt'.format(opt['model_dir'], opt['model'], cv_num))
        torch.save(sample_per_class, '{}/{}/folder_{}_sample_per_class.pt'.format(opt['model_dir'], opt['model'], cv_num))
        loss_weight = error_per_class / sample_per_class * 2
        base_epoch = 0
        opt['static'] = True
        del model
        model = Model(embed_mat, opt)
        if opt['cuda']:
            model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])
        loss_function = Loss(weight=loss_weight)
		
def test(**kwargs):
    opt = DefaultConfig()
    opt.update(**kwargs)

    logger = Logger()

    logger.info('Load {} data starting...'.format('char' if opt['use_char'] else 'word'))
    if opt['use_double_length']: prefix = '_2'
    else: prefix = ''
    if opt['use_char']:
        opt['embed_num'] = opt['char_embed_num']
        embed_mat = np.load(opt['char_embed'])
        test_title = np.load(opt['test_title_char'+prefix])
        test_desc = np.load(opt['test_desc_char'+prefix])
    else:
        opt['embed_num'] = opt['word_embed_num']
        embed_mat = np.load(opt['word_embed'])
        test_title = np.load(opt['test_title_word'+prefix])
        test_desc = np.load(opt['test_desc_word'+prefix])
    logger.info('Load {} data finished!'.format('char' if opt['use_char'] else 'word'))

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
    for i, batch in enumerate(test_loader, 0):
        title, desc = batch
        title, desc = Variable(title), Variable(desc)
        if opt['cuda']:
            title, desc = title.cuda(), desc.cuda()
        logit = model(title, desc)
        predict_label_list += [list(ii) for ii in logit.topk(5, 1)[1].data]

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