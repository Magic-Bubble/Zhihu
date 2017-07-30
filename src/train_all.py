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
        error_per_class = torch.load('{}/{}/folder_{}_error_per_class_{}.pt'.format(opt['model_dir'], opt['model'], opt['base_folder']), map_location=lambda storage, loc: storage)
        sample_per_class = torch.load('{}/{}/folder_{}_sample_per_class_{}.pt'.format(opt['model_dir'], opt['model'], opt['base_folder']), map_location=lambda storage, loc: storage)
        loss_weight = error_per_class / sample_per_class * 2
    else:
        error_per_class = torch.zeros(opt['class_num'])
        sample_per_class = torch.zeros(opt['class_num'])
        loss_weight = torch.ones(opt['class_num'])

    # loss_weight = torch.ones(opt['class_num'])
    # if opt['boost']:
    #     if opt['base_layer'] != 0:
    #         cal_res = torch.load('{}/{}/layer_{}_cal_res.pt'.format(opt['model_dir'], opt['model'], opt['base_layer']), map_location=lambda storage, loc: storage)
    #         loss_weight = torch.load('{}/{}/layer_{}_loss_weight.pt'.format(opt['model_dir'], opt['model'], opt['base_layer']+1), map_location=lambda storage, loc: storage)
    #     else:
    #         cal_res = torch.zeros(299997, opt['class_num'])
        
    if opt['device'] != None:
        torch.cuda.set_device(opt['device'])    
    
    if opt['cuda']:
        error_per_class = error_per_class.cuda()
        sample_per_class = sample_per_class.cuda()
        loss_weight = loss_weight.cuda()

    loss_function = Loss(weight=loss_weight)
    # loss_function = Loss(weight=loss_weight)
    
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
            elif epoch >= 5:
                # if opt['boost']:
                #     res, truth = eval(val_loader, model, opt, return_res=True)
                #     cal_res += torch.from_numpy(res.numpy() * loss_weight.numpy())
                #     loss_weight = get_loss_weight(cal_res, truth)
                #     torch.save(cal_res, '{}/{}/layer_{}_cal_res.pt'.format(opt['model_dir'], opt['model'], opt['base_layer']+1))
                #     torch.save(loss_weight, '{}/{}/layer_{}_loss_weight.pt'.format(opt['model_dir'], opt['model'], opt['base_layer']+2))
                break
        logger.info('Training folder {} finished!'.format(cv_num))
        error_per_class_tmp, sample_per_class_tmp = eval(val_loader, model, opt, return_err=True)
        error_per_class += error_per_class_tmp
        sample_per_class += sample_per_class_tmp
        torch.save(error_per_class, '{}/{}/folder_{}_error_per_class_{}.pt'.format(opt['model_dir'], opt['model'], cv_num))
        torch.save(sample_per_class, '{}/{}/folder_{}_sample_per_class_{}.pt'.format(opt['model_dir'], opt['model'], cv_num))
        loss_weight = error_per_class / sample_per_class * 2
        base_epoch = 0
        opt['static'] = True
        del model
        model = Model(embed_mat, opt)
        if opt['cuda']:
            model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])
        loss_function = Loss(weight=loss_weight)