import torch
import datetime
import os

def load_model(model, **kwargs):
	load_dir = '{}/{}'.format(kwargs['model_dir'], kwargs['model_name'])
	if kwargs.get('name', None) is None:		
		checkpoints = '{}/checkpoints'.format(load_dir)
		if os.path.exists(checkpoints):
			name = open(checkpoints).readline().strip()
	else:
		name = kwargs.get('name')
	if name:
		model.load_state_dict(torch.load('{}/{}'.format(load_dir, name), map_location=lambda storage, loc: storage))
		print 'Load model {} successful!'.format('{}/{}'.format(load_dir, name))
	else:
		print 'Cannot find model to load'
	return model

def save_model(model, **kwargs):
	save_dir = '{}/{}'.format(kwargs['model_dir'], kwargs['model_name'])
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	if kwargs.get('name', None) is None:
		cur_time = datetime.datetime.now().strftime('%Y-%m-%d#%H:%M:%S')
		if kwargs.get('score', None) is None:
			name = 'epoch_{}_{}.params'.format(kwargs['epoch'], cur_time)
		elif kwargs.get('folder', None) is None:
			name = 'epoch_{}_{}_{:.4f}.params'.format(kwargs['epoch'], cur_time, kwargs['score'])
		else:
			name = 'folder_{}_epoch_{}_{}_{:.4f}.params'.format(kwargs['folder'], kwargs['epoch'], cur_time, kwargs['score'])
	else:
		name = kwargs.get('name')
	torch.save(model.state_dict(), '{}/{}'.format(save_dir, name))
	print 'Save model {} successful!'.format('{}/{}'.format(save_dir, name))
	checkpoints = '{}/checkpoints'.format(save_dir)
	add_checkpoints(checkpoints, name)
	print 'Added to {}'.format(checkpoints)
		
def add_checkpoints(checkpoints, model_name):
	with open(checkpoints, 'wr+') as f:
		content = f.read()
		f.seek(0, 0)
		f.write(model_name.rstrip('\r\n') + '\n' + content)