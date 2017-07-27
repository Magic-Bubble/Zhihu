import os
import sys

def write_result(lines, **kwargs):
	load_dir = '{}/{}'.format(kwargs.get('model_dir', 'results'), kwargs['model_name'])
	name = kwargs.get('name', None)
	if name is None:		
		checkpoints = '{}/checkpoints'.format(load_dir)
		if os.path.exists(checkpoints):
			name = open(checkpoints).readline().strip()
	if name: result_file = '{}/{}_{}.csv'.format(kwargs['result_dir'], kwargs['model_name'], name)
	else: result_file = '{}/{}_result.csv'.format(kwargs['result_dir'], kwargs['model_name'])
        
	with open(result_file, 'w') as output:
		output.write('\n'.join(lines))
        
	print 'Write to {} successful!'.format(result_file)
