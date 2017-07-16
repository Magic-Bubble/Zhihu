import os
import sys

def write_result(lines, **kwargs):
	load_dir = '{}/{}'.format(kwargs['model_dir'], kwargs['model_name'])
	if kwargs.get('name', None) is None:		
		checkpoints = '{}/checkpoints'.format(load_dir)
		if os.path.exists(checkpoints):
			name = open(checkpoints).readline().strip()
	else:
		name = kwargs.get('name')
	if name: result_file = '{}/{}_{}.csv'.format(kwargs['result_dir'], kwargs['model_name'], name)
	else: result_file = '{}/{}_result.csv'.format(kwargs['result_dir'], kwargs['model_name'])
        
	with open(result_file, 'w') as output:
		output.write('\n'.join(lines))
        
	print 'Write to {} successful!'.format(result_file)