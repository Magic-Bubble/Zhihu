class DefaultConfig(dict):
    def __init__(self):
		self['data_dir'] = '../data_preprocess/'
		self['embed_dir'] = self['data_dir'] + 'embed/'
		self['char_embed'] = self['embed_dir'] + 'char_embed_mat.npy'
		self['word_embed'] = self['embed_dir'] + 'word_embed_mat.npy'

		self['train_dir'] = self['data_dir'] + 'train/'
		self['train_title_char'] = self['train_dir'] + 'train_title_char_indices.npy'
		self['train_title_word'] = self['train_dir'] + 'train_title_word_indices.npy'
		self['train_desc_char'] = self['train_dir'] + 'train_desc_char_indices.npy'
		self['train_desc_word'] = self['train_dir'] + 'train_desc_word_indices.npy'
		self['train_label'] = self['train_dir'] + 'train_label_indices.npy'

		self['train_title_char_2'] = self['train_dir'] + 'train_title_char_indices_2.npy'
		self['train_title_word_2'] = self['train_dir'] + 'train_title_word_indices_2.npy'
		self['train_desc_char_2'] = self['train_dir'] + 'train_desc_char_indices_2.npy'
		self['train_desc_word_2'] = self['train_dir'] + 'train_desc_word_indices_2.npy'

		self['train_title_char_2_shuffle'] = self['train_dir'] + 'train_title_char_indices_2_shuffle.npy'
		self['train_title_word_2_shuffle'] = self['train_dir'] + 'train_title_word_indices_2_shuffle.npy'
		self['train_desc_char_2_shuffle'] = self['train_dir'] + 'train_desc_char_indices_2_shuffle.npy'
		self['train_desc_word_2_shuffle'] = self['train_dir'] + 'train_desc_word_indices_2_shuffle.npy'

		self['val_dir'] = self['data_dir'] + 'val/'
		self['val_title_char'] = self['val_dir'] + 'val_title_char_indices.npy'
		self['val_title_word'] = self['val_dir'] + 'val_title_word_indices.npy'
		self['val_desc_char'] = self['val_dir'] + 'val_desc_char_indices.npy'
		self['val_desc_word'] = self['val_dir'] + 'val_desc_word_indices.npy'
		self['val_label'] = self['val_dir'] + 'val_label_indices.npy'

		self['val_title_char_2'] = self['val_dir'] + 'val_title_char_indices_2.npy'
		self['val_title_word_2'] = self['val_dir'] + 'val_title_word_indices_2.npy'
		self['val_desc_char_2'] = self['val_dir'] + 'val_desc_char_indices_2.npy'
		self['val_desc_word_2'] = self['val_dir'] + 'val_desc_word_indices_2.npy'

		self['val_title_char_2_shuffle'] = self['val_dir'] + 'val_title_char_indices_2_shuffle.npy'
		self['val_title_word_2_shuffle'] = self['val_dir'] + 'val_title_word_indices_2_shuffle.npy'
		self['val_desc_char_2_shuffle'] = self['val_dir'] + 'val_desc_char_indices_2_shuffle.npy'
		self['val_desc_word_2_shuffle'] = self['val_dir'] + 'val_desc_word_indices_2_shuffle.npy'

		self['train_all_dir'] = self['data_dir'] + 'train_all/'
		self['train_title_char_all'] = self['train_all_dir'] + 'train_title_char_indices_all.npy'
		self['train_title_word_all'] = self['train_all_dir'] + 'train_title_word_indices_all.npy'
		self['train_desc_char_all'] = self['train_all_dir'] + 'train_desc_char_indices_all.npy'
		self['train_desc_word_all'] = self['train_all_dir'] + 'train_desc_word_indices_all.npy'
		self['train_label_all'] = self['train_all_dir'] + 'train_label_indices_all.npy'

		self['train_title_char_all_2'] = self['train_all_dir'] + 'train_title_char_indices_all_2.npy'
		self['train_title_word_all_2'] = self['train_all_dir'] + 'train_title_word_indices_all_2.npy'
		self['train_desc_char_all_2'] = self['train_all_dir'] + 'train_desc_char_indices_all_2.npy'
		self['train_desc_word_all_2'] = self['train_all_dir'] + 'train_desc_word_indices_all_2.npy'

		self['train_title_char_all_2_shuffle'] = self['train_all_dir'] + 'train_title_char_indices_all_2_shuffle.npy'
		self['train_title_word_all_2_shuffle'] = self['train_all_dir'] + 'train_title_word_indices_all_2_shuffle.npy'
		self['train_desc_char_all_2_shuffle'] = self['train_all_dir'] + 'train_desc_char_indices_all_2_shuffle.npy'
		self['train_desc_word_all_2_shuffle'] = self['train_all_dir'] + 'train_desc_word_indices_all_2_shuffle.npy'

		self['test_dir'] = self['data_dir'] + 'test/'
		self['test_idx'] = self['test_dir'] + 'test_idx.npy'
		self['test_title_char'] = self['test_dir'] + 'test_title_char_indices.npy'
		self['test_title_word'] = self['test_dir'] + 'test_title_word_indices.npy'
		self['test_desc_char'] = self['test_dir'] + 'test_desc_char_indices.npy'
		self['test_desc_word'] = self['test_dir'] + 'test_desc_word_indices.npy'

		self['test_title_char_2'] = self['test_dir'] + 'test_title_char_indices_2.npy'
		self['test_title_word_2'] = self['test_dir'] + 'test_title_word_indices_2.npy'
		self['test_desc_char_2'] = self['test_dir'] + 'test_desc_char_indices_2.npy'
		self['test_desc_word_2'] = self['test_dir'] + 'test_desc_word_indices_2.npy'

		self['test_title_char_2_shuffle'] = self['test_dir'] + 'test_title_char_indices_2_shuffle.npy'
		self['test_title_word_2_shuffle'] = self['test_dir'] + 'test_title_word_indices_2_shuffle.npy'
		self['test_desc_char_2_shuffle'] = self['test_dir'] + 'test_desc_char_indices_2_shuffle.npy'
		self['test_desc_word_2_shuffle'] = self['test_dir'] + 'test_desc_word_indices_2_shuffle.npy'

		self['topic_dir'] = self['data_dir'] + 'topic/'
		self['topic_idx'] = self['topic_dir'] + 'topic_idx.npy'

		self['use_word'] = False
		self['use_char'] = False
		self['use_char_word'] = False
		self['use_word_char'] = False
		self['static'] = True
		self['use_double_length'] = True
		self['data_shuffle'] = False
        
		self['class_num'] = 1999
		self['word_embed_num'] = 411722
		self['char_embed_num'] = 11975
		self['embed_dim'] = 256
		self['val_num'] = 299997
		self['test_num'] = 217360
        
		self['kernel_num'] = 512
		self['kernel_sizes'] = [1,2,3,4,5]
		self['dropout'] = 0.5    

		self['model'] = 'TextCNN'
		self['use_self_loss'] = False
		self['loss_function'] = 'MultiLabelSoftMarginLoss'
		self['load_loss_weight'] = False
		self['load'] = False
		self['load_name'] = None
		self['model_dir'] = 'snapshots'
		
		self['lr'] = 0.001
		self['lr_decay'] = 0.1
		self['rho'] = 0.95
		self['epochs'] = 5
		self['base_epoch'] = 0
		self['batch_size'] = 256
		self['folder_num'] = 10
		self['base_folder'] = 0
		self['log_interval'] = 10
		self['eval_interval'] = 1000
		self['save_interval'] = 5000
		self['device'] = 0
		self['cuda'] = True
		self['threshold'] = 0.5
		self['boost'] = False
		self['base_layer'] = 0

		self['result_dir'] = result_dir = 'results'
        
    def update(self, **args):
        for key in args:
            self[key] = args[key]
