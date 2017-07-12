import logging

class Logger:
	def __init__(self):
		self.logger = logging.getLogger('')
		logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
		logging.root.setLevel(level=logging.INFO)

	def info(self, info):
		self.logger.info(info)