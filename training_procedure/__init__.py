from .train import train
from .evaluate import evaluate
from .prepare import prepare_train , prepare_model , init , prepare_split
from functools import partial

class Trainer:
	def __init__(self , C , logger):
		self.C = C
		self.logger = logger

		self.prepare_split 	= partial(prepare_split	, self)
		self.prepare_train 	= partial(prepare_train	, self)
		self.prepare_model 	= partial(prepare_model	, self)
		self.init 			= partial(init 			, self)

		self.train 			= partial(train			, self)		
		self.evaluate 		= partial(evaluate		, self)

		self.flags = {}
		self.split_info = None