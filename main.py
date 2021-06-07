import os
import pdb
import random
import sys
import time
from pprint import pformat

import numpy
import torch as tc
import torch.nn as nn
from tqdm import tqdm
from utils.logger import Logger
from utils.random_seeder import set_random_seed
from config import get_config
from training_procedure import Trainer


def main(C , logger , run_id = 0):
	
	T = Trainer(C = C , logger = logger)

	T.flags["change split"] = ( run_id % C.change_split == 0 )

	(graph , labels) , (train_nodes , dev_nodes , test_nodes) , model , (optimizer , loss_func) = T.init(idx = run_id)

	patience_cnt 		= 0
	maj_metric 			= "micro"
	best_metric 	  	= 0
	best_metric_epoch 	= -1 # best number on dev set
	report_dev_res 		= 0
	report_tes_res 		= 0
	pbar = tqdm(range(C.num_epoch) , ncols = 130)
	for epoch_id in pbar:

		model , loss = T.train(graph, labels, train_nodes, model, loss_func, optimizer)

		dev_res, tes_res, tra_res = T.evaluate(
			graph, labels, [dev_nodes, test_nodes, train_nodes], model, loss_func
		)

		now_metric = dev_res[maj_metric] # current number on dev set

		if C.no_dev or best_metric <= now_metric:
			best_metric 		= now_metric
			best_metric_epoch 	= epoch_id
			report_dev_res 		= dev_res
			report_tes_res 		= tes_res

			patience_cnt 		= 0
		else:
			patience_cnt += 1

		if C.patience > 0 and patience_cnt >= C.patience:
			break

		postfix_str = "<%d> [Dev] %.2f [Test] %.2f (%.2f) [Train] %.2f" % ( epoch_id , 
			dev_res[maj_metric], tes_res[maj_metric], report_tes_res[maj_metric], tra_res[maj_metric]
		)
		pbar.set_postfix_str(postfix_str)

	logger.log("best epoch is %d" % best_metric_epoch)
	logger.log("Best Epoch Valid Acc is %.2f" % (report_dev_res[maj_metric]))
	logger.log("Best Epoch Test  Acc is %.2f" % (report_tes_res[maj_metric]))

	# note returned tra_res is always that of last epoch
	return model , report_dev_res , report_tes_res , tra_res

if __name__ == "__main__":

	C = get_config()

	# init logger
	logger = Logger(mode = [print])
	logger.add_line = lambda : logger.log("-" * 50)
	logger.log(" ".join(sys.argv))
	logger.add_line()
	logger.log()

	if C.seed > 0:
		set_random_seed(C.seed)
		logger.log ("Seed set. %d" % (C.seed))

	# start run
	seeds = [random.randint(0,233333333) for _ in range(C.multirun)]
	dev_ress = []
	tes_ress = []
	tra_ress = []
	for run_id in range(C.multirun):
		logger.add_line()
		logger.log ("\t\t%d th Run" % run_id)
		logger.add_line()
		set_random_seed(seeds[run_id])
		logger.log ("Seed set to %d." % seeds[run_id])

		model , dev_res , tes_res , tra_res = main(C , logger , run_id)

		logger.log("%d th Run ended. Best Epoch Valid Result is %s" % (run_id , str(dev_res)))
		logger.log("%d th Run ended. Best Epoch Test  Result is %s" % (run_id , str(tes_res)))
		logger.log("%d th Run ended. Final      Train Result is %s" % (run_id , str(tra_res)))

		dev_ress.append(dev_res)
		tes_ress.append(tes_res)
		tra_ress.append(tra_res)

	logger.add_line()

	for metric in ["micro" , "macro"]:
		for res , name in zip(
			[dev_ress , tes_ress , tra_ress] , 
			["Dev" , "Test" , "Train"]
		):
			now_res = [x[metric] for x in res]

			logger.log ("%s of %s : %s" % (metric , name , str([round(x,2) for x in now_res])))

			avg = sum(now_res) / C.multirun
			std = (sum([(x - avg) ** 2 for x in now_res]) / C.multirun) ** 0.5

			logger.log("%s of %s : avg / std = %.2f / %.2f" % (metric , name , avg , std))
		logger.log("")
