import torch
import utils
import gc
import pickle
from torch.utils.data import DataLoader
from ValuesDataset import ValuesDataset, CreateSubSet, TrainingSubset
from modelset import ModelSet
from tensorboardX import SummaryWriter

frame_file = 'gooddata0.pt'
state_file = "./state.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
workers = 0 #(4 if device == "cpu" else 0)
writer = SummaryWriter()
DEBUG_QUICK = False

class PremotorCortex():
	def __init__(self):
		self.data = ValuesDataset(frame_file,device)
		self.modelsets = []
		self.primeModel = None

		self.batch_count = 20000 #(100 if DEBUG_QUICK else 20000)				#10000 frames is aprox a full games worth (including overtime at 30fps)
		self.train_batchsize = (100 if DEBUG_QUICK else 500)
		print("device")
		return

	def cacheData(self):
		if not self.data.use_cache:
			print("loading data to cache")
			for i in range(len(self.data)):
				o = self.data.__getitem__(i)
			self.data.use_cache = True

	def primeModels(self):
		from modelset import ModelSet
		iter = 0
		self.primeModel = ModelSet(16, 8, device)
		self.primeModel._name = " " + self.primeModel._name
		
		self.loader = DataLoader(self.data, batch_size=self.train_batchsize, shuffle=True, num_workers=workers) 
		self.time_start = utils.TimestampMillisec64()

		while iter < 500000:
			for	batch_iter, batch_sample in enumerate(self.loader):
				batch_size = eval_size = len(batch_sample[0])
				iter += batch_size

				time_train_start = utils.TimestampMillisec64()

				self.primeModel.TrainSelector(batch_sample)
				self.primeModel.TrainSelector(batch_sample, train_for_invalid=True, train_with_invalid=True)
				self.primeModel.TrainCreator(batch_sample)

				time_train_stop = time_eval_start = utils.TimestampMillisec64()

				pct, loss = self.primeModel.Evaluation(batch_sample)

				time_cur = utils.TimestampMillisec64()
				run_mins = (time_cur - self.time_start) / 1000 / 60
				train_rate = batch_size / (time_train_stop - time_train_start) * 1000 
				eval_rate = (0 if (time_cur - time_eval_start) == 0 else eval_size / (time_cur - time_eval_start) * 1000)
				selector_acc = pct * 100
				action_loss = loss
				print("Iter: {0:7d}    Trains/s: {3:8.1f}    Evals/s: {4:8.1f}   run_mins: {5:5.1f}    selector_acc: {1:7.3f}    action_loss: {2:7.6f}".format(iter, selector_acc,action_loss, train_rate, eval_rate, run_mins))

				if DEBUG_QUICK: break
			if DEBUG_QUICK: break
		gc.collect()

	def mainloop(self):
		self.cacheData()
		self.hopper_col = []
		self.hopper_max = 1#(5 if DEBUG_QUICK else 10)
		self.hopper_mask = torch.zeros(len(self.data), device=device, dtype=torch.uint8)

		# self.hopper_col.append(self.primeModel.clone())
		# self.hopper_col.append(self.primeModel.clone())

		self.MaskLessThan(0.01)
		
		self.boarditer = 0
		while(True): #todo - exit loop is nothing is left to categorize
			while len(self.hopper_col) < 2:
				self.hopper_col.append(self.primeModel.clone())

			mask = torch.eq(self.hopper_mask, 0) #flip
			self.hopper_subset = CreateSubSet(self.data, mask)
			self.hopper_train()
			self.hopper_evaluate()
			self.hopper_promote()
			gc.collect()
			self.boarditer += 1
		return

	def MaskLessThan(self, loss_limit):
		mask = torch.eq(self.hopper_mask, 0)
		self.hopper_subset = CreateSubSet(self.data, mask)
		evalLoader = DataLoader(self.hopper_subset, batch_size=self.batch_count, num_workers=workers)
		losses = None
		for	batch_iter, batch_sample in enumerate(evalLoader):
			_losses = self.primeModel.getCreatorLoss(batch_sample)
			if losses is None:
				losses = _losses
			else:
				losses = torch.cat((losses,_losses))

		self.hopper_mask += losses.le(loss_limit).flatten()



	def hopper_train(self):
		_s = len(self.hopper_col)
		s = "training hopper collection -- size: {0}"

		print(s.format(_s))
		evalLoader = DataLoader(self.hopper_subset, batch_size=self.batch_count, num_workers=workers)
		all_losses = None
		all_pcts = None
		for m_index in range(_s):
			losses = None
			pcts = None
			for	batch_iter, batch_sample in enumerate(evalLoader):
				_losses = self.hopper_col[m_index].getCreatorLoss(batch_sample)
				#_pcts = self.hopper_col[m_index].getSelectorPct(batch_sample)

				if losses is None:
					losses = _losses
					#pcts = _pcts
				else:
					losses = torch.cat((losses,_losses))
					#cts = torch.cat((pcts,_pcts))
			if all_losses is None:
				all_losses = losses
				eq_losses = torch.ones(len(self.hopper_subset), dtype=torch.uint8)
				#all_pcts = pcts
			else:
				all_losses = torch.cat((all_losses, losses), dim=1) 
				#eq_losses = torch.eq(all_losses[:, m_index - 1], all_losses[:, m_index]) * eq_losses
				#all_pcts = torch.cat((all_pcts, pcts), dim=1) 
		
		
		#rnd = torch.randint_like(eq_losses, low=0, high=_s)

		ranks_loss = torch.argsort(all_losses, descending=True) 
		ranks_loss = torch.t(ranks_loss)

		#ranks_pcts = torch.argsort(all_pcts) 
		#ranks_pcts = torch.t(ranks_pcts)
		
		
		toRemove = []
		toClone = []
		
		for m_index in range(_s):
			mset = self.hopper_col[m_index]
			ignoreMask = all_losses[:, m_index].le(mset.UpdateLimiter())
			goodMask = ranks_loss[m_index].ge(1)
			badMask = torch.eq(goodMask, 0) #flip
			# goodMask_loss = ranks_loss[m_index].ge(gr)
			# badMask_loss = ranks_loss[m_index].le(br)

			# goodMask_pct = ranks_pcts[m_index].ge(gr)
			# badMask_pct = ranks_pcts[m_index].le(br)

			# goodMask = (goodMask_loss + goodMask_pct).ge(1)
			# badMask = badMask_loss * badMask_pct

			#break up ties
			#m_rnd = torch.eq(rnd, m_index)
			#m_eq_losses = torch.eq(eq_losses * m_rnd, 0)

			goodMask *= ignoreMask #* m_eq_losses
			badMask *= ignoreMask #* m_eq_losses
			
			ignoreIndic = torch.nonzero(ignoreMask)
			suffex = ("_limit" if mset.LimiterNear() else "")
			writer.add_histogram(mset.name() + "/losses" + suffex, all_losses[:, m_index].clone().cpu().data.numpy(), self.boarditer)
			#writer.add_histogram(mset.name() + "/ignoreIndic", ignoreIndic.clone().cpu().data.numpy(), self.boarditer)
			trainingSubset = TrainingSubset(self.hopper_subset, goodMask, badMask, min_repeat=5)
			if len(trainingSubset.goodIndices) > 0:
				s = "training hopper collection: name:{0:>27}  good: {1}  bad: {2}"
				print(s.format(mset.name(), len(trainingSubset.goodIndices), len(trainingSubset.badIndices)))
				writer.add_histogram(mset.name() + "/good" + suffex, trainingSubset.goodIndices.clone().cpu().data.numpy(), self.boarditer)
				writer.add_scalar(mset.name() + "/Num_Good", len(trainingSubset.goodIndices), self.boarditer)
				writer.add_scalar(mset.name() + "/Num_Bad", len(trainingSubset.badIndices), self.boarditer)
				writer.add_scalar(mset.name() + "/Num_Ignore", len(ignoreIndic), self.boarditer)
				toClone.append(mset)
				if len(trainingSubset) > 0:
					trainingLoader = DataLoader(trainingSubset, batch_size=self.train_batchsize, shuffle=True, num_workers=workers)
					for	batch_iter, batch_sample in enumerate(trainingLoader):
						mset.TrainSelector_hopper(batch_sample)
						mset.TrainCreator_hopper(batch_sample)

			elif len(trainingSubset.badIndices) > 0:
				s = "training hopper collection: name:{0:>27}  good: {1}  bad: {2} -removing"
				print(s.format(mset.name(), len(trainingSubset.goodIndices), len(trainingSubset.badIndices)))
				toRemove.append(self.hopper_col[m_index])
			else:
				s = "training hopper collection: name:{0:>27}  good: {1}  bad: {2} - Skipping"
				print(s.format(mset.name(), len(trainingSubset.goodIndices), len(trainingSubset.badIndices)))
				mset.coverageMask = None

		# for mset in toRemove:
		# 	self.hopper_col.remove(mset)
		
		# for mset in toClone:
		# 	self.hopper_col.append(mset.clone())


	def hopper_evaluate(self):
		print("eval hopper collection")
		_s = len(self.hopper_col)
		evalLoader = DataLoader(self.data, batch_size=self.batch_count, num_workers=workers) #using full data, not hopper
		for m_index in range(_s):
			coverage = None
			for	batch_iter, batch_sample in enumerate(evalLoader):
				if coverage is None:
					coverage = self.hopper_col[m_index].GetCoverage(batch_sample, max_loss=0.001, min_pct=0.99)
				else:
					coverage = torch.cat((coverage,self.hopper_col[m_index].GetCoverage(batch_sample,max_loss=0.001, min_pct=0.99)))
			self.hopper_col[m_index].CoverageAppend(coverage)

	def hopper_promote(self):
		print("promoting hopper collection")
		_s = len(self.hopper_col)
		toRemove = []
		for m_index in range(_s):
			mset = self.hopper_col[m_index]
			if mset.LimiterReached():
				covMask = mset.GetCoverageMask()
				self.hopper_mask = self.hopper_mask + covMask
				self.modelsets.append(mset)
				toRemove.append(mset)

		for mset in toRemove:
			self.hopper_col.remove(mset)
