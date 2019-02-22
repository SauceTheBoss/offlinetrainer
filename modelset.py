import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

class ModelSet():
	id = 0

	def __init__(self, state_dims, action_dims, device, generation=0, parent=0):
		from models import ActionSelecter, M1pM1, ActionCreator
		self.base1 = M1pM1(state_dims, action_dims).to(device)
		self.base2 = M1pM1(state_dims, action_dims).to(device)
		self.selector = ActionSelecter(self.base1).to(device)
		self.creator = ActionCreator(self.base1).to(device)
		self.optimizer_selector = self._opti(self.selector.parameters())
		self.optimizer_creator = self._opti(self.creator.parameters())
		self.device = device
		self.state_dims = state_dims
		self.action_dims = action_dims
		self.gen = generation
		self._name = self._set_modelname()
		self._parent = parent
		self._id = ModelSet.id
		ModelSet.id += 1

		self.coverageMask = None

		self.limiter = 0.1
		self.limiter_count = 0


	def _set_modelname(self):
		return random.sample(["ERIC", "CATS", "DOGS", "FROG", "PIPE", "ONES", "TWOS", "FINE", "LIME", "DIME", "PINE", "DAMN", "BOYS", "JESS", "MESS", "DUDE", "DUMB", "BUTT", "LOVE", "LIFE", "HUGS", "NEAR", "RING", "WOLF", "FISH", "FIVE", "KING", "ELSE", "TREE", "OVER", "TIME", "ABLE", "HAVE", "SING", "STAR", "CITY", "SOUL", "RICH", "DUCK", "FOOT", "FILM", "LION", "ANNA", "MEME", "LIVE", "SAFE", "PAIN", "RAIN", "SION", "IRON", "ONCE", "BALL", "WITH", "FIRE", "WOOD", "CARE", "CAKE", "BACK", "LADY", "AWAY", "WORK", "SELF", "MOLE", "MOON", "GOLF", "ALLY", "NINE", "MARY", "BODY", "LESS", "DOWN", "LAND", "BLUE", "GONE", "KATE", "COME", "HIGH", "HARD", "ROCK", "TEEN", "ROSE", "WISH", "TING", "BABY", "HOME", "LONG", "HAND", "GIRL", "FOOD", "HOPE", "WIND", "TINA", "BORN", "OPEN", "WIFE", "BIRD", "SURE", "BEAN", "HAIR", "ROOM", "LATE", "MINE", "FALL", "HERO", "BELL", "EVER", "DARK", "JACK", "EVIL", "LEAF", "LIST", "MATH", "GOAT", "KIDS", "ADAM", "HEAD", "SHIP", "FACE", "ERIN", "WINE", "MANY", "HATE", "GOOD", "EDGE", "LIKE", "OVEN", "ZONE", "PEAR", "DESK", "FEAR", "ZEUS", "SIDE", "GATE", "DANA", "ZING", "DONE", "ASIA", "BEAR", "BONE", "PINK", "EMMA", "TACO", "NINA", "BAND", "SAND", "MATE", "EAST", "SNOW", "MAYA", "UNIT", "BEST", "GIFT", "BUSH", "SIAN", "KISS", "RATE", "YEAH", "MOVE", "MARK", "CORN", "PLAY", "ZERO", "CARD", "LING", "BOOK", "ALSO", "TOWN", "FREE", "LOST", "BOMB", "SICK", "PROM", "WORD", "LORD", "ACID", "LILY", "DOVE", "THAT", "REST", "RYAN", "IDEA", "THEM", "CENT", "TATE", "GRAM", "FOUR", "GOLD", "TUBE", "GAME", "YEAR", "THIN", "COOK", "DISH", "FULL", "BEAU", "ITCH", "BATH", "GROW", "RAGE", "HOOD", "WORM", "KITE", "SOFT", "SHOT", "EVEN", "POEM", "KNOW", "DING", "NICE", "STEP", "LOAF", "FORM", "JUNE", "RICE", "PATH", "UGLY", "SILK"], 1)[0]


	def UpdateLimiter(self):
		self.limiter_count += 1
		self.limiter = max(0.001, self.limiter * 0.999**self.limiter_count)
		return self.limiter

	def LimiterNear(self):
		return (self.limiter <= 0.01)

	def LimiterReached(self):
		return (self.limiter <= 0.001)

	def CoverageAppend(self, coverageMask):
		if self.coverageMask is None:
			self.coverageMask = coverageMask
		else:
			self.coverageMask *= coverageMask #both are masks, so this should the same as doing a bitwise AND.  Should reset some to 0 if missing this round
			self.coverageMask += coverageMask

	def CheckCoverageMask(self, repeats_req = 10, trigger_count = 1):
		triggered = torch.masked_select(self.coverageMask, self.coverageMask.ge(repeats_req))
		t1 = torch.masked_select(self.coverageMask, self.coverageMask.ge(1))
		t3 = torch.masked_select(self.coverageMask, self.coverageMask.ge(3))
		t5 = torch.masked_select(self.coverageMask, self.coverageMask.ge(5))
		s = "CheckCoverageMask name:{0:>27}  t1:{1:5d}  t3:{2:5d}  t5:{3:5d}  t10:{4:5d}"
		print(s.format(self.name(), len(t1), len(t3), len(t5), len(triggered)))
		return (trigger_count <= len(triggered))

	def GetCoverageMask(self, repeats_req = 1):
		return self.coverageMask.ge(repeats_req).flatten()

	def GetCoverage(self, batch, max_loss, min_pct):
		losses = self.getCreatorLoss(batch)
		pcts = self.getSelectorPct(batch)

		losses = torch.le(losses, max_loss)
		pcts = torch.ge(pcts, min_pct)

		return losses * pcts #both are masks, so this should the same as doing a bitwise AND



	def getSelectorPct(self, data) -> float:
		self.lock()
		with torch.no_grad():
			s1, s2, prev_action, action, _, _ = data
			rtn = self.selector.forward(s1, s2, prev_action)
			
		return rtn

	def getCreatorLoss(self, data):
		self.lock()

		with torch.no_grad():
			s1, s2, prev_action, action, _, _ = data
			c_action = self.creator.forward(s1, s2, prev_action)
			rtn_loss = self._loss_fn(c_action, action, loss_reduction='none')
			del c_action
			
		return rtn_loss


	

	def TrainCreator(self, data):
		self.unlock()

		self.optimizer_creator.zero_grad()

		s1, s2, prev_action, action, _, _ = data

		size = len(s1)
		if size == 0:
			return

		c_vals = self.creator.forward(s1, s2, prev_action)
		c_loss = self._loss_fn(c_vals, action)
		c_loss.backward()
		self.optimizer_creator.step()
		del c_vals, c_loss

	def TrainCreator_hopper(self, data):
		self.unlock()

		self.optimizer_creator.zero_grad()

		data, mask = data
		s1, s2, prev_action, action, _, _ = data

		mask = mask.flatten()
		mask = torch.nonzero(mask).flatten()
		s1 = torch.index_select(s1, 0, mask)
		s2 = torch.index_select(s2, 0, mask)
		prev_action = torch.index_select(prev_action, 0, mask)
		action = torch.index_select(action, 0, mask)

		size = len(s1)
		if size == 0:
			return

		c_vals = self.creator.forward(s1, s2, prev_action)
		c_loss = self._loss_fn(c_vals, action)
		c_loss.backward()
		self.optimizer_creator.step()
		del c_vals, c_loss

	def TrainSelector_hopper(self, data):
		self.unlock()

		self.optimizer_selector.zero_grad()

		data, mask = data
		s1, s2, prev_action, action, _, _ = data
		size = len(s1)
		if size == 0:
			return
		target = mask.flatten()

		s_val = self.selector.forward(s1, s2, prev_action)
		s_loss = self._loss_fn(s_val, target)
		s_loss.backward()
		self.optimizer_selector.step()
		del s_val, s_loss

	def TrainSelector(self, data, train_for_invalid=False, train_with_invalid=False):
		self.unlock()

		self.optimizer_selector.zero_grad()

		s1, s2, prev_action, action, invalid1, invalid2 = data

		size = len(s1)
		if size == 0:
			return

		if train_for_invalid:
			if train_with_invalid:
				s1 = invalid1
				s2 = invalid2
			target = torch.zeros(size, device=self.device)
		else:
			target = torch.ones(size, device=self.device)

		s_val = self.selector.forward(s1, s2, prev_action)
		s_loss = self._loss_fn(s_val, target)
		s_loss.backward()
		self.optimizer_selector.step()
		del s_val, s_loss


	def Evaluation(self, data, single_output=True):
		self.lock()
		with torch.no_grad():
			s1, s2, prev_action, action, _, _ = data
			rtn_selector = self.selector.forward(s1, s2, prev_action)
			c_action = self.creator.forward(s1, s2, prev_action)

			rtn_loss = self._loss_fn(c_action, action)
			rtn_selector = torch.mean(rtn_selector)
			del c_action

		return float(rtn_selector.cpu()), float(rtn_loss.cpu())

	def lock(self):
		self._m_lock(self.selector)
		self._m_lock(self.creator)
		self._m_lock(self.base1)
		self._m_lock(self.base2)

	def unlock(self):
		self._m_unlock(self.selector)
		self._m_unlock(self.creator)
		self._m_unlock(self.base1)
		self._m_unlock(self.base2)

	def _m_lock(self, model):
		if model.training:
			model.eval()
			for param in model.parameters():
				param.requires_grad = False

	def _m_unlock(self, model):
		if not model.training:
			model.train()
			for param in model.parameters():
				param.requires_grad = True

	def _loss_fn(self, input, target, loss_reduction='mean'):
		rtn = F.mse_loss(input, target, reduction=loss_reduction)
		if loss_reduction == "none":
			rtn = torch.mean(rtn, dim=1, keepdim=True)
		return rtn


	def _opti(self, parameters):
		return optim.Adamax(parameters)

	def clone(self, namePrefix=" "):
		rtn = ModelSet(self.state_dims, self.action_dims, self.device, self.gen+1, parent=self.id)
		rtn.creator.load_state_dict(self.creator.state_dict())
		rtn.selector.load_state_dict(self.selector.state_dict())
		rtn.base1.load_state_dict(self.base1.state_dict())
		rtn.base2.load_state_dict(self.base2.state_dict())
		rtn.optimizer_selector = self._opti(rtn.selector.parameters())
		rtn.optimizer_creator = self._opti(rtn.creator.parameters())
		rtn.optimizer_creator.load_state_dict(self.optimizer_creator.state_dict())
		rtn.optimizer_selector.load_state_dict(self.optimizer_selector.state_dict())

		rtn._name += " " + self._name
		rtn._name = namePrefix + rtn._name
		rtn._name = rtn._name[:23]

		return rtn

	def name(self):
		return self._name + " " + str(self.gen)

	
	def Export(self):
		state = {
			'selector': self.selector.state_dict(),
			'creator': self.creator.state_dict(),
			'base1': self.base1.state_dict(),
			'base2': self.base2.state_dict(),
			'optimizer_selector': self.optimizer_selector.state_dict(),
			'optimizer_creator': self.optimizer_creator.state_dict(),
			'name': self._name,
			'gen': self.gen,
			'id': self.id,
			'state_dims': self.state_dims,
			'action_dims': self.action_dims
		}
		return state

	def Import(self, state):
		self.creator.load_state_dict(state["creator"])
		self.selector.load_state_dict(state["selector"])
		self.base1.load_state_dict(state["base1"])
		self.base2.load_state_dict(state["base2"])
		self.optimizer_creator.load_state_dict(state["optimizer_creator"])
		self.optimizer_selector.load_state_dict(state["optimizer_selector"])
		self._name = state["name"]
		self.gen = state["gen"]
		self._id = state["id"]
		self.action_dims = state["action_dims"]
		self.state_dims = state["state_dims"]
