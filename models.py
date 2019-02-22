import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# class SpacialConv(nn.Module):
# 	def __init__(self, input_dims, output_dims):
# 		super(SpacialConv, self).__init__()

# 		self.input_sqr = self.nextPerfectSquare(input_dims)

# 		self.conv = nn.Conv2d(1, 1, kernel_size=2, padding=1)

# 		conv_size = self.get_flat_fts((math.sqrt(self.input_sqr),math.sqrt(self.input_sqr)), self.conv)
# 		self.f_out = nn.Sequential(
# 			nn.Linear(conv_size, conv_size*2),
# 			nn.Linear(conv_size*2, output_dims)
# 		)

# 		return

# 	def get_flat_fts(self, in_size, fts):
# 		f = fts(torch.ones(1,*in_size))
# 		return int(np.prod(f.size()[1:]))

# 	def nextPerfectSquare(N): 
# 		nextN = math.floor(math.sqrt(N)) + 1
# 		return nextN * nextN 

# 	def forward(self, state):
# 		return



	

# class M1(nn.Module):
# 	def __init__(self, input_dims):
# 		super(M1, self).__init__()

# 		self.input_dims = input_dims
# 		self.output_dims = int(input_dims / 2)
		
# 		self.seq = nn.Sequential(
# 			nn.Linear(self.input_dims, self.input_dims*3),
			
# 			nn.Linear(self.input_dims*3, self.input_dims*3),
			
# 			nn.Linear(self.input_dims*3, self.input_dims*3),
			
# 			nn.Linear(self.input_dims*3, self.output_dims)
# 		)

# 		self.l1 = nn.Linear(self.input_dims, self.output_dims)

# 	def forward(self, state):
# 		return self.seq(state) + self.l1(state)

drop_rate = 0.01

def cheese_layer():
	return nn.Sequential(
		nn.PReLU()
	)

class M1pM1(nn.Module):
	input_dims = 0
	def __init__(self, input_dims, action_dims):
		super(M1pM1, self).__init__()

		self.input_dims = input_dims
		self.action_dims = action_dims

		self.seq = nn.Sequential(
			nn.Linear(input_dims*2+action_dims, input_dims*4),
			cheese_layer(),
			nn.Linear(input_dims*4, input_dims*4),
			cheese_layer(),
			nn.Linear(input_dims*4, input_dims*4),
			cheese_layer(),
			nn.Linear(input_dims*4, input_dims)
		)

		self.seq2 = nn.Sequential(
			nn.Linear(input_dims, input_dims*4),
			cheese_layer(),
			nn.Linear(input_dims*4, input_dims*4),
			cheese_layer(),
			nn.Linear(input_dims*4, input_dims*4),
			cheese_layer(),
			nn.Linear(input_dims*4, input_dims)
		)

		self.bi2 = nn.Bilinear(input_dims, input_dims, input_dims)



	def forward(self, s1, s2, prev_action):
		cc = torch.cat((s1,s2,prev_action),dim=(1 if prev_action.dim() == 2 else 0))

		return self.seq(cc) + self.seq2(self.bi2(s1,s2))

class ActionSelecter(nn.Module):
	input_dims = 0
	# last_layer = None
	def __init__(self, base_m1pm1):
		super(ActionSelecter, self).__init__()

		self.cc = base_m1pm1
		self.input_dims = base_m1pm1.input_dims
		self.last_layer = nn.Linear(self.input_dims*2, 1)


		self.seq = nn.Sequential(
			nn.Linear(self.input_dims, self.input_dims*4),
			cheese_layer(),
			nn.Linear(self.input_dims*4, self.input_dims*4),
			cheese_layer(),
			nn.Linear(self.input_dims*4, self.input_dims*2),
			cheese_layer()
		)

	def forward(self, s1, s2, prev_action):
		x = self.seq(self.cc(s1, s2, prev_action))
		return torch.sigmoid(self.last_layer(x))
		

	def reset_last_layer(self):
		self.last_layer.reset_parameters()


class ActionCreator(nn.Module):
	def __init__(self, base_m1pm1):
		super(ActionCreator, self).__init__()

		self.cc = base_m1pm1
		input_dims = base_m1pm1.input_dims
		output_dims = base_m1pm1.action_dims


		self.seqs = nn.ModuleList()
		for i in range(output_dims):
			self.seqs.append(SubActionCreator(input_dims, nn.Tanh()))

	def forward(self, s1, s2, prev_action):
		cc_val = self.cc(s1,s2, prev_action)

		outputs = []
		for	s in self.seqs:
			outputs.append(s(cc_val))

		rtn = torch.cat(outputs, dim=(1 if prev_action.dim() == 2 else 0))

		return rtn + prev_action

class SubActionCreator(nn.Module):
	def __init__(self, input_dims, last_layer):
		super(SubActionCreator, self).__init__()	

		self.seq = nn.Sequential(
			nn.Linear(input_dims, input_dims*4),
			cheese_layer(),
			nn.Linear(input_dims*4, input_dims*4),
			cheese_layer(),
			nn.Linear(input_dims*4, input_dims*2),
			cheese_layer(),
			nn.Linear(input_dims*2, 1),
			last_layer
		)

		self.identy_ish = nn.Linear(input_dims, 1)

		self.last_layer = last_layer

	def forward(self, input):
		return self.last_layer(self.seq(input) + self.identy_ish(input))