import datetime


def TimestampMillisec64():
    return float((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)

def SetSeed(seed):
    import torch
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    import numpy
    numpy.random.seed(seed)

def S_Size(batch):
    return len(batch[0])

def MakeSmallData(file_name):
	import pickle
	file = open(file_name + '.pt', 'rb')
	memory = pickle.load(file)
	file.close()

	file = open(file_name + '-small.pt', 'wb')
	smallMemory = memory[:30000]
	pickle.dump(smallMemory, file)
	file.close()

class StopWatch():
    
    def __init__(self, name, perMin=False):
        self.name = name
        self.start = TimestampMillisec64()
        self.perMin = perMin

    def lap(self):
        stop = TimestampMillisec64()
        dif = (stop - self.start)
        rate = 1 / (-1 if dif == 0 else dif) * 1000 * (60 if self.perMin else 1)
        unit = ("per/min" if self.perMin else "per/sec")
        print("{0:>20}: {1:6.2f} {2}".format(self.name, rate,unit))

    def reset(self):
        self.start = TimestampMillisec64()