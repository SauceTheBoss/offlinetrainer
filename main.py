import os
import sys
from pathlib import Path
from PremotorCortex import PremotorCortex, state_file

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, path)  # this is for first process imports


if __name__ == '__main__':
	import utils
	#utils.MakeSmallData("../Saltie/framedata/gooddata0")
	utils.SetSeed(771986)
	p = PremotorCortex()
	p.cacheData()

	# my_file = Path(state_file)
	# if my_file.is_file():
	# 	p.LoadStates()
	# else:
	# 	p.primeModels()
	
	p.primeModels()
	p.mainloop()

