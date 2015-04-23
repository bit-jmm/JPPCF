from model.ttarm import Ttarm
import os
ttarm = Ttarm(k=20, lambd=10, eta=0.3)
log_path = os.path.realpath(os.path.join(__file__, '../log'))
if not os.path.isdir(log_path):
    os.mkdir(log_path)
ttarm.run()