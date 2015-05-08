import numpy as np
import matplotlib.pyplot as plt
import random
from datautil import *

#x = np.linspace(0, 10, 1000)
x = range(1, 11)

c = get_rating_dist('CiteUlike2')
m = get_rating_dist('MovieLens2')

plt.figure(figsize=(10, 8))
n, bins, patches = plt.hist([c, m], histtype='bar',
                            color=['crimson', 'blue'],
                            label=['CiteULike', 'MovieLens'])
plt.legend(loc='upper left', numpoints=1)
filename = 'rating_dist.png'
plt.savefig('/Users/jiangming/Dropbox/Research/Latex/papers/TTARM/figures/' + filename)
