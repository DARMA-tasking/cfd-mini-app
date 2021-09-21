from parser import *

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

d = dict()

cells = ["10", "15", "20", "fast10", "fast15", "fast20"]

for i in cells:
    d[i] = readTranspose(
        "./data/test." + i + "cells.dat",
        [0, 1, 2, 3],
        ["threads", "samples", "avg", "std"]
    )

################################################################################

matplotlib.rcParams.update({'font.size': 18})

c = cm.get_cmap("Set1")

fig, ax = plt.subplots(figsize=(getLongGraphLen(),getLongGraphWidth()))

x = 0.1
for i in cells:
    ax.errorbar(d[i]["threads"], d[i]["avg"], d[i]["std"], label=i + " cells", linewidth=1.6, color=c(0.1+x), marker='^')
    x = 0.1 + x

ax.legend(loc='upper right', ncol=2, fontsize=14)
ax.yaxis.labelpad=16
ax.set_xlabel('OpenMP Threads')
ax.set_ylabel('CG Time (s)')

# ax.set_xscale('log')

# ax.set_yscale('log')
# plt.ylim([10, 1000])
plt.ylim([0, 60])

# plt.ylim([0, 420])
# plt.ylim([0, 170])
# plt.xlim([0, 1600])
plt.rc('font', size=18)

plt.title("CFD Test Program")

# plt.text(-150, 1.25, '(d)', fontsize=20.0, va='center')

addAllGridLines(ax)

writeFile(__file__, fig)

