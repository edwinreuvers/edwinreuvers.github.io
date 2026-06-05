#| label: fig-i-ssc-parameterisation
#| fig-cap: Representation of the parameterisation of stretch-shortening cycles. $T_{short}$ and $T_{leng}$ denote the 
#|   shortening and lengthening durations, respectively, of either the muscle-tendon-complex (MTC) or the muscle fibres. 
#|   FTS denotes the fraction of the cycle time spent shortening. In the example shown, the MTC/muscle fibres shorten 65% 
#|   of the cycle duration (i.e., FTS = 0.65).

#%% Load packages & set directories
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import cust_fig, trajectories

plt.close('all')

#%% Generate lmtc(t)
cf = 1
fts = 0.65
amp = 1 # [m]
lmtcAvg = 0
acc = 100

time = np.linspace(0,1/cf,1000)
lmtc = trajectories.scv(time,cf,fts,amp,lmtcAvg,acc)[0]

#%% Make figure
cust_fig.style(plt, fontname='Minion Pro',fontsize=11,grid='on')

fig = plt.figure(figsize=(7.96/2.54+0.002, (4.80)/2.54), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)
gs = fig.add_gridspec(1,1)
axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(gs.ncols)] for i in range(gs.nrows)])

#%% Plot
ax = axs[0,0]
ax.plot(time,lmtc,'k')
ax.set_xlim(0,1/cf+0.05)
ax.set_ylim(-1.2,1.1)

# Plot amp arrow
ax.annotate("", xy=(0, -1), xytext=(0, 1),
            arrowprops=dict(arrowstyle="<->"))
text = ax.text(-0.075,0, "MTC/muscle fibre \n length excursion", color='k',ha='center', va='center', rotation = 90)

#
ax.annotate("", xy=(fts/cf, -1.1), xytext=(0, -1.1),
            arrowprops=dict(arrowstyle="<->"))
ax.text(fts/cf/2,-1.35, '\large $T_{short}$', color='k',ha='center', usetex=True)

ax.annotate("", xy=(1, -1.1), xytext=(fts/cf, -1.1),
            arrowprops=dict(arrowstyle="<->"))
ax.text(fts/cf+(1-fts)/cf/2,-1.35, "\large $T_{length}$", color='k',ha='center', usetex=True)

# CF & FTS
ax.text(
    -0.128, -1.8,
    #r"$\substack{\text{\normalsize Cycle} \\ \text{\normalsize frequency}} = \frac{1}{T_{short} + T_{length}}$",
    r"\large $\substack{\text{Cycle} \\ \text{frequency}} \ = \ $" +r"\Large $\frac{1}{T_{short} + T_{length}}$",
    color='k',
    ha='left',
    va='center',
    usetex=True
)
ax.text(
    1/cf,-1.8,
    #r"$\text{\normalsize FTS} = \frac{T_{short}}{T_{short} + T_{length}}$",
    r"\normalsize $\text{FTS} \ = \ $" + r"\Large $\frac{T_{short}}{T_{short} + T_{length}}$",
    color='k',
    ha='right',
    va='center',
    usetex=True
)

ax.set_xticks([0,fts/cf,1/cf])
ax.set_xticklabels(['','','']) 
ax.set_yticks([])
# ax.set_yticks([-1,0,1])
# ax.set_yticklabels(['','$L_{MTC}^{avg}$',''])
# ax.set_yticklabels(['$L_{MTC}^{avg} - AMP$','$L_{MTC}^{avg}$','$L_{MTC}^{avg} + AMP$']) 
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# %% Show and save
plt.show()
# fig.savefig("i_ssc_parameterisation.png", bbox_inches="tight", pad_inches=0, dpi=600)
fig.savefig("i_ssc_parameterisation.pdf", bbox_inches="tight", pad_inches=0)
# fig.savefig("i_ssc_parameterisation.svg", bbox_inches="tight", pad_inches=0)

# %% Checks
if len(sys.argv) > 1:
    check_size = sys.argv[1]
else:
    check_size = True  

if check_size == True or check_size == 'True':
    cust_fig.report_axes_size(fig,axs)
    # cust_fig.report_fig_size("i_ssc_parameterisation.png") 
    cust_fig.report_fig_size("i_ssc_parameterisation.pdf") 
    # cust_fig.report_fig_size("i_ssc_parameterisation.svg") 