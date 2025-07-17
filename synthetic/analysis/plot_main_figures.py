"""
Plot the (python relevant) figures for the main paper
========================================================

    Usage: python plot_main_figures.py

    This script will plot:
        figure 3: Detailed analysis of the baseline test case
        figure 4: Sensitivity tests 
        figure 5: velocity response

"""
import sys
import os
import pickle
import cmocean
import matplotlib
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np


ISSM_DIR = os.getenv('ISSM_DIR')
sys.path.append(os.path.join(ISSM_DIR, 'src/m/dev/'))
import devpath
from issmversion import issmversion
from model import model
from meshconvert import meshconvert

from src.utils import *


md = model()
# read in mesh and pass to ISSM


# read in mesh
meshfile = '../data/geometry/synthetic_mesh.pkl'
with open(meshfile, 'rb') as meshin:
    mesh = pickle.load(meshin)

bedfile = '../data/geometry/synthetic_bed.npy'
bed = np.load(bedfile)
surfacefile = '../data/geometry/synthetic_surface.npy'
surface = np.load(surfacefile)
thickness = surface - bed
md = meshconvert(md, mesh['elements'], mesh['x'], mesh['y'])
lakepos = mesh['lakepos']

filedir = 'figures/'

# FIGURE 3
############################################################

#-------- | --------#
#   lh    |    Qr
#-------- | --------#
#    Qc time i      #
#-------- | --------#
#    Qc time ii     #
#-------- | --------#
#    Qc time iii    #
#-------- | --------#
# Qr vs N | Ub vs x #
#-------- | --------#


# Load default results
resdir = '../experiments/RES/output_run_1_Default'
ff_path = os.path.join(resdir, 'ff.npy')
ff = np.load(ff_path)
phi_path = os.path.join(resdir, 'phi.npy')
phi = np.load(phi_path)
N_path = os.path.join(resdir, 'N.npy')
N = np.load(N_path)
tt_path = os.path.join(resdir, 'tt.npy')
tt = np.load(tt_path)
lh_path = os.path.join(resdir, 'l_h.npy')
lh = np.load(lh_path)
Qr_path = os.path.join(resdir, 'Qr.npy')
Qr = np.load(Qr_path)
vel_path = os.path.join(resdir, 'vel.npy')
vel = np.load(vel_path)
Qc_path = os.path.join(resdir, 'Qc.npy')
Qc = np.load(Qc_path)
h_s_path = os.path.join(resdir, 'h_s.npy')
h_s = np.load(h_s_path)

file = os.path.join(resdir, 'paramsdict.pkl')
with open(file, 'rb') as f:
    paramsdict = pickle.load(f)

# Set the geometry
md.geometry.bed = mesh['x'] * np.sin(paramsdict['bed_angle'])
md.geometry.thickness = 50 + np.sqrt(paramsdict['surface_parabola'] * mesh['x'])
md.geometry.surface = md.geometry.bed + md.geometry.thickness
md.geometry.base = md.geometry.bed 

# indexing channels
# Find index of value closest to 7 and 9
idx1 = np.argmin(np.abs(tt - 7.1))
idx3 = np.argmin(np.abs(tt - 9))
# Find idx2: index of max Qr value between idx1 and idx3 (inclusive)
idx2_relative = np.argmax(Qr[lakepos,idx1:idx3 + 1])
idx2 = idx1 + idx2_relative



print('creating figure 3!')

fig3 = plt.figure(figsize=(7.04724, 8.75))  # J.Glac dual column figure spec
gs = gridspec.GridSpec(9, 2, figure=fig3, hspace=0.15, wspace=0.25, height_ratios=[
    0.75,   # Row 0
    0.25,   # Spacer (between 0 and 1)
    1.4,    # Row 1
    0.0005,   # Spacer (between 1 and 2)
    1.4,    # Row 2
    0.0005,   # Spacer (between 2 and 3)
    1.4,    # Row 3
    0.5,    # Spacer (between 3 and 4)
    1.4,    # Row 4
])
fig3.subplots_adjust(left=0.075, right=0.925, top=0.98, bottom=0.05)

# --- Ax1: top-left plot (l_h)
ax1 = fig3.add_subplot(gs[0, 0])
ax1.set_xlabel('Model time [yrs]')
ax1.set_ylabel('$l_h$ [m]')
ax1.plot(tt, lh[lakepos, :], color='black', linestyle='-', label='$l_h$ (Lake Height)')
rect = matplotlib.patches.Rectangle(
    (6.9, 17),         # (x, y) bottom-left corner
    11.1 - 6.9,        # width
    59,              # height
    linewidth=1,
    edgecolor='black',
    facecolor='none',  # transparent fill
    linestyle='-'     # dashed outline (optional)
)
ax1.add_patch(rect)
ax1.scatter(tt[idx1], lh[lakepos, idx1], color='black', marker='o', s=8, label='c')
ax1.scatter(tt[idx2], lh[lakepos, idx2], color='black', marker='o', s=8, label='d')
ax1.scatter(tt[idx3], lh[lakepos, idx3], color='black', marker='o', s=8, label='e')
ax1.text(tt[idx1] + 0.2, lh[lakepos, idx1] - 2, 'c', fontsize=10)
ax1.text(tt[idx2] + 0.05, lh[lakepos, idx2] + 2, 'd', fontsize=10)
ax1.text(tt[idx3] + 0.2, lh[lakepos, idx3] - 2, 'e', fontsize=10)
ax1.set_ylim(15, 80)
ax1.set_xlim(0, 15)
ax1.margins(x=0, y=0)
ax1.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax1.xaxis.set_minor_locator(MultipleLocator(1))   

# --- Ax2: top-right plot (Qr)
ax2 = fig3.add_subplot(gs[0, 1])
ax2.set_xlabel('Model time [yrs]')
ax2.set_ylabel('$Q_r$ [m$^3$s$^-1$]',rotation=270, labelpad=15)  # Reduce padding between label and tick labels
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.plot(tt, Qr[lakepos, :], color='gray', linestyle='-', label='$Q_r$ (Lake Discharge)')
# Add rectangle bounding box to ax1
rect = matplotlib.patches.Rectangle(
    (6.9, -2),         # (x, y) bottom-left corner
    11.1 - 6.9,        # width
    54,              # height
    linewidth=1,
    edgecolor='gray',
    facecolor='none',  # transparent fill
    linestyle='-'     # dashed outline (optional)
)
ax2.add_patch(rect)
#ax2.text(9.1 + 0.1, 0.5 + 42, 'g', fontsize=10, ha='left', va='center', color='black')
ax2.scatter(tt[idx1],Qr[lakepos, idx1],color='gray',marker='o',s=8,label='c')
ax2.scatter(tt[idx2],Qr[lakepos, idx2],color='gray',marker='o',s=8,label='d')
ax2.scatter(tt[idx3],Qr[lakepos, idx3],color='gray',marker='o',s=8,label='e')
ax2.text(tt[idx1] + 0.05, Qr[lakepos, idx1] + 4, 'c', fontsize=10)
ax2.text(tt[idx2] + 0.2, Qr[lakepos, idx2] - 2.85, 'd', fontsize=10)
ax2.text(tt[idx3] - 0.25, Qr[lakepos, idx3] + 3, 'e', fontsize=10)

ax2.set_ylim(-5,55)
ax2.set_xlim(0, 15)
ax2.margins(x=0, y=0)
ax2.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax2.xaxis.set_minor_locator(MultipleLocator(1))   
# --- Ax3: second row plot (Q_c time i)
ax3 = fig3.add_subplot(gs[2, :])
ax3, sm1 = plotchannels(mesh,np.abs(Qc[:,idx1]),contours=True,phi=phi[:,idx1]/1e6,ax=ax3,min=0.5,quiver=False)
ax3.set_xlim([0, 10e3])
ax3.set_ylim([0, 3e3])
ax3.set_xticks([])  # Hide the x-axis ticks
ax3.set_xlabel('')  # Hide the x-axis label
ax3.set_yticks(ax3.get_yticks())
ax3.set_yticklabels([f"{y / 1e3:.0f}" for y in ax3.get_yticks()])
ax3.text(0.01, 0.9, f"{tt[idx1]:.2f} yrs", transform=plt.gca().transAxes,
    ha='left', va='center', fontsize=10)
divider1 = make_axes_locatable(ax3)
cax1 = divider1.append_axes("right", size="3%", pad=0.1)  # Reduce size and padding
cbar1 = fig3.colorbar(sm1, cax=cax1)



# --- Ax4: third row plot (Q_c time ii)
ax4 = fig3.add_subplot(gs[4, :])  # Extend ax4 across both columns
ax4, sm2 = plotchannels(mesh, np.abs(Qc[:, idx2]),contours=True,phi=phi[:,idx2]/1e6, ax=ax4, min=0.5, quiver=False)
ax4.set_xlim([0, 10e3])
ax4.set_ylim([0, 3e3])
ax4.text(0.01, 0.9, f"{tt[idx2]:.2f} yrs", transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=10)
ax4.set_xticks([])  # Hide the x-axis ticks
ax4.set_xlabel('')  # Hide the x-axis label
ax4.set_ylabel('Y [km]')  # Set the y-axis label
ax4.set_yticks(ax4.get_yticks())
ax4.set_yticklabels([f"{y / 1e3:.0f}" for y in ax4.get_yticks()])
# Adjust colorbar size
divider2 = make_axes_locatable(ax4)
cax2 = divider2.append_axes("right", size="3%", pad=0.1)  # Reduce size and padding
cbar2 = fig3.colorbar(sm2, cax=cax2)
cbar2.set_label('$Q_c$ [m$^3$s$^{-1}$]',rotation=270,labelpad=15)  # Rotate label and adjust padding

# --- Ax5: fourth row plot (Q_c time iii)
ax5 = fig3.add_subplot(gs[6, :])
ax5, sm3 = plotchannels(mesh, np.abs(Qc[:, idx3]),contours=True,phi=phi[:,idx3]/1e6, ax=ax5, min=0.5, quiver=False)
ax5.set_xlim([0, 10e3])
ax5.set_ylim([0, 3e3])
ax5.text(0.01, 0.9, f"{tt[idx3]:.2f} yrs", transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=10)

# Modify x and y tick labels
ax5.xaxis.set_minor_locator(MultipleLocator(1e3))  # Minor ticks every 2 km
ax5.set_xticks(ax5.get_xticks())
ax5.set_xticklabels([f"{x / 1e3:.0f}" for x in ax5.get_xticks()])
ax5.set_yticks(ax5.get_yticks())
ax5.set_yticklabels([f"{y / 1e3:.0f}" for y in ax5.get_yticks()])

ax5.set_xlabel('X [km]',labelpad=0.02)  # Set the x-axis label

divider3 = make_axes_locatable(ax5)
cax3 = divider3.append_axes("right", size="3%", pad=0.1)  # Reduce size and padding
cbar3 = fig3.colorbar(sm3, cax=cax3)


# --- Ax6: bottom-left plot (Qr vs N phase plane)

ax6 = fig3.add_subplot(gs[8,0])
QrLake = Qr[lakepos, :]
NLake = N[lakepos, :] / 1e6  # Convert to MPa
ax6.set_xlabel('$N$ [MPa]')
ax6.set_ylabel('$Q_r$ [m$^3$s$^{-1}$]')

# Plot line in the background
ax6.plot(NLake, QrLake, alpha=0.5, color='gray')
# Plot scatter points on top, using a greyscale colormap
scatter = ax6.scatter(NLake, QrLake, c=tt, cmap='Greys', s=15, edgecolor='none', zorder=10)

#set up divider for ax6
divider4 = make_axes_locatable(ax6)
cax4 = divider4.append_axes("top", size="5%", pad=0.05)
cbar4 = fig3.colorbar(scatter, cax = cax4, orientation='horizontal')
cax4.xaxis.set_ticks_position('top')
cax4.xaxis.set_label_position('top')
cbar4.set_label('Model time [yrs]')
formatter = matplotlib.ticker.FormatStrFormatter('%d')
cbar4.formatter = formatter
cbar4.set_ticks(np.arange(0, 20, 5))
cbar4.update_ticks() # Important: update the ticks after changing the formatter


# Add vertical dashed line for "Lake empty"
lake_empty_N = (9.81 * md.materials.rho_ice * md.geometry.thickness[lakepos]) / 1e6  # Convert to MPa

ax6.axvline(x=lake_empty_N, linestyle='--', color='black', label='Lake empty')

# Annotate "lake empty" aligned with lake_empty_N
text_str = '$l_h=0$ m'
text_obj = ax6.text(
    lake_empty_N,
    QrLake.min() + 0.75 * (QrLake.max() - QrLake.min()), 
    text_str,
    rotation=90,
    fontsize=8,
    verticalalignment='center',
    horizontalalignment='center',
    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1')
)


# Add horizontal dashed line for Qr = Qin
Qr_line = paramsdict['Q_in']
ax6.axhline(y=Qr_line, linestyle='--', color='black')
# Annotate "lake draining" and "lake filling"
mid_x = (NLake.min() + NLake.max()) / 2  # Calculate the middle of the x-axis span
ax6.text(mid_x, Qr_line + 0.05 * (QrLake.max() - QrLake.min()), 'lake draining', 
    verticalalignment='bottom', horizontalalignment='center',fontsize=8)
ax6.text(mid_x, Qr_line - 0.05 * (QrLake.max() - QrLake.min()), 'lake filling', 
     verticalalignment='top', horizontalalignment='center',fontsize=8)



num_points = len(NLake)
arrow_indices = [int(num_points * 0.05), int(num_points * 0.15), int(num_points * 0.25), 
                 int(num_points * 0.35), int(num_points * 0.45), int(num_points * 0.55), 
                 int(num_points * 0.65), int(num_points * 0.75), int(num_points * 0.85), 
                 int(num_points * 0.95)]

for i in arrow_indices:
    # Get the point for the arrow's head
    x_head, y_head = NLake[i], QrLake[i]
    # Get a preceding point for the arrow's tail to indicate direction
    # A small step-back (e.g., 5 points) makes the arrow clear
    step_back = 5
    if i > step_back:
        x_tail, y_tail = NLake[i - step_back], QrLake[i - step_back]
        ax6.annotate(
            '', # No text is needed for the arrow
            xy=(x_head, y_head),
            xytext=(x_tail, y_tail),
            arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
            zorder=20 # Ensure arrows are drawn on top
        )


ax6.set_xlim(NLake.min()-0.05, lake_empty_N+0.05)
ax6.set_ylim(-5, QrLake.max()+5)

ax6.margins(x=0, y=0)


# --- Ax7: bottom-right plot (ub vs x hovmoller)
"""
ax7 = fig3.add_subplot(gs[8,1])
ax7.set_xlabel('Model time [yrs]')
ax7.set_ylabel('X [km]', labelpad=0.05)  # Reduce padding between label and tick labels

# width average vel
[vel_avg, xedge] = width_average(mesh,vel,dx=0.15)

# compute time edges
dt = np.diff(tt).mean()
time_edge = np.concatenate(([tt[0] - dt / 2], tt + dt / 2))
[X,T] = np.meshgrid(xedge, time_edge)
pc = ax7.pcolormesh(T, X, np.transpose(vel_avg), cmap=cmocean.cm.deep_r, shading='flat', vmin=50,vmax=250)

# Add contour lines for velocity
contour_levels = np.arange(0, np.max(vel_avg)+10,10)
cs = ax7.contour(T[:-1, :-1], X[:-1, :-1], np.transpose(vel_avg), 
                 levels=contour_levels, colors='white', linewidths=0.15,)
levels_to_label = [200]  # Contour levels you want to label
#ax7.clabel(cs, levels=levels_to_label, inline=False, fontsize=5,
#           fmt=lambda x: f"${{{x:.0f}}}\\ \\mathrm{{ma}}^{{-1}}$")
"""


ax7 = fig3.add_subplot(gs[8, 1])
ax7.set_xlabel('Model time [yrs]')
ax7.xaxis.set_minor_locator(MultipleLocator(1))   
ax7.set_ylabel('X [km]', labelpad=0.001)  # Reduce padding between label and tick labels

# Width average velocity
[vel_avg, xedge] = width_average(mesh, vel, dx=0.15)

# Compute time edges
dt = np.diff(tt).mean()
time_edge = np.concatenate(([tt[0] - dt / 2], tt + dt / 2))
[X, T] = np.meshgrid(xedge, time_edge)

# Create filled contour plot
contour_levels = np.arange(50, 260, 10)  # Adjust to match vmin/vmax
cf = ax7.contourf(T[:-1, :-1], X[:-1, :-1], np.transpose(vel_avg),
                  levels=contour_levels, cmap=cmocean.cm.deep_r, vmin=50, vmax=230)

# Add white contour lines on top
cs = ax7.contour(T[:-1, :-1], X[:-1, :-1], np.transpose(vel_avg),
                 levels=contour_levels, colors='white', linewidths=0.15)

# add Qr
ax7r = ax7.twinx()
ax7r.plot(tt, QrLake, color='k', label='Qr',linewidth=1.5)
ax7r.set_ylabel('$Q_r$ [m$^3$ s$^{-1}$]', color='k',rotation=270,labelpad=15)  # Rotate label and adjust padding)
ax7r.tick_params(axis='y', labelcolor='k')

ax7.set_xlim(7, 11)
ax7.set_ylim(0.25, 10)  # Convert to km

# Set up divider for ax7
divider5 = make_axes_locatable(ax7)
divider6 = make_axes_locatable(ax7r)

# Add new Axes above ax7 for the colorbar
cax5 = divider5.append_axes("top", size="5%", pad=0.05)
cax6 = divider6.append_axes("top", size="5%", pad=0.05)

# Create horizontal colorbar in that new axis
cbar5 = fig3.colorbar(cf, cax=cax5, orientation='horizontal')
cbar6 = fig3.colorbar(cf, cax=cax6, orientation='horizontal')
cbar6.set_label('Ice velocity [m a$^{-1}$]')


# Move ticks and label to top
cax5.xaxis.set_ticks_position('top')
cax5.xaxis.set_label_position('top')
cax6.xaxis.set_ticks_position('top')
cax6.xaxis.set_label_position('top')

# Labels for subplots: (a) to (g)
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

# Custom (x, y) positions for each label in axis coordinates
label_positions = [
    (-0.14, 1.1),  # ax1
    (-0.06, 1.1),   # ax2
    (-0.1, 1.1),  # ax3
    (-0.1, 1.1),  # ax4
    (-0.1, 1.1),  # ax5
    (-0.14, 1.2),  # ax6
    (-0.17, 1.2)    # ax7
]

for ax, label, (x, y) in zip(axes, labels, label_positions):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=10,
        fontweight='bold'
    )


# Save figure
os.makedirs(os.path.join(filedir, 'figure3'), exist_ok=True)
fig3.savefig(os.path.join(filedir, 'figure3/figure3.png'), dpi=300)
fig3.savefig(os.path.join(filedir, 'figure3/figure3.pdf'), dpi=300)
fig3.savefig(os.path.join(filedir, 'figure3/figure3.eps'), dpi=300)
fig3.savefig(os.path.join(filedir, 'figure3/figure3.svg'), dpi=300)

############################################################

print('creating figure 4!')

fig4 = plt.figure(figsize=(7.04724, 8.75))  # J.Glac dual column figure spec
gs = gridspec.GridSpec(7, 3, figure=fig4, hspace=0.35, wspace=0.15, width_ratios=[0.4, 0.4, 0.2])
fig4.subplots_adjust(left=0.075, right=0.925, top=0.98, bottom=0.05)

# Load default results once
default_dir = '../experiments/RES/output_run_1_Default'
lh_default = np.load(os.path.join(default_dir, 'l_h.npy'))
Qr_default = np.load(os.path.join(default_dir, 'Qr.npy'))
tt_default = np.load(os.path.join(default_dir, 'tt.npy'))

# --- Ax1-2: Channel conductivity
HiKc_dir = '../experiments/RES/output_run_2_HiChK'
LoKc_dir = '../experiments/RES/output_run_3_LoChK'
lh_HiKc = np.load(os.path.join(HiKc_dir, 'l_h.npy'))
lh_LoKc = np.load(os.path.join(LoKc_dir, 'l_h.npy'))
Qr_HiKc = np.load(os.path.join(HiKc_dir, 'Qr.npy'))
Qr_LoKc = np.load(os.path.join(LoKc_dir, 'Qr.npy'))
tt_HiKc = np.load(os.path.join(HiKc_dir, 'tt.npy'))
tt_LoKc = np.load(os.path.join(LoKc_dir, 'tt.npy'))

ax1 = fig4.add_subplot(gs[0, 0])
ax1.plot(tt_default, lh_default[lakepos, :], color='gray', linestyle='-', label='Default')
ax1.plot(tt_HiKc, lh_HiKc[lakepos, :], color='red', linestyle='-', label='High $K_c$')
ax1.plot(tt_LoKc, lh_LoKc[lakepos, :], color='blue', linestyle='-', label='Low $K_c$')
ax1.set_xlim(0, 9)
ax1.set_ylim(0, 150)
ax1.margins(x=0, y=0)
ax1.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax1.xaxis.set_minor_locator(MultipleLocator(1))

ax2 = fig4.add_subplot(gs[0, 1])
ax2.plot(tt_default, Qr_default[lakepos, :], color='gray', linestyle='-', label='Default')
ax2.plot(tt_HiKc, Qr_HiKc[lakepos, :], color='red', linestyle='-', label='High $K_c$')
ax2.plot(tt_LoKc, Qr_LoKc[lakepos, :], color='blue', linestyle='-', label='Low $K_c$')
ax2.set_xlim(0, 9)
ax2.set_ylim(0, 80)
ax2.margins(x=0, y=0)
ax2.yaxis.tick_right()
ax2.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax2.xaxis.set_minor_locator(MultipleLocator(1))
ax2.text(1.275, 0.7, r"$k_c = 0.25\ \mathrm{m}^{3/2}\ \mathrm{kg}^{-1/2}$", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='red')
ax2.text(1.275, 0.5, r"$k_c = 0.1\ \mathrm{m}^{3/2}\ \mathrm{kg}^{-1/2}$", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='grey')
ax2.text(1.275, 0.3, r"$k_c = 0.01\ \mathrm{m}^{3/2}\ \mathrm{kg}^{-1/2}$", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='blue')
# --- Ax3-4: Sheet conductivity
HiKs_dir = '../experiments/RES/output_run_4_HiShK'
LoKs_dir = '../experiments/RES/output_run_5_LowShK'
lh_HiKs = np.load(os.path.join(HiKs_dir, 'l_h.npy'))
lh_LoKs = np.load(os.path.join(LoKs_dir, 'l_h.npy'))
Qr_HiKs = np.load(os.path.join(HiKs_dir, 'Qr.npy'))
Qr_LoKs = np.load(os.path.join(LoKs_dir, 'Qr.npy'))
tt_HiKs = np.load(os.path.join(HiKs_dir, 'tt.npy'))
tt_LoKs = np.load(os.path.join(LoKs_dir, 'tt.npy'))

ax3 = fig4.add_subplot(gs[1, 0])
ax3.plot(tt_default, lh_default[lakepos, :], color='gray', linestyle='-', label='Default')
ax3.plot(tt_HiKs, lh_HiKs[lakepos, :], color='red', linestyle='-', label='High $K_s$')
ax3.plot(tt_LoKs, lh_LoKs[lakepos, :], color='blue', linestyle='-', label='Low $K_s$')
ax3.set_xlim(0, 9)
ax3.set_ylim(0, 110)
ax3.margins(x=0, y=0)
ax3.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax3.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax3.xaxis.set_minor_locator(MultipleLocator(1))

ax4 = fig4.add_subplot(gs[1, 1])
ax4.plot(tt_default, Qr_default[lakepos, :], color='gray', linestyle='-', label='Default')
ax4.plot(tt_HiKs, Qr_HiKs[lakepos, :], color='red', linestyle='-', label='High $K_s$')
ax4.plot(tt_LoKs, Qr_LoKs[lakepos, :], color='blue', linestyle='-', label='Low $K_s$')
ax4.set_xlim(0, 9)
ax4.set_ylim(0, 100)
ax4.margins(x=0, y=0)
ax4.yaxis.tick_right()
ax4.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax4.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax4.xaxis.set_minor_locator(MultipleLocator(1))
ax4.text(1.275, 0.7, r"$k_s = 0.1\ \mathrm{Pa}\ \mathrm{s}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='red')
ax4.text(1.275, 0.5, r"$k_s = 0.02\ \mathrm{Pa}\ \mathrm{s}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='grey')
ax4.text(1.275, 0.3, r"$k_s = 0.01\ \mathrm{Pa}\ \mathrm{s}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='blue')

# --- Ax5-6: bump height
HiBh_dir = '../experiments/RES/output_run_8_HiBedBum'
LoBh_dir = '../experiments/RES/output_run_9_LoBedBum'
lh_HiBh = np.load(os.path.join(HiBh_dir, 'l_h.npy'))
lh_LoBh = np.load(os.path.join(LoBh_dir, 'l_h.npy'))
Qr_HiBh = np.load(os.path.join(HiBh_dir, 'Qr.npy'))
Qr_LoBh = np.load(os.path.join(LoBh_dir, 'Qr.npy'))
tt_HiBh = np.load(os.path.join(HiBh_dir, 'tt.npy'))
tt_LoBh = np.load(os.path.join(LoBh_dir, 'tt.npy'))

ax5 = fig4.add_subplot(gs[2, 0])
ax5.plot(tt_default, lh_default[lakepos, :], color='gray', linestyle='-', label='Default')
ax5.plot(tt_HiBh, lh_HiBh[lakepos, :], color='red', linestyle='-', label='High $B_h$')
ax5.plot(tt_LoBh, lh_LoBh[lakepos, :], color='blue', linestyle='-', label='Low $B_h$')
ax5.set_xlim(0, 9)
ax5.set_ylim(-2, 150)
ax5.margins(x=0, y=0)
ax5.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax5.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax5.xaxis.set_minor_locator(MultipleLocator(1))

ax6 = fig4.add_subplot(gs[2, 1])
ax6.plot(tt_default, Qr_default[lakepos, :], color='gray', linestyle='-', label='Default')
ax6.plot(tt_HiBh, Qr_HiBh[lakepos, :], color='red', linestyle='-', label='High $B_h$')
ax6.plot(tt_LoBh, Qr_LoBh[lakepos, :], color='blue', linestyle='-', label='Low $B_h$')
ax6.set_xlim(0, 9)
ax6.set_ylim(-2, 350)
ax6.margins(x=0, y=0)
ax6.yaxis.tick_right()
ax6.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax6.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax6.xaxis.set_minor_locator(MultipleLocator(1))

ax6.text(1.275, 0.7, r"$h_b = 1\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='red')
ax6.text(1.275, 0.5, r"$h_b = 0.1\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='grey')
ax6.text(1.275, 0.3, r"$h_b = 0.01\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='blue')

# --- Ax7-8: channel sheet width
HiLc_dir = '../experiments/RES/output_run_10_HiChWid'
LoLc_dir = '../experiments/RES/output_run_11_LoChWid'
lh_HiLc = np.load(os.path.join(HiLc_dir, 'l_h.npy'))
lh_LoLc = np.load(os.path.join(LoLc_dir, 'l_h.npy'))
Qr_HiLc = np.load(os.path.join(HiLc_dir, 'Qr.npy'))
Qr_LoLc = np.load(os.path.join(LoLc_dir, 'Qr.npy'))
tt_HiLc = np.load(os.path.join(HiLc_dir, 'tt.npy'))
tt_LoLc = np.load(os.path.join(LoLc_dir, 'tt.npy'))

ax7 = fig4.add_subplot(gs[3, 0])
ax7.plot(tt_default, lh_default[lakepos, :], color='gray', linestyle='-', label='Default')
ax7.plot(tt_HiLc, lh_HiLc[lakepos, :], color='red', linestyle='-', label='High $l_c$')
ax7.plot(tt_LoLc, lh_LoLc[lakepos, :], color='blue', linestyle='-', label='Low $l_c$')
ax7.set_xlim(0, 9)
ax7.set_ylim(-2, 110)
ax7.margins(x=0, y=0)
ax7.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax7.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax7.xaxis.set_minor_locator(MultipleLocator(1))
ax7.set_ylabel('$l_h$ [m]', labelpad=0.05)  # Reduce padding between label and tick labels

ax8 = fig4.add_subplot(gs[3, 1])
ax8.plot(tt_default, Qr_default[lakepos, :], color='gray', linestyle='-', label='Default')
ax8.plot(tt_HiLc, Qr_HiLc[lakepos, :], color='red', linestyle='-', label='High $l_c$')
ax8.plot(tt_LoLc, Qr_LoLc[lakepos, :], color='blue', linestyle='-', label='Low $l_c$')
ax8.set_xlim(0, 9)
ax8.set_ylim(-2, 70)
ax8.margins(x=0, y=0)
ax8.yaxis.tick_right()
ax8.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax8.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax8.xaxis.set_minor_locator(MultipleLocator(1))
ax8.set_ylabel('$Q_r$ [m$^3$s$^-1$]',rotation=270, labelpad=15)  # Reduce padding between label and tick labels
ax8.yaxis.set_label_position("right")

ax8.text(1.275, 0.7, r"$l_c = 50\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='red')
ax8.text(1.275, 0.5, r"$l_c = 20\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='grey')
ax8.text(1.275, 0.3, r"$l_c = 5\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='blue')

# --- Ax8-9: cavity spacing
HiLr_dir = '../experiments/RES/output_run_12_HiCavSp'
LoLr_dir = '../experiments/RES/output_run_13_LoCavSp'
lh_HiLr = np.load(os.path.join(HiLr_dir, 'l_h.npy'))
lh_LoLr = np.load(os.path.join(LoLr_dir, 'l_h.npy'))
Qr_HiLr = np.load(os.path.join(HiLr_dir, 'Qr.npy'))
Qr_LoLr = np.load(os.path.join(LoLr_dir, 'Qr.npy'))
tt_HiLr = np.load(os.path.join(HiLr_dir, 'tt.npy'))
tt_LoLr = np.load(os.path.join(LoLr_dir, 'tt.npy'))

ax9 = fig4.add_subplot(gs[4, 0])
ax9.plot(tt_default, lh_default[lakepos, :], color='gray', linestyle='-', label='Default')
ax9.plot(tt_HiLr, lh_HiLr[lakepos, :], color='red', linestyle='-', label='High $l_r$')
ax9.plot(tt_LoLr, lh_LoLr[lakepos, :], color='blue', linestyle='-', label='Low $l_r$')
ax9.set_xlim(0, 9)
ax9.set_ylim(-2, 110)
ax9.margins(x=0, y=0)
ax9.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax9.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax9.xaxis.set_minor_locator(MultipleLocator(1))

ax10 = fig4.add_subplot(gs[4, 1])
ax10.plot(tt_default, Qr_default[lakepos, :], color='gray', linestyle='-', label='Default')
ax10.plot(tt_HiLr, Qr_HiLr[lakepos, :], color='red', linestyle='-', label='High $l_r$')
ax10.plot(tt_LoLr, Qr_LoLr[lakepos, :], color='blue', linestyle='-', label='Low $l_r$')
ax10.set_xlim(0, 9)
ax10.set_ylim(-2, 70)
ax10.margins(x=0, y=0)
ax10.yaxis.tick_right()
ax10.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax10.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax10.xaxis.set_minor_locator(MultipleLocator(1))

ax10.text(1.275, 0.7, r"$l_r = 10\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='red')
ax10.text(1.275, 0.5, r"$l_r = 5\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='grey')
ax10.text(1.275, 0.3, r"$l_r = 5\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='blue')

# --- Ax8-9: melt rate
HiMr_dir = '../experiments/RES/output_run_22_HiMelt'
LoMr_dir = '../experiments/RES/output_run_23_LoMelt'
lh_HiMr = np.load(os.path.join(HiMr_dir, 'l_h.npy'))
lh_LoMr = np.load(os.path.join(LoMr_dir, 'l_h.npy'))
Qr_HiMr = np.load(os.path.join(HiMr_dir, 'Qr.npy'))
Qr_LoMr = np.load(os.path.join(LoMr_dir, 'Qr.npy'))
tt_HiMr = np.load(os.path.join(HiMr_dir, 'tt.npy'))
tt_LoMr = np.load(os.path.join(LoMr_dir, 'tt.npy'))

ax11 = fig4.add_subplot(gs[5, 0])
ax11.plot(tt_default, lh_default[lakepos, :], color='gray', linestyle='-', label='Default')
ax11.plot(tt_HiMr, lh_HiMr[lakepos, :], color='red', linestyle='-', label='High $M_r$')
ax11.plot(tt_LoMr, lh_LoMr[lakepos, :], color='blue', linestyle='-', label='Low $M_r$')
ax11.set_xlim(0, 9)
ax11.set_ylim(-2, 110)
ax11.margins(x=0, y=0)
ax11.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax11.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax11.xaxis.set_minor_locator(MultipleLocator(1))

ax12 = fig4.add_subplot(gs[5, 1])
ax12.plot(tt_default, Qr_default[lakepos, :], color='gray', linestyle='-', label='Default')
ax12.plot(tt_HiMr, Qr_HiMr[lakepos, :], color='red', linestyle='-', label='High $M_r$')
ax12.plot(tt_LoMr, Qr_LoMr[lakepos, :], color='blue', linestyle='-', label='Low $M_r$')
ax12.set_xlim(0, 9)
ax12.set_ylim(-2, 70)
ax12.margins(x=0, y=0)
ax12.yaxis.tick_right()
ax12.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax12.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax12.xaxis.set_minor_locator(MultipleLocator(1))

ax12.text(1.275, 0.7, r"$M_r = 0.5\ \mathrm{m}\ \mathrm{a}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='red')
ax12.text(1.275, 0.5, r"$M_r = 0.25\ \mathrm{m}\ \mathrm{a}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='grey')
ax12.text(1.275, 0.3, r"$M_r = 0.125\ \mathrm{m}\ \mathrm{a}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='blue')

# --- Ax10-11: melt rate
HiQin_dir = '../experiments/RES/output_run_24_HiQin'
LoQin_dir = '../experiments/RES/output_run_25_LoQin'
lh_HiQin = np.load(os.path.join(HiQin_dir, 'l_h.npy'))
lh_LoQin = np.load(os.path.join(LoQin_dir, 'l_h.npy'))
Qr_HiQin = np.load(os.path.join(HiQin_dir, 'Qr.npy'))
Qr_LoQin = np.load(os.path.join(LoQin_dir, 'Qr.npy'))
tt_HiQin = np.load(os.path.join(HiQin_dir, 'tt.npy'))
tt_LoQin = np.load(os.path.join(LoQin_dir, 'tt.npy'))

ax13 = fig4.add_subplot(gs[6, 0])
ax13.plot(tt_default, lh_default[lakepos, :], color='gray', linestyle='-', label='Default')
ax13.plot(tt_HiQin, lh_HiQin[lakepos, :], color='red', linestyle='-', label='High $Q_in$')
ax13.plot(tt_LoQin, lh_LoQin[lakepos, :], color='blue', linestyle='-', label='Low $Q_in$')
ax13.set_xlim(0, 9)
ax13.set_ylim(-2, 110)
ax13.margins(x=0, y=0)
ax13.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax13.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax13.xaxis.set_minor_locator(MultipleLocator(1))
ax13.set_xlabel('Model time [yrs]')

ax14 = fig4.add_subplot(gs[6, 1])
ax14.plot(tt_default, Qr_default[lakepos, :], color='gray', linestyle='-', label='Default')
ax14.plot(tt_HiQin, Qr_HiQin[lakepos, :], color='red', linestyle='-', label='High $Q_in$')
ax14.plot(tt_LoQin, Qr_LoQin[lakepos, :], color='blue', linestyle='-', label='Low $Q_in$')
ax14.set_xlim(0, 9)
ax14.set_ylim(-2, 70)
ax14.margins(x=0, y=0)
ax14.yaxis.tick_right()
ax14.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax14.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax14.xaxis.set_minor_locator(MultipleLocator(1))
ax14.set_xlabel('Model time [yrs]')

ax14.text(1.275, 0.7, r"$Q_{in} = 10\ \mathrm{m}^{3}\ \mathrm{s}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='red')
ax14.text(1.275, 0.5, r"$Q_{in} = 10\ \mathrm{m}^{3}\ \mathrm{s}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='grey')
ax14.text(1.275, 0.3, r"$Q_{in} = 0.5\ \mathrm{m}^{3}\ \mathrm{s}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=9,color='blue')


# Labels for subplots: (a) to (g)
labels = ['a','i', 'ii', 'b','i','ii', 'c','i','ii', 'd','i','ii', 'e','i','ii','f','i','ii', 'g','i','ii']
axes = [ax1,ax1,ax2, ax3, ax3, ax4, ax5, ax5, ax6, ax7, ax7, ax8, ax9, ax9, ax10, ax11, ax11, ax12, ax13, ax13, ax14]

# Custom (x, y) positions for each label in axis coordinates
label_positions = [
    (-0.23, 1.1),  # ax1
    (0.02, 0.95),  # ax1
    (0.02, 0.95),  # ax2
    (-0.23, 1.1),  # ax3
    (0.02, 0.95),  # ax3
    (0.02, 0.95),  # ax4
    (-0.23, 1.1),  # ax5
    (0.02, 0.95),  # ax5
    (0.02, 0.95),  # ax6
    (-0.23, 1.1),  # ax7
    (0.02, 0.95),  # ax7
    (0.02, 0.95),  # ax8
    (-0.23, 1.1),  # ax9
    (0.02, 0.95),  # ax9
    (0.02, 0.95),  # ax10
    (-0.23, 1.1),  # ax11
    (0.02, 0.95),  # ax11
    (0.02, 0.95),  # ax12
    (-0.23, 1.1),  # ax13
    (0.02, 0.95),   # ax13
    (0.02, 0.95)   # ax14
]

for ax, label, (x, y) in zip(axes, labels, label_positions):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=10,
        fontweight='bold'
    )


# Save figure
os.makedirs(os.path.join(filedir, 'figure4'), exist_ok=True)
fig4.savefig(os.path.join(filedir, 'figure4/figure4.png'), dpi=300)
fig4.savefig(os.path.join(filedir, 'figure4/figure4.pdf'), dpi=300)
fig4.savefig(os.path.join(filedir, 'figure4/figure4.eps'), dpi=300)
fig4.savefig(os.path.join(filedir, 'figure4/figure4.svg'), dpi=300)

############################################################

print('creating figure 5!')

fig5 = plt.figure(figsize=(7.04724, 5))  # J.Glac dual column figure spec
gs = gridspec.GridSpec(2, 3, figure=fig5, hspace=0.65, wspace=0.45)
fig5.subplots_adjust(left=0.085, right=0.905, top=0.86, bottom=0.1)

# ax1,ax2 = channel conductivity
Vel_HiKc = np.load(os.path.join(HiKc_dir, 'vel.npy'))
Vel_LoKc = np.load(os.path.join(LoKc_dir, 'vel.npy'))

ax1 = fig5.add_subplot(gs[0, 0])
ax1.set_ylabel('X [km]', labelpad=0.001)
#width average velocity
[vel_avg_HiKc, xedge_HiKc] = width_average(mesh, Vel_HiKc, dx=0.15)

# compute time edges
dt_HiKc = np.diff(tt_HiKc).mean()
time_edge_HiKc = np.concatenate(([tt_HiKc[0] - dt_HiKc / 2], tt_HiKc + dt_HiKc / 2))
[X_HiKc, T_HiKc] = np.meshgrid(xedge_HiKc, time_edge_HiKc)

# Create filled contour plot
maxvel_HiKc = round(np.max(vel_avg_HiKc[:, -500:]) / 10) * 10
minvel_HiKc = round(np.min(vel_avg_HiKc[:, -500:]) / 10) * 10

#print(f'Maximum velocity in LoKc: {maxvel_LoKc} m a^-1')
#print(f'Minimum velocity in LoKc: {minvel_LoKc} m a^-1')

contour_levels_HiKc = np.arange(minvel_HiKc, maxvel_HiKc+10, 10)  # Adjust to match vmin/vmax
cf_HiKc = ax1.contourf(T_HiKc[:-1, :-1], X_HiKc[:-1, :-1], np.transpose(vel_avg_HiKc),
                      levels=contour_levels_HiKc, cmap=cmocean.cm.deep_r)
# Add white contour lines on top        
cs_HiKc = ax1.contour(T_HiKc[:-1, :-1], X_HiKc[:-1, :-1], np.transpose(vel_avg_HiKc),
                     levels=contour_levels_HiKc, colors='white', linewidths=0.15)

# Add Qr
ax1r = ax1.twinx()
ax1r.plot(tt_HiKc, Qr_HiKc[lakepos, :], color='k', label='Qr', linewidth=1.5)
#ax1r.set_ylabel('$Q_r$ [m$^3$ s$^{-1}$]', color='k', rotation=270, labelpad=15)  # Rotate label and adjust padding
ax1r.tick_params(axis='y', labelcolor='k')
ax1.set_xlim(7, 11)
ax1.set_ylim(0.25, 10)  # Convert to km

# Set up divider for ax1
divider1 = make_axes_locatable(ax1)
divider2 = make_axes_locatable(ax1r)
# Add new Axes above ax1 for the colorbar
cax1 = divider1.append_axes("top", size="5%", pad=0.05)
cax2 = divider2.append_axes("top", size="5%", pad=0.05)
# Create horizontal colorbar in that new axis
cbar1 = fig5.colorbar(cf_HiKc, cax=cax1, orientation='horizontal')
cbar2 = fig5.colorbar(cf_HiKc, cax=cax2, orientation='horizontal')
cbar2.set_label('Ice velocity [m a$^{-1}$]')
# Move ticks and label to top
cax1.xaxis.set_ticks_position('top')
cax1.xaxis.set_label_position('top')
cax2.xaxis.set_ticks_position('top')
cax2.xaxis.set_label_position('top')
# Set tick labels every 100 units based on colorbar limits
vmin, vmax = cf_HiKc.get_clim()  # Get the color limits of the plot
ticks = np.arange(np.ceil(vmin / 100) * 100, vmax + 1, 100)

cbar1.set_ticks(ticks)
cbar2.set_ticks(ticks)

ax1.text(0.5, 9.5, r"$k_c = 0.25\ \mathrm{m}^{3/2}\ \mathrm{kg}^{-1/2}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='black')

ax2 = fig5.add_subplot(gs[1, 0])
ax2.set_ylabel('X [km]', labelpad=0.001)
#width average velocity
[vel_avg_LoKc, xedge_LoKc] = width_average(mesh, Vel_LoKc, dx=0.15)

# compute time edges
dt_LoKc = np.diff(tt_LoKc).mean()
time_edge_LoKc = np.concatenate(([tt_LoKc[0] - dt_LoKc / 2], tt_LoKc + dt_LoKc / 2))
[X_LoKc, T_LoKc] = np.meshgrid(xedge_LoKc, time_edge_LoKc)

# Create filled contour plot
maxvel_LoKc = round(np.max(vel_avg_LoKc[:, -500:]) / 10) * 10
minvel_LoKc = round(np.min(vel_avg_LoKc[:, -500:]) / 10) * 10

#print(f'Maximum velocity in LoKc: {maxvel_LoKc} m a^-1')
#print(f'Minimum velocity in LoKc: {minvel_LoKc} m a^-1')

contour_levels_LoKc = np.arange(minvel_LoKc, maxvel_LoKc+10, 10)  # Adjust to match vmin/vmax
cf_LoKc = ax2.contourf(T_LoKc[:-1, :-1], X_LoKc[:-1, :-1], np.transpose(vel_avg_LoKc),
                      levels=contour_levels_LoKc, cmap=cmocean.cm.deep_r)
# Add white contour lines on top        
cs_LoKc = ax2.contour(T_LoKc[:-1, :-1], X_LoKc[:-1, :-1], np.transpose(vel_avg_LoKc),
                     levels=contour_levels_LoKc, colors='white', linewidths=0.15)

# Add Qr
ax2r = ax2.twinx()
ax2r.plot(tt_LoKc, Qr_LoKc[lakepos, :], color='k', label='Qr', linewidth=1.5)
#ax2r.set_ylabel('$Q_r$ [m$^3$ s$^{-1}$]', color='k', rotation=270, labelpad=15)  # Rotate label and adjust padding
ax2r.tick_params(axis='y', labelcolor='k')
ax2.set_xlim(7, 11)
ax2.set_ylim(0.25, 10)  # Convert to km
ax2.set_xlabel('Model time [yrs]')

# Set up divider for ax1
divider3 = make_axes_locatable(ax2)
divider4 = make_axes_locatable(ax2r)
# Add new Axes above ax1 for the colorbar
cax3 = divider3.append_axes("top", size="5%", pad=0.05)
cax4 = divider4.append_axes("top", size="5%", pad=0.05)
# Create horizontal colorbar in that new axis
cbar3 = fig5.colorbar(cf_LoKc, cax=cax3, orientation='horizontal')
cbar4 = fig5.colorbar(cf_LoKc, cax=cax4, orientation='horizontal')
cbar4.set_label('Ice velocity [m a$^{-1}$]')
# Move ticks and label to top
cax3.xaxis.set_ticks_position('top')
cax3.xaxis.set_label_position('top')
cax4.xaxis.set_ticks_position('top')
cax4.xaxis.set_label_position('top')
# Set tick labels every 100 units based on colorbar limits
vmin, vmax = cf_LoKc.get_clim()  # Get the color limits of the plot
ticks = np.arange(np.ceil(vmin / 100) * 100, vmax + 1, 100)

cbar3.set_ticks(ticks)
cbar4.set_ticks(ticks)

ax2.text(0.5, 9.5, r"$k_c = 0.01\ \mathrm{m}^{3/2}\ \mathrm{kg}^{-1/2}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='black')

# ax3 and ax4: sheet conductivity

Vel_HiKs = np.load(os.path.join(HiKs_dir, 'vel.npy'))
Vel_LoKs = np.load(os.path.join(LoKs_dir, 'vel.npy'))

ax3 = fig5.add_subplot(gs[0, 1])
#ax3.set_ylabel('X [km]', labelpad=0.001)
#width average velocity
[vel_avg_HiKs, xedge_HiKs] = width_average(mesh, Vel_HiKs, dx=0.15)

# compute time edges
dt_HiKs = np.diff(tt_HiKs).mean()
time_edge_HiKs = np.concatenate(([tt_HiKs[0] - dt_HiKs / 2], tt_HiKs + dt_HiKs / 2))
[X_HiKs, T_HiKs] = np.meshgrid(xedge_HiKs, time_edge_HiKs)

# Create filled contour plot
maxvel_HiKs = round(np.max(vel_avg_HiKs[:, -500:]) / 10) * 10
minvel_HiKs = round(np.min(vel_avg_HiKs[:, -500:]) / 10) * 10


contour_levels_HiKs = np.arange(minvel_HiKs, maxvel_HiKs+20, 10)  # Adjust to match vmin/vmax
cf_HiKs = ax3.contourf(T_HiKs[:-1, :-1], X_HiKs[:-1, :-1], np.transpose(vel_avg_HiKs),
                      levels=contour_levels_HiKs, cmap=cmocean.cm.deep_r)
# Add white contour lines on top        
cs_HiKs = ax3.contour(T_HiKs[:-1, :-1], X_HiKs[:-1, :-1], np.transpose(vel_avg_HiKs),
                     levels=contour_levels_HiKs, colors='white', linewidths=0.15)

# Add Qr
ax3r = ax3.twinx()
ax3r.plot(tt_HiKs, Qr_HiKs[lakepos, :], color='k', label='Qr', linewidth=1.5)
#a31r.set_ylabel('$Q_r$ [m$^3$ s$^{-1}$]', color='k', rotation=270, labelpad=15)  # Rotate label and adjust padding
ax3r.tick_params(axis='y', labelcolor='k')
ax3.set_xlim(7, 11)
ax3.set_ylim(0.25, 10)  # Convert to km

# Set up divider for ax3
divider3 = make_axes_locatable(ax3)
divider4 = make_axes_locatable(ax3r)
# Add new Axes above ax3 for the colorbar
cax5 = divider3.append_axes("top", size="5%", pad=0.05)
cax6 = divider4.append_axes("top", size="5%", pad=0.05)
# Create horizontal colorbar in that new axis
cbar5 = fig5.colorbar(cf_HiKs, cax=cax5, orientation='horizontal')
cbar6 = fig5.colorbar(cf_HiKs, cax=cax6, orientation='horizontal')
cbar6.set_label('Ice velocity [m a$^{-1}$]')
# Move ticks and label to top
cax5.xaxis.set_ticks_position('top')
cax5.xaxis.set_label_position('top')
cax6.xaxis.set_ticks_position('top')
cax6.xaxis.set_label_position('top')
# Set tick labels every 100 units based on colorbar limits
vmin, vmax = cf_HiKs.get_clim()  # Get the color limits of the plot
ticks = np.arange(np.ceil(vmin / 100) * 100, vmax + 1, 100)

cbar5.set_ticks(ticks)
cbar6.set_ticks(ticks)

ax3.text(0.5, 9.5, r"$k_s = 0.1\ \mathrm{Pa}\ \mathrm{s}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='black')

ax4 = fig5.add_subplot(gs[1, 1])
#width average velocity
[vel_avg_LoKs, xedge_LoKs] = width_average(mesh, Vel_LoKs, dx=0.15)

# compute time edges
dt_LoKs = np.diff(tt_LoKs).mean()
time_edge_LoKs = np.concatenate(([tt_LoKs[0] - dt_LoKs / 2], tt_LoKs + dt_LoKs / 2))
[X_LoKs, T_LoKs] = np.meshgrid(xedge_LoKs, time_edge_LoKs)

# Create filled contour plot
maxvel_LoKs = round(np.max(vel_avg_LoKs[:, -500:]) / 10) * 10
minvel_LoKs = round(np.min(vel_avg_LoKs[:, -500:]) / 10) * 10

contour_levels_LoKs = np.arange(minvel_LoKs, maxvel_LoKs+10, 10)  # Adjust to match vmin/vmax
cf_LoKs = ax4.contourf(T_LoKs[:-1, :-1], X_LoKs[:-1, :-1], np.transpose(vel_avg_LoKs),
                      levels=contour_levels_LoKs, cmap=cmocean.cm.deep_r)
# Add white contour lines on top        
cs_LoKs = ax4.contour(T_LoKs[:-1, :-1], X_LoKs[:-1, :-1], np.transpose(vel_avg_LoKs),
                     levels=contour_levels_LoKs, colors='white', linewidths=0.15)

# Add Qr
ax4r = ax4.twinx()
ax4r.plot(tt_LoKs, Qr_LoKs[lakepos, :], color='k', label='Qr', linewidth=1.5)
#a42r.set_ylabel('$Q_r$ [m$^3$ s$^{-1}$]', color='k', rotation=270, labelpad=15)  # Rotate label and adjust padding
ax4r.tick_params(axis='y', labelcolor='k')
ax4.set_xlim(7, 11)
ax4.set_ylim(0.25, 10)  # Convert to km
ax4.set_xlabel('Model time [yrs]')

# Set up divider for ax1
divider7 = make_axes_locatable(ax4)
divider8 = make_axes_locatable(ax4r)
# Add new Axes above ax1 for the colorbar
cax7 = divider7.append_axes("top", size="5%", pad=0.05)
cax8 = divider8.append_axes("top", size="5%", pad=0.05)
# Create horizontal colorbar in that new axis
cbar7 = fig5.colorbar(cf_LoKs, cax=cax7, orientation='horizontal')
cbar8 = fig5.colorbar(cf_LoKs, cax=cax8, orientation='horizontal')
cbar8.set_label('Ice velocity [m a$^{-1}$]')
# Move ticks and label to top
cax7.xaxis.set_ticks_position('top')
cax7.xaxis.set_label_position('top')
cax8.xaxis.set_ticks_position('top')
cax8.xaxis.set_label_position('top')
# Set tick labels every 100 units based on colorbar limits
vmin, vmax = cf_LoKs.get_clim()  # Get the color limits of the plot
ticks = np.arange(np.ceil(vmin / 100) * 100, vmax + 1, 100)

cbar7.set_ticks(ticks)
cbar8.set_ticks(ticks)

ax4.text(0.5, 9.5, r"$k_s = 0.01\ \mathrm{Pa}\ \mathrm{s}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='black')
         #ha='center', va='center', fontsize=9,color='black',bbox=dict(edgecolor='blue', alpha=1, facecolor='white'))


# ax5 and ax6: bed bump height

Vel_HiBh = np.load(os.path.join(HiBh_dir, 'vel.npy'))
Vel_LoBh = np.load(os.path.join(LoBh_dir, 'vel.npy'))

ax5 = fig5.add_subplot(gs[0, 2])
#ax3.set_ylabel('X [km]', labelpad=0.001)
#width average velocity
[vel_avg_HiBh, xedge_HiBh] = width_average(mesh, Vel_HiBh, dx=0.15)

# compute time edges
dt_HiBh = np.diff(tt_HiBh).mean()
time_edge_HiBh = np.concatenate(([tt_HiBh[0] - dt_HiBh / 2], tt_HiBh + dt_HiBh / 2))
[X_HiBh, T_HiBh] = np.meshgrid(xedge_HiBh, time_edge_HiBh)

# Create filled contour plot
maxvel_HiBh = round(np.max(vel_avg_HiBh[:, -500:]) / 10) * 10
minvel_HiBh = round(np.min(vel_avg_HiBh[:, -500:]) / 10) * 10


contour_levels_HiBh = np.arange(minvel_HiBh, maxvel_HiBh+20, 10)  # Adjust to match vmin/vmax
cf_HiBh = ax5.contourf(T_HiBh[:-1, :-1], X_HiBh[:-1, :-1], np.transpose(vel_avg_HiBh),
                      levels=contour_levels_HiBh, cmap=cmocean.cm.deep_r)
# Add white contour lines on top        
cs_HiBh = ax5.contour(T_HiBh[:-1, :-1], X_HiBh[:-1, :-1], np.transpose(vel_avg_HiBh),
                     levels=contour_levels_HiBh, colors='white', linewidths=0.15)

# Add Qr
ax5r = ax5.twinx()
ax5r.plot(tt_HiBh, Qr_HiBh[lakepos, :], color='k', label='Qr', linewidth=1.5)
ax5r.set_ylabel('$Q_r$ [m$^3$ s$^{-1}$]', color='k', rotation=270, labelpad=15)  # Rotate label and adjust padding
ax5r.tick_params(axis='y', labelcolor='k')
ax5.set_xlim(7, 11)
ax5.set_ylim(0.25, 10)  # Convert to km

# Set up divider for ax3
divider9 = make_axes_locatable(ax5)
divider10 = make_axes_locatable(ax5r)
# Add new Axes above ax3 for the colorbar
cax9 = divider9.append_axes("top", size="5%", pad=0.05)
cax10 = divider10.append_axes("top", size="5%", pad=0.05)
# Create horizontal colorbar in that new axis
cbar9 = fig5.colorbar(cf_HiBh, cax=cax9, orientation='horizontal')
cbar10 = fig5.colorbar(cf_HiBh, cax=cax10, orientation='horizontal')
cbar10.set_label('Ice velocity [m a$^{-1}$]')
# Move ticks and label to top
cax9.xaxis.set_ticks_position('top')
cax9.xaxis.set_label_position('top')
cax10.xaxis.set_ticks_position('top')
cax10.xaxis.set_label_position('top')
# Set tick labels every 100 units based on colorbar limits
vmin, vmax = cf_HiBh.get_clim()  # Get the color limits of the plot
ticks = np.arange(np.ceil(vmin / 100) * 100, vmax + 1, 100)

cbar9.set_ticks(ticks)
cbar10.set_ticks(ticks)

ax5.text(0.5, 9.5, r"$h_b = 1\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='black')

ax6 = fig5.add_subplot(gs[1, 2])
#width average velocity
[vel_avg_LoBh, xedge_LoBh] = width_average(mesh, Vel_LoBh, dx=0.15)

# compute time edges
dt_LoBh = np.diff(tt_LoBh).mean()
time_edge_LoBh = np.concatenate(([tt_LoBh[0] - dt_LoBh / 2], tt_LoBh + dt_LoBh / 2))
[X_LoBh, T_LoBh] = np.meshgrid(xedge_LoBh, time_edge_LoBh)

# Create filled contour plot
maxvel_LoBh = round(np.max(vel_avg_LoBh[:, -500:]) / 10) * 10
minvel_LoBh = round(np.min(vel_avg_LoBh[:, -500:]) / 10) * 10

contour_levels_LoBh = np.arange(minvel_LoBh, maxvel_LoBh+10, 10)  # Adjust to match vmin/vmax
cf_LoBh = ax6.contourf(T_LoBh[:-1, :-1], X_LoBh[:-1, :-1], np.transpose(vel_avg_LoBh),
                      levels=contour_levels_LoBh, cmap=cmocean.cm.deep_r)
# Add white contour lines on top        
cs_LoBh = ax6.contour(T_LoBh[:-1, :-1], X_LoBh[:-1, :-1], np.transpose(vel_avg_LoBh),
                     levels=contour_levels_LoBh, colors='white', linewidths=0.15)

# Add Qr
ax6r = ax6.twinx()
ax6r.plot(tt_LoBh, Qr_LoBh[lakepos, :], color='k', label='Qr', linewidth=1.5)
ax6r.set_ylabel('$Q_r$ [m$^3$ s$^{-1}$]', color='k', rotation=270, labelpad=15)  # Rotate label and adjust padding
ax6r.tick_params(axis='y', labelcolor='k')
ax6.set_xlim(7, 11)
ax6.set_ylim(0.25, 10)  # Convert to km
ax6.set_xlabel('Model time [yrs]')

# Set up divider for ax1
divider11 = make_axes_locatable(ax6)
divider12 = make_axes_locatable(ax6r)
# Add new Axes above ax1 for the colorbar
cax11 = divider11.append_axes("top", size="5%", pad=0.05)
cax12 = divider12.append_axes("top", size="5%", pad=0.05)
# Create horizontal colorbar in that new axis
cbar11 = fig5.colorbar(cf_LoBh, cax=cax11, orientation='horizontal')
cbar12 = fig5.colorbar(cf_LoBh, cax=cax12, orientation='horizontal')
cbar12.set_label('Ice velocity [m a$^{-1}$]')
# Move ticks and label to top
cax11.xaxis.set_ticks_position('top')
cax11.xaxis.set_label_position('top')
cax12.xaxis.set_ticks_position('top')
cax12.xaxis.set_label_position('top')
# Set tick labels every 100 units based on colorbar limits
vmin, vmax = cf_LoBh.get_clim()  # Get the color limits of the plot
ticks = np.arange(np.ceil(vmin / 100) * 100, vmax + 1, 100)

cbar11.set_ticks(ticks)
cbar12.set_ticks(ticks)

ax6.text(0.5, 9.5, r"$h_b = 0.01\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='black')
         #ha='center', va='center', fontsize=9,color='black',bbox=dict(edgecolor='blue', alpha=1, facecolor='white'))
 # ax3 and ax4: sheet conductivity

 # Labels for subplots: (a) to (g)
labels = ['a', 'b', 'c', 'd', 'e','f']
axes = [ax1, ax2, ax3, ax4, ax5, ax6]

# Custom (x, y) positions for each label in axis coordinates
label_positions = [
    (-0.3, 1.5),  # ax1
    (-0.3, 1.5),   # ax2
    (-0.3, 1.5), # ax3
    (-0.3, 1.5), # ax4
    (-0.3, 1.5), # ax5
    (-0.3, 1.5),  # ax6
]

for ax, label, (x, y) in zip(axes, labels, label_positions):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=10,
        fontweight='bold'
    )
 


# Save figure
os.makedirs(os.path.join(filedir, 'figure5'), exist_ok=True)
fig5.savefig(os.path.join(filedir, 'figure5/figure5.png'), dpi=300)
fig5.savefig(os.path.join(filedir, 'figure5/figure5.pdf'), dpi=300)
fig5.savefig(os.path.join(filedir, 'figure5/figure5.eps'), dpi=300)
fig5.savefig(os.path.join(filedir, 'figure5/figure5.svg'), dpi=300)