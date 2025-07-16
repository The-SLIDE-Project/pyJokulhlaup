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

fig = plt.figure(figsize=(7.04724, 10))  # J.Glac dual column figure spec
gs = gridspec.GridSpec(9, 2, figure=fig, hspace=0.15, wspace=0.2, height_ratios=[
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
fig.subplots_adjust(left=0.1, right=0.9, top=0.98, bottom=0.05)

# --- Ax1: top-left plot (l_h)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlabel('Model time [yrs]')
ax1.set_ylabel('$l_h$ [m]')
ax1.plot(tt, lh[lakepos, :], color='black', linestyle='-', label='$l_h$ (Lake Height)')
ax1.scatter(tt[idx1], lh[lakepos, idx1], color='black', marker='o', s=8, label='c')
ax1.scatter(tt[idx2], lh[lakepos, idx2], color='black', marker='o', s=8, label='d')
ax1.scatter(tt[idx3], lh[lakepos, idx3], color='black', marker='o', s=8, label='e')
ax1.text(tt[idx1] + 0.1, lh[lakepos, idx1] - 4, 'c', fontsize=10)
ax1.text(tt[idx2] + 0.05, lh[lakepos, idx2] + 2, 'd', fontsize=10)
ax1.text(tt[idx3] + 0.05, lh[lakepos, idx3] + 2, 'e', fontsize=10)
ax1.set_ylim(20, 80)
ax1.set_xlim(0, 10)
ax1.margins(x=0, y=0)

# --- Ax2: top-right plot (Qr)
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlabel('Model time [yrs]')
ax2.set_ylabel('$Q_r$ [m$^3$s$^-1$]')
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.plot(tt, Qr[lakepos, :], color='gray', linestyle='-', label='$Q_r$ (Lake Discharge)')
# Add rectangle bounding box to ax1
rect = matplotlib.patches.Rectangle(
    (6.9, 0.5),         # (x, y) bottom-left corner
    9.1 - 6.9,        # width
    42,              # height
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
ax2.text(tt[idx1] + 0.05, Qr[lakepos, idx1] + 2, 'c', fontsize=10)
ax2.text(tt[idx2] + 0.1, Qr[lakepos, idx2] - 3.85, 'd', fontsize=10)
ax2.text(tt[idx3] - 0.25, Qr[lakepos, idx3] + 2, 'e', fontsize=10)

ax2.set_ylim(0,45)
ax2.set_xlim(0, 10)
ax2.margins(x=0, y=0)
# --- Ax3: second row plot (Q_c time i)
ax3 = fig.add_subplot(gs[2, :])
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
cbar1 = fig.colorbar(sm1, cax=cax1)



# --- Ax4: third row plot (Q_c time ii)
ax4 = fig.add_subplot(gs[4, :])  # Extend ax4 across both columns
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
cbar2 = fig.colorbar(sm2, cax=cax2)
cbar2.set_label('$Q_c$ [m$^3$s$^{-1}$]')

# --- Ax5: fourth row plot (Q_c time iii)
ax5 = fig.add_subplot(gs[6, :])
ax5, sm3 = plotchannels(mesh, np.abs(Qc[:, idx3]),contours=True,phi=phi[:,idx3]/1e6, ax=ax5, min=0.5, quiver=False)
ax5.set_xlim([0, 10e3])
ax5.set_ylim([0, 3e3])
ax5.text(0.01, 0.9, f"{tt[idx3]:.2f} yrs", transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=10)

# Modify x and y tick labels
ax5.set_xticks(ax5.get_xticks())
ax5.set_xticklabels([f"{x / 1e3:.0f}" for x in ax5.get_xticks()])
ax5.set_yticks(ax5.get_yticks())
ax5.set_yticklabels([f"{y / 1e3:.0f}" for y in ax5.get_yticks()])

ax5.set_xlabel('X [km]',labelpad=0.02)  # Set the x-axis label

divider3 = make_axes_locatable(ax5)
cax3 = divider3.append_axes("right", size="3%", pad=0.1)  # Reduce size and padding
cbar3 = fig.colorbar(sm3, cax=cax3)


# --- Ax6: bottom-left plot (Qr vs N phase plane)

ax6 = fig.add_subplot(gs[8,0])
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
cbar4 = fig.colorbar(scatter, cax = cax4, orientation='horizontal')
cax4.xaxis.set_ticks_position('top')
cax4.xaxis.set_label_position('top')
cbar4.set_label('Model time [yrs]')

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
    verticalalignment='bottom', horizontalalignment='center')
ax6.text(mid_x, Qr_line - 0.05 * (QrLake.max() - QrLake.min()), 'lake filling', 
     verticalalignment='top', horizontalalignment='center')



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
ax6.set_ylim(0, QrLake.max()+5)
ax6.margins(x=0, y=0)


# --- Ax7: bottom-right plot (ub vs x hovmoller)

ax7 = fig.add_subplot(gs[8,1])
ax7.set_xlabel('Model time [yrs]')
ax7.set_ylabel('X [km]', labelpad=0.05)  # Reduce padding between label and tick labels

# width average vel
[vel_avg, xedge] = width_average(mesh,vel,dx=0.15)

# compute time edges
dt = np.diff(tt).mean()
time_edge = np.concatenate(([tt[0] - dt / 2], tt + dt / 2))
[X,T] = np.meshgrid(xedge, time_edge)
pc = ax7.pcolormesh(T, X, np.transpose(vel_avg), cmap=cmocean.cm.deep_r, shading='flat', vmin=50,vmax=300)

# Add contour lines for velocity
contour_levels = np.arange(0, np.max(vel_avg)+20, 20)
cs = ax7.contour(T[:-1, :-1], X[:-1, :-1], np.transpose(vel_avg), 
                 levels=contour_levels, colors='white', linewidths=0.25)

# add Qr
ax7r = ax7.twinx()
ax7r.plot(tt, QrLake, color='k', label='Qr',linewidth=1)
ax7r.set_ylabel('$Q_r$ [m$^3$ s$^{-1}$]', color='k')
ax7r.tick_params(axis='y', labelcolor='k')

ax7.set_xlim(7, 9)
ax7.set_ylim(0.25, 10)  # Convert to km

# Set up divider for ax7
divider5 = make_axes_locatable(ax7)
divider6 = make_axes_locatable(ax7r)

# Add new Axes above ax7 for the colorbar
cax5 = divider5.append_axes("top", size="5%", pad=0.05)
cax6 = divider6.append_axes("top", size="5%", pad=0.05)

# Create horizontal colorbar in that new axis
cbar5 = fig.colorbar(pc, cax=cax5, orientation='horizontal')
cbar6 = fig.colorbar(pc, cax=cax6, orientation='horizontal')
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
    (-0.14, 1.15),  # ax1
    (-0.06, 1.15),   # ax2
    (-0.06, 1.1),  # ax3
    (-0.06, 1.1),  # ax4
    (-0.06, 1.1),  # ax5
    (-0.16, 1.25),  # ax6
    (-0.12, 1.25)    # ax7
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
fig.savefig(os.path.join(filedir, 'figure3/figure3.png'), dpi=300)
fig.savefig(os.path.join(filedir, 'figure3/figure3.pdf'), dpi=300)
fig.savefig(os.path.join(filedir, 'figure3/figure3.eps'), dpi=300)
fig.savefig(os.path.join(filedir, 'figure3/figure3.svg'), dpi=300)





















############################################################