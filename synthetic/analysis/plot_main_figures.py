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
import imageio
import matplotlib
import matplotlib.patches
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D

ISSM_DIR = os.getenv('ISSM_DIR')
print(f"ISSM_DIR: {ISSM_DIR}")
sys.path.append(os.path.join(ISSM_DIR, 'src/m/dev/'))
import devpath
from issmversion import issmversion
from model import model
from meshconvert import meshconvert

from src.utils import *
from matplotlib.collections import LineCollection


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

# FIGURE 2
############################################################

# Plot the model geometry and mesh


# Create figure with J. Glaciol. dual column spec
fig2 = plt.figure(figsize=(7.04724 / 2, 8.75 / 2))
gs = gridspec.GridSpec(2, 1, figure=fig2, hspace=0.15, wspace=0.25)
fig2.subplots_adjust(left=0.125, right=0.8, top=0.99, bottom=0.05)

# Add subplot
ax1 = fig2.add_subplot(gs[0, 0])

# Mesh triangulation and outline plot
mtri = Triangulation(md.mesh.x, md.mesh.y, md.mesh.elements - 1)
ax1.tripcolor(mtri, 0 * md.mesh.x, facecolor='none', edgecolor='k')

# Annotate lake outlet
x_pos, y_pos = md.mesh.x[lakepos], md.mesh.y[lakepos]
ax1.annotate(
    'Lake \noutlet',
    xy=(x_pos, y_pos),
    xycoords='data',
    xytext=(1.1, 0.5),
    textcoords='axes fraction',
    arrowprops=dict(facecolor='black', arrowstyle='->'),
    fontsize=10,
    color='black',
    ha='left',
    va='center',
    bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='white', lw=0.9)
)


# Set aspect ratio
ax1.set_aspect('equal')

# Axis tick settings
ax1.xaxis.set_major_locator(MultipleLocator(5e3))
ax1.xaxis.set_minor_locator(MultipleLocator(1e3))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

ax1.yaxis.set_major_locator(MultipleLocator(3e3))
ax1.yaxis.set_minor_locator(MultipleLocator(1.5e3))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Set axis limits
ax1.set_xlim([0, 10e3])
ax1.set_ylim([0, 3e3])

# Convert ticks to km labels
ax1.set_xticklabels([f"{x / 1e3:.0f}" for x in ax1.get_xticks()])
ax1.set_yticklabels([f"{y / 1e3:.0f}" for y in ax1.get_yticks()])
ax1.set_xlabel('$X$ [km]')
ax1.set_ylabel('$Y$ [km]')

ax2 = fig2.add_subplot(gs[1, 0],projection='3d', computed_zorder=False, facecolor='none')
ax2.plot_trisurf(mtri, bed, cmap=cmocean.cm.ice, vmin=0, vmax=60,
    edgecolor='none', linewidth=0., antialiased=False, rasterized=True)
tripc3d = ax2.plot_trisurf(mtri, surface, cmap=cmocean.cm.ice, alpha=1,
    antialiased=True, vmin=0, vmax=300, zorder=5, rasterized=True)
ax2.view_init(elev=30, azim=-125) #Works!
ax2.set_box_aspect((20, 5, 5))
ax2.set_aspect('equalxy')
# Axis tick settings
ax2.xaxis.set_major_locator(MultipleLocator(5e3))
ax2.xaxis.set_minor_locator(MultipleLocator(1e3))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax2.yaxis.set_major_locator(MultipleLocator(3e3))
ax2.yaxis.set_minor_locator(MultipleLocator(1.5e3))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Set axis limits
ax2.set_xlim([0, 10e3])
ax2.set_ylim([0, 3e3])

# Convert ticks to km labels
ax2.set_xticklabels([f"{x / 1e3:.0f}" for x in ax1.get_xticks()])
ax2.set_yticklabels([f"{y / 1e3:.0f}" for y in ax1.get_yticks()])
ax2.set_xlabel('$X$ [km]',labelpad=1)
ax2.set_ylabel('$Y$ [km]',labelpad=0.5)
ax2.set_zlim([0, 300])
ax2.xaxis.set_rotate_label(True)
ax2.yaxis.set_rotate_label(True)
ax2.zaxis.set_rotate_label(False)
ax2.set_zlabel('Elevation (m asl.)', rotation=90, labelpad=0)



# Save figure
os.makedirs(os.path.join(filedir, 'figure2'), exist_ok=True)
fig2.savefig(os.path.join(filedir, 'figure2/figure2.png'), dpi=300)
fig2.savefig(os.path.join(filedir, 'figure2/figure2.pdf'), dpi=300)
fig2.savefig(os.path.join(filedir, 'figure2/figure2.eps'), dpi=300)
fig2.savefig(os.path.join(filedir, 'figure2/figure2.svg'), dpi=300)

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
#resdir = '../experiments/RES/output_run_19_LoBSin'
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
idx1 = np.argmin(np.abs(tt - 7.9))
idx3 = np.argmin(np.abs(tt - 8.45))
# Find idx2: index of max Qr value between idx1 and idx3 (inclusive)
idx2_relative = np.argmax(Qr[lakepos,idx1:idx3 + 1])
idx2 = idx1 + idx2_relative



print('creating figure 3!')

fig3 = plt.figure(figsize=(7.04724*0.95, 8.75*0.95))  # J.Glac dual column figure spec
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
ax1.set_xlabel('$t$ [yrs]',labelpad=0.05)  # Reduce padding between label and tick labels
ax1.set_ylabel('$l_{\mathrm{h}}$ [m]',labelpad=2.5)
ax1.plot(tt, lh[lakepos, :], color='black', linestyle='-', label='$l_\mathrm{h}$ (Lake Height)')
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
ax1.text(tt[idx1] + 0.2, lh[lakepos, idx1] - 4.5, 'c', fontsize=10)
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
ax2.set_xlabel('$t$ [yrs]',labelpad=0.05)
ax2.set_ylabel('$Q_{\mathrm{r}}$ [m$^3$s$^{-1}$]',rotation=270, labelpad=15)  # Reduce padding between label and tick labels
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
ax2.text(tt[idx1] - 0.5, Qr[lakepos, idx1] + 4, 'c', fontsize=10)
ax2.text(tt[idx2] + 0.2, Qr[lakepos, idx2] - 2.85, 'd', fontsize=10)
ax2.text(tt[idx3] + 0.2, Qr[lakepos, idx3] + 3, 'e', fontsize=10)

ax2.set_ylim(-5,55)
ax2.set_xlim(0, 15)
ax2.margins(x=0, y=0)
ax2.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax2.xaxis.set_minor_locator(MultipleLocator(1))   
# --- Ax3: second row plot (Q_c time i)
cmap_plot = cmocean.cm.ice_r
lscale_plot = 3  # Scale for the quiver plot
Qcmin = 0.01
Qcmax = 50

ax3 = fig3.add_subplot(gs[2, :])

#ax3, sm1 = plotchannels(mesh,np.abs(Qc[:,idx1]),contours=True,phi=phi[:,idx1]/1e6,ax=ax3,min=0.5,quiver=False)
ax3, sm1  = plotchannels2(mesh, Qc[:, idx1], contours=True, phi=phi[:, idx1]/1e6, ax=ax3, 
                          vmin=Qcmin, vmax=Qcmax,lscale=lscale_plot,colormap=cmap_plot)

ax3.set_xlim([0, 10e3])
ax3.set_ylim([0, 3e3])
ax3.xaxis.set_minor_locator(MultipleLocator(1e3))
ax3.set_xticklabels([])  # Hide the x-axis ticks
ax3.set_xlabel('')  # Hide the x-axis label
ax3.set_yticks(ax3.get_yticks())
ax3.set_yticklabels([f"{y / 1e3:.0f}" for y in ax3.get_yticks()])
ax3.text(0.01, 0.9, f"{tt[idx1]:.2f} yrs", transform=plt.gca().transAxes,
    ha='left', va='center', fontsize=10)

# Annotate lake outlet
x_pos, y_pos = md.mesh.x[lakepos], md.mesh.y[lakepos]
ax3.annotate(
    'Lake \noutlet',
    xy=(x_pos, y_pos),
    xycoords='data',
    xytext=(1.1, 0.5),
    textcoords='axes fraction',
    arrowprops=dict(facecolor='black', arrowstyle='->'),
    fontsize=10,
    color='black',
    ha='left',
    va='center',
    bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='white', lw=0.9)
)



# --- Ax4: third row plot (Q_c time ii)
ax4 = fig3.add_subplot(gs[4, :])  # Extend ax4 across both columns
#ax4, sm2 = plotchannels(mesh, np.abs(Qc[:, idx2]),contours=True,phi=phi[:,idx2]/1e6, ax=ax4, min=0.5, quiver=False)
ax4, sm2  = plotchannels2(mesh, Qc[:, idx2], contours=True, phi=phi[:, idx2]/1e6, ax=ax4,
                          vmin=Qcmin, vmax=Qcmax,lscale=lscale_plot,colormap=cmap_plot)
ax4.set_xlim([0, 10e3])
ax4.set_ylim([0, 3e3])
ax4.text(0.01, 0.9, f"{tt[idx2]:.2f} yrs", transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=10)
ax4.xaxis.set_minor_locator(MultipleLocator(1e3))
ax4.set_xticklabels([])  # Hide the x-axis ticks
ax4.set_xlabel('')  # Hide the x-axis label
ax4.set_ylabel('$Y$ [km]')  # Set the y-axis label
ax4.set_yticks(ax4.get_yticks())
ax4.set_yticklabels([f"{y / 1e3:.0f}" for y in ax4.get_yticks()])
cax2 = fig3.add_axes([0.86, 0.468, 0.02, 0.2])  # [x0, y0, width, height] in figure coords
cbar2 = fig3.colorbar(sm2, cax=cax2)

# Add colorbar
cbar2 = fig3.colorbar(sm2, cax=cax2)

cbar2.set_label('$Q_{\mathrm{c}}$ [m$^3$s$^{-1}$]', rotation=270, labelpad=15)  # Rotate label and adjust padding
# Add tick labels every 10
cbar2.set_ticks([1,10,20,30,40,50])



# --- Ax5: fourth row plot (Q_c time iii)
ax5 = fig3.add_subplot(gs[6, :])
#ax5, sm3 = plotchannels(mesh, np.abs(Qc[:, idx3]),contours=True,phi=phi[:,idx3]/1e6, ax=ax5, min=0.5, quiver=False)
ax5, sm3  = plotchannels2(mesh, Qc[:, idx3], contours=True, phi=phi[:, idx3]/1e6, ax=ax5,
                           vmin=Qcmin, vmax=Qcmax,lscale=lscale_plot,colormap=cmap_plot)
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

ax5.set_xlabel('$X$ [km]',labelpad=0.02)  # Set the x-axis label


# --- Ax6: bottom-left plot (Qr vs N phase plane)

ax6 = fig3.add_subplot(gs[8,0])
QrLake = Qr[lakepos, :]
NLake = N[lakepos, :] / 1e6  # Convert to MPa
ax6.set_xlabel('$N$ [MPa]',labelpad=0.5)
ax6.set_ylabel('$Q_{\mathrm{r}}$ [m$^3$s$^{-1}$]',labelpad=2.5)

# Plot line in the background
#ax6.plot(NLake, QrLake, alpha=0.5, color='gray')
# Plot scatter points on top, using a greyscale colormap
# Create a line whose color varies with `tt`
points = np.array([NLake, QrLake]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=cmocean.cm.matter, norm=plt.Normalize(tt.min(), tt.max()))
lc.set_array(tt)
lc.set_linewidth(3)  # Set line width
ax6.add_collection(lc)

# Add a colorbar for the line
scatter = lc  # For compatibility with the existing colorbar setup

#set up divider for ax6
divider4 = make_axes_locatable(ax6)
cax4 = divider4.append_axes("top", size="5%", pad=0.05)
cbar4 = fig3.colorbar(scatter, cax = cax4, orientation='horizontal')
cax4.xaxis.set_ticks_position('top')
cax4.xaxis.set_label_position('top')
cbar4.set_label('$t$ [yrs]')
formatter = matplotlib.ticker.FormatStrFormatter('%d')
cbar4.formatter = formatter
cbar4.set_ticks(np.arange(0, 20, 5))
cbar4.update_ticks() # Important: update the ticks after changing the formatter


# Add vertical dashed line for "Lake empty"
lake_empty_N = (9.81 * md.materials.rho_ice * md.geometry.thickness[lakepos]) / 1e6  # Convert to MPa

ax6.axvline(x=lake_empty_N, linestyle='--', color='black', label='Lake empty')

# Annotate "lake empty" aligned with lake_empty_N
text_str = '$l_{\mathrm{h}}=0$ m'
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
ax6.text(mid_x, Qr_line + 0.04 * (QrLake.max() - QrLake.min()), 'lake draining', 
    verticalalignment='bottom', horizontalalignment='center',fontsize=8)
ax6.text(mid_x, Qr_line - 0.04 * (QrLake.max() - QrLake.min()), 'lake filling', 
     verticalalignment='top', horizontalalignment='center',fontsize=8)


"""
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
    step_back = 10
    if i > step_back:
        x_tail, y_tail = NLake[i - step_back], QrLake[i - step_back]
        ax6.annotate(
            '', # No text is needed for the arrow
            xy=(x_head, y_head),
            xytext=(x_tail, y_tail),
            arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
            zorder=20 # Ensure arrows are drawn on top
        )
"""

ax6.set_xlim(NLake.min()-0.05, lake_empty_N+0.05)
ax6.set_ylim(-5, QrLake.max()+5)

ax6.margins(x=0, y=0)


# --- Ax7: bottom-right plot (ub vs x hovmoller)
"""
ax7 = fig3.add_subplot(gs[8,1])
ax7.set_xlabel('$t$ [yrs]')
ax7.set_ylabel('$X$ [km]', labelpad=0.05)  # Reduce padding between label and tick labels

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
ax7.set_xlabel('$t$ [yrs]',labelpad=0.5)
ax7.xaxis.set_minor_locator(MultipleLocator(1))   
ax7.set_ylabel('$X$ [km]', labelpad=0.001)  # Reduce padding between label and tick labels

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
ax7r.set_ylabel('$Q_{\mathrm{r}}$ [m$^3$ s$^{-1}$]', color='k',rotation=270,labelpad=13)  # Rotate label and adjust padding)
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
cbar6.set_label('$U_{\mathrm{b}}$ [m a$^{-1}$]')


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
# Figure 3 /v2 plotting the flotation fraction, velocity, 
# and channel discharge during a flood cycle
fig3b = plt.figure(figsize=(7.04724*0.95, 8.75*0.95))
gs = gridspec.GridSpec(7, 3, figure=fig3b, hspace=0.01, wspace=0.05)
fig3b.subplots_adjust(left=0.055, right=0.995, top=0.96, bottom=0.05)

#define indexes for a full flood cycle
idx1b = np.argmin(np.abs(tt - 7))
idx2b = np.argmin(np.abs(tt - 7.93))
idx3b = np.argmin(np.abs(tt - 8.24))
idx4b = np.argmin(np.abs(tt - 8.4))
idx5b = np.argmin(np.abs(tt - 8.68))

# Create mesh
mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)

# First row
ax1a = fig3b.add_subplot(gs[0,0])
tric1a = ax1a.tripcolor(mtri, N[:,idx1b]/1e6, shading='gouraud', cmap=cmocean.cm.rain,vmin=0, vmax=1.6)
ax1a.set_aspect('equal')
ax1a.set_xlim([0, 10e3])
ax1a.set_ylim([0, 3e3])
ax1a.set_xticks(np.linspace(0, 10e3, 5))
ax1a.set_xticklabels([])
ax1a.set_yticks(np.linspace(0, 3e3, 2))
ax1a.set_yticklabels((ax1a.get_yticks() / 1e3).astype(int))

# FF colorbar
divider1a = make_axes_locatable(ax1a)
#set up divider for ax6
cax1a = divider1a.append_axes("top", size="20%", pad=0.175)
cbar1a = fig3b.colorbar(tric1a, cax = cax1a, orientation='horizontal')
cax1a.xaxis.set_ticks_position('top')
cax1a.xaxis.set_label_position('top')
cbar1a.set_label('$N$ [MPa]')

ax1b = fig3b.add_subplot(gs[0,1])
tric1b = ax1b.tripcolor(mtri, vel[:,idx1b], shading='gouraud', cmap=cmocean.cm.deep,vmin=0, vmax=200)
ax1b.set_aspect('equal')
ax1b.set_xlim([0, 10e3])
ax1b.set_ylim([0, 3e3])
ax1b.set_xticks(np.linspace(0, 10e3, 5))
ax1b.set_xticklabels([])
ax1b.set_yticks(np.linspace(0, 3e3, 2))
ax1b.set_yticklabels([])

# FF colorbar
divider1b = make_axes_locatable(ax1b)
#set up divider for ax6
cax1b = divider1b.append_axes("top", size="20%", pad=0.175)
cbar1b = fig3b.colorbar(tric1b, cax = cax1b, orientation='horizontal')
cax1b.xaxis.set_ticks_position('top')
cax1b.xaxis.set_label_position('top')
cbar1b.set_label('$U_{\mathrm{b}}$ [$m^{3}s^{-1}$]')


ax1c = fig3b.add_subplot(gs[0,2])
ax1c, sm1c = plotchannels(mesh, np.abs(Qc[:, idx1b]),contours=True,phi=phi[:,idx1b]/1e6, ax=ax1c, min=0.5,max=50, quiver=False)
#ax1c, sm1c = plotchannels2(mesh, Qc[:, idx1b], contours=True, phi=phi[:, idx1b] / 1e6, ax=ax1c, vmin=0.5, vmax=50)
ax1c.set_xlim([0, 10e3])
ax1c.set_ylim([0, 3e3])
ax1c.text(-0.55, 1.15, f"{tt[idx1b]:.2f} yrs", transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=10)
ax1c.set_xticks(np.linspace(0, 10e3, 5))
ax1c.set_xticklabels([])
ax1c.set_yticks(np.linspace(0, 3e3, 2))
ax1c.set_yticklabels([])

# Adjust colorbar size
divider1c = make_axes_locatable(ax1c)
cax1c = divider1c.append_axes("top", size="20%", pad=0.175)  # Reduce size and padding
cbar1c = fig3.colorbar(sm1c, cax=cax1c,orientation='horizontal')
cax1c.xaxis.set_ticks_position('top')
cax1c.xaxis.set_label_position('top')
cbar1c.set_label('$Q_{\mathrm{c}}$ [m$^3$s$^{-1}$]')  # Rotate label and adjust padding

# Second row
ax2a = fig3b.add_subplot(gs[1,0])
tric2a = ax2a.tripcolor(mtri, N[:,idx2b]/1e6, shading='gouraud', cmap=cmocean.cm.rain,vmin=0, vmax=1.6)
ax2a.set_aspect('equal')
ax2a.set_xlim([0, 10e3])
ax2a.set_ylim([0, 3e3])
ax2a.set_xticks(np.linspace(0, 10e3, 5))
ax2a.set_xticklabels([])
ax2a.set_yticks(np.linspace(0, 3e3, 2))
ax2a.set_yticklabels((ax2a.get_yticks() / 1e3).astype(int))


ax2b = fig3b.add_subplot(gs[1,1])
tric2b = ax2b.tripcolor(mtri, vel[:,idx2b], shading='gouraud', cmap=cmocean.cm.deep,vmin=20, vmax=200)
ax2b.set_aspect('equal')
ax2b.set_xlim([0, 10e3])
ax2b.set_ylim([0, 3e3])
ax2b.set_xticks(np.linspace(0, 10e3, 5))
ax2b.set_xticklabels([])
ax2b.set_yticks(np.linspace(0, 3e3, 2))
ax2b.set_yticklabels([])

ax2c = fig3b.add_subplot(gs[1,2])
ax2c, sm2c = plotchannels(mesh, np.abs(Qc[:, idx2b]),contours=True,phi=phi[:,idx2b]/1e6, ax=ax2c, min=0.5,max=50, quiver=False)
ax2c.set_xlim([0, 10e3])
ax2c.set_ylim([0, 3e3])
ax2c.text(-0.55, 1.15, f"{tt[idx2b]:.2f} yrs", transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=10)
ax2c.set_xticks(np.linspace(0, 10e3, 5))
ax2c.set_xticklabels([])
ax2c.set_yticks(np.linspace(0, 3e3, 2))
ax2c.set_yticklabels([])



os.makedirs(os.path.join(filedir, 'figure3b'), exist_ok=True)
fig3b.savefig(os.path.join(filedir, 'figure3b/figure3b.png'), dpi=300)
fig3b.savefig(os.path.join(filedir, 'figure3b/figure3b.pdf'), dpi=300)
fig3b.savefig(os.path.join(filedir, 'figure3b/figure3b.eps'), dpi=300)
fig3b.savefig(os.path.join(filedir, 'figure3b/figure3b.svg'), dpi=300)



############################################################
# BONUS, make a movie
makemovie = True

if makemovie:

    # Folder to store temporary frames
    frame_dir = os.path.join(filedir, 'movie1', 'frames')
    os.makedirs(frame_dir, exist_ok=True)

    # Time indices
    idxmov1 = np.argmin(np.abs(tt - 8.27))
    idxmov3 = np.argmin(np.abs(tt - 10))
    t = np.arange(idxmov1, idxmov3)

    # Generate and save each frame
    for i, ti in enumerate(t):
        fig = plt.figure(figsize=[8,2.5])
        gs = gridspec.GridSpec(1, 2, figure= fig,wspace=0.25, width_ratios=[0.9, 0.1])
        fig.subplots_adjust(left=0.06, right=0.925, top=0.9, bottom=0.1)
        ax = fig.add_subplot(gs[0, 0])

        # Plot frame
        ax, sm = plotchannels(mesh, np.abs(Qc[:, ti]), contours=True, phi=phi[:, ti] / 1e6,
                              ax=ax, min=0.5, max=50, quiver=False)

        ax.set_xlim([0, 10e3])
        ax.set_ylim([0, 3e3])
        ax.set_xlabel('$X$ [km]')
        ax.set_ylabel('$Y$ [km]')
        ax.set_xticklabels([f"{x / 1e3:.0f}" for x in ax.get_xticks()])
        ax.set_yticklabels([f"{y / 1e3:.0f}" for y in ax.get_yticks()])
        ax.text(0.01, 1.2, f"$t$ = {tt[ti]:.2f} yrs", transform=ax.transAxes,
                ha='left', va='center', fontsize=10)
        ax.text(0.01, 1.1, fr"$Q_{{\mathrm{{r}}}} = {Qr[lakepos, ti]:.2f}\ \mathrm{{m^3\,s^{{-1}}}}$",
                transform=ax.transAxes,
                ha='left', va='center', fontsize=10)

        # Adjust colorbar size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)  # Reduce size and padding
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('$Q_{\mathrm{c}}$ [m$^3$s$^{-1}$]',rotation=270,labelpad=15)  # Rotate label and adjust padding

        ax1 = fig.add_subplot(gs[0, 1])
        lake_label = ['lake 1']
        lake_height = lh[lakepos, ti]
        ax1.bar(lake_label, lake_height, color='blue',width=0.3)
        ax1.set_ylabel('$l_{\mathrm{h}}$ [m]',rotation=270)
        ax1.set_ylim(0, 100)
        ax1.set_xticks([])  # Hide the x-axis ticks
        ax1.margins(x=0, y=0)
        ax1.yaxis.set_label_position("right")
        ax1.yaxis.tick_right()


        # Save frame
        frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=200)
        plt.close(fig)

    # Build GIF from saved frames
    gif_path = os.path.join(filedir, 'movie1', 'channels_movie.gif')
    with imageio.get_writer(gif_path, mode='I', duration=0.2) as writer:
        for i in range(len(t)):
            frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
            image = imageio.imread(frame_path)
            writer.append_data(image)

    print("GIF saved to", gif_path)



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
ax1.plot(tt_default, lh_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax1.plot(tt_HiKc, lh_HiKc[lakepos, :], color='red', linestyle='-', linewidth=1.25,label='High $K_c$')
ax1.plot(tt_LoKc, lh_LoKc[lakepos, :], color='blue', linestyle='-', linewidth=1.25,label='Low $K_c$')
ax1.set_xlim(0, 15)
ax1.set_ylim(-5, 125)
ax1.margins(x=0, y=0)
ax1.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax1.xaxis.set_minor_locator(MultipleLocator(1))

ax2 = fig4.add_subplot(gs[0, 1])
ax2.plot(tt_default, Qr_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax2.plot(tt_HiKc, Qr_HiKc[lakepos, :], color='red', linestyle='-', linewidth=1.25,label='High $K_c$')
ax2.plot(tt_LoKc, Qr_LoKc[lakepos, :], color='blue', linestyle='-', linewidth=1.25,label='Low $K_c$')
ax2.set_xlim(0, 15)
ax2.set_ylim(-5, 150)
ax2.margins(x=0, y=0)
ax2.yaxis.tick_right()
ax2.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax2.xaxis.set_minor_locator(MultipleLocator(1))
ax2.text(1.52, 0.7, r"$k_{\mathrm{c}} = 0.25\ \mathrm{m}^{3/2}\ \mathrm{kg}^{-1/2}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='red')
ax2.text(1.5, 0.5, r"$k_{\mathrm{c}} = 0.1\ \mathrm{m}^{3/2}\ \mathrm{kg}^{-1/2}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='grey')
ax2.text(1.52, 0.3, r"$k_{\mathrm{c}} = 0.01\ \mathrm{m}^{3/2}\ \mathrm{kg}^{-1/2}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='blue')
# --- Ax3-4: Sheet conductivity
HiKs_dir = '../experiments/RES/output_run_4_HiShK'
LoKs_dir = '../experiments/RES/output_run_5_LoShK'
lh_HiKs = np.load(os.path.join(HiKs_dir, 'l_h.npy'))
lh_LoKs = np.load(os.path.join(LoKs_dir, 'l_h.npy'))
Qr_HiKs = np.load(os.path.join(HiKs_dir, 'Qr.npy'))
Qr_LoKs = np.load(os.path.join(LoKs_dir, 'Qr.npy'))
tt_HiKs = np.load(os.path.join(HiKs_dir, 'tt.npy'))
tt_LoKs = np.load(os.path.join(LoKs_dir, 'tt.npy'))

ax3 = fig4.add_subplot(gs[1, 0])
ax3.plot(tt_default, lh_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax3.plot(tt_HiKs, lh_HiKs[lakepos, :], color='red', linestyle='-', linewidth=1.25,label='High $K_s$')
ax3.plot(tt_LoKs, lh_LoKs[lakepos, :], color='blue', linestyle='-', linewidth=1.25,label='Low $K_s$')
ax3.set_xlim(0, 15)
ax3.set_ylim(-5, 125)
ax3.margins(x=0, y=0)
ax3.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax3.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax3.xaxis.set_minor_locator(MultipleLocator(1))

ax4 = fig4.add_subplot(gs[1, 1])
ax4.plot(tt_default, Qr_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax4.plot(tt_HiKs, Qr_HiKs[lakepos, :], color='red', linestyle='-', linewidth=1.25,label='High $K_s$')
ax4.plot(tt_LoKs, Qr_LoKs[lakepos, :], color='blue', linestyle='-', linewidth=1.25,label='Low $K_s$')
ax4.set_xlim(0, 15)
ax4.set_ylim(-5, 150)
ax4.margins(x=0, y=0)
ax4.yaxis.tick_right()
ax4.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax4.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax4.xaxis.set_minor_locator(MultipleLocator(1))
ax4.text(1.5, 0.7, r"$k_{\mathrm{s}} = 0.05\ \mathrm{Pa}\ \mathrm{s}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='red')
ax4.text(1.5, 0.5, r"$k_{\mathrm{s}} = 0.02\ \mathrm{Pa}\ \mathrm{s}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='grey')
ax4.text(1.52, 0.3, r"$k_{\mathrm{s}} = 0.005\ \mathrm{Pa}\ \mathrm{s}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='blue')

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
ax5.plot(tt_default, lh_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax5.plot(tt_HiBh, lh_HiBh[lakepos, :], color='red', linestyle='-', linewidth=1.25,label='High $B_h$')
ax5.plot(tt_LoBh, lh_LoBh[lakepos, :], color='blue', linestyle='-', linewidth=1.25,label='Low $B_h$')
ax5.set_xlim(0, 15)
ax5.set_ylim(-5, 125)
ax5.margins(x=0, y=0)
ax5.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax5.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax5.xaxis.set_minor_locator(MultipleLocator(1))

ax6 = fig4.add_subplot(gs[2, 1])
ax6.plot(tt_default, Qr_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax6.plot(tt_HiBh, Qr_HiBh[lakepos, :], color='red', linestyle='-', linewidth=1.25,label='High $B_h$')
ax6.plot(tt_LoBh, Qr_LoBh[lakepos, :], color='blue', linestyle='-', linewidth=1.25,label='Low $B_h$')
ax6.set_xlim(0, 15)
ax6.set_ylim(-5, 150)
ax6.margins(x=0, y=0)
ax6.yaxis.tick_right()
ax6.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax6.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax6.xaxis.set_minor_locator(MultipleLocator(1))

ax6.text(1.52, 0.7, r"$h_{\mathrm{b}} = 0.25\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='red')
ax6.text(1.5, 0.5, r"$h_{\mathrm{b}} = 0.1\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='grey')
ax6.text(1.52, 0.3, r"$h_{\mathrm{b}} = 0.05\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='blue')

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
ax7.plot(tt_default, lh_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax7.plot(tt_HiLc, lh_HiLc[lakepos, :], color='red', linestyle='-', linewidth=1.25,label='High $l_c$')
ax7.plot(tt_LoLc, lh_LoLc[lakepos, :], color='blue', linestyle='-', linewidth=1.25,label='Low $l_c$')
ax7.set_xlim(0, 15)
ax7.set_ylim(-5, 125)
ax7.margins(x=0, y=0)
ax7.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax7.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax7.xaxis.set_minor_locator(MultipleLocator(1))
ax7.set_ylabel('$l_{\mathrm{h}}$ [m]', labelpad=0.05)  # Reduce padding between label and tick labels

ax8 = fig4.add_subplot(gs[3, 1])
ax8.plot(tt_default, Qr_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax8.plot(tt_HiLc, Qr_HiLc[lakepos, :], color='red', linestyle='-', linewidth=1.25,label='High $l_c$')
ax8.plot(tt_LoLc, Qr_LoLc[lakepos, :], color='blue', linestyle='-', linewidth=1.25,label='Low $l_c$')
ax8.set_xlim(0, 15)
ax8.set_ylim(-5, 150)
ax8.margins(x=0, y=0)
ax8.yaxis.tick_right()
ax8.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax8.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax8.xaxis.set_minor_locator(MultipleLocator(1))
ax8.set_ylabel('$Q_{\mathrm{r}}$ [m$^3$s$^{-1}$]',rotation=270, labelpad=15)  # Reduce padding between label and tick labels
ax8.yaxis.set_label_position("right")

ax8.text(1.52, 0.7, r"$l_{\mathrm{c}} = 50\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='red')
ax8.text(1.52, 0.5, r"$l_{\mathrm{c}} = 20\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='grey')
ax8.text(1.5, 0.3, r"$l_{\mathrm{c}} = 5\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='blue')

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
ax9.plot(tt_default, lh_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax9.plot(tt_HiLr, lh_HiLr[lakepos, :], color='red', linestyle='-', linewidth=1.25,label='High $l_r$')
ax9.plot(tt_LoLr, lh_LoLr[lakepos, :], color='blue', linestyle='-', linewidth=1.25,label='Low $l_r$')
ax9.set_xlim(0, 15)
ax9.set_ylim(-5, 125)
ax9.margins(x=0, y=0)
ax9.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax9.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax9.xaxis.set_minor_locator(MultipleLocator(1))

ax10 = fig4.add_subplot(gs[4, 1])
ax10.plot(tt_default, Qr_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax10.plot(tt_HiLr, Qr_HiLr[lakepos, :], color='red', linestyle='-', linewidth=1.25,label='High $l_r$')
ax10.plot(tt_LoLr, Qr_LoLr[lakepos, :], color='blue', linestyle='-', linewidth=1.25,label='Low $l_r$')
ax10.set_xlim(0, 15)
ax10.set_ylim(-5, 150)
ax10.margins(x=0, y=0)
ax10.yaxis.tick_right()
ax10.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax10.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax10.xaxis.set_minor_locator(MultipleLocator(1))

ax10.text(1.52, 0.7, r"$l_{\mathrm{r}} = 20\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='red')
ax10.text(1.5, 0.5, r"$l_{\mathrm{r}} = 5\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='grey')
ax10.text(1.5, 0.3, r"$l_{\mathrm{r}} = 1\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='blue')

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
ax11.plot(tt_default, lh_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax11.plot(tt_HiMr, lh_HiMr[lakepos, :], color='red', linestyle='-', linewidth=1.25,label='High $M_r$')
ax11.plot(tt_LoMr, lh_LoMr[lakepos, :], color='blue', linestyle='-', linewidth=1.25,label='Low $M_r$')
ax11.set_xlim(0, 15)
ax11.set_ylim(-5, 125)
ax11.margins(x=0, y=0)
ax11.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax11.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax11.xaxis.set_minor_locator(MultipleLocator(1))

ax12 = fig4.add_subplot(gs[5, 1])
ax12.plot(tt_default, Qr_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax12.plot(tt_HiMr, Qr_HiMr[lakepos, :], color='red', linestyle='-', linewidth=1.25,label='High $M_r$')
ax12.plot(tt_LoMr, Qr_LoMr[lakepos, :], color='blue', linestyle='-', linewidth=1.25,label='Low $M_r$')
ax12.set_xlim(0, 15)
ax12.set_ylim(-5, 150)
ax12.margins(x=0, y=0)
ax12.yaxis.tick_right()
ax12.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax12.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax12.xaxis.set_minor_locator(MultipleLocator(1))

ax12.text(1.5, 0.7, r"$M_{\mathrm{r}} = 5\ \mathrm{m}\ \mathrm{a}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='red')
ax12.text(1.54, 0.5, r"$M_{\mathrm{r}} = 0.25\ \mathrm{m}\ \mathrm{a}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='grey')
ax12.text(1.52, 0.3, r"$M_{\mathrm{r}} = 0.1\ \mathrm{m}\ \mathrm{a}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='blue')

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
ax13.plot(tt_default, lh_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax13.plot(tt_HiQin, lh_HiQin[lakepos, :], color='red', linestyle='-', linewidth=1.25,label='High $Q_in$')
ax13.plot(tt_LoQin, lh_LoQin[lakepos, :], color='blue', linestyle='-', linewidth=1.25,label='Low $Q_in$')
ax13.set_xlim(0, 15)
ax13.set_ylim(-5, 125)
ax13.margins(x=0, y=0)
ax13.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax13.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax13.xaxis.set_minor_locator(MultipleLocator(1))
ax13.set_xlabel('$t$ [yrs]')

ax14 = fig4.add_subplot(gs[6, 1])
ax14.plot(tt_default, Qr_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25, label='Default')
ax14.plot(tt_HiQin, Qr_HiQin[lakepos, :], color='red', linestyle='-', linewidth=1.25, label='High $Q_in$')
ax14.plot(tt_LoQin, Qr_LoQin[lakepos, :], color='blue', linestyle='-', linewidth=1.25, label='Low $Q_in$')
ax14.set_xlim(0, 15)
ax14.set_ylim(-5, 150)
ax14.margins(x=0, y=0)
ax14.yaxis.tick_right()
ax14.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax14.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax14.xaxis.set_minor_locator(MultipleLocator(1))
ax14.set_xlabel('$t$ [yrs]')

ax14.text(1.52, 0.7, r"$Q_{\mathrm{in}} = 20\ \mathrm{m}^{3}\ \mathrm{s}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='red')
ax14.text(1.52, 0.5, r"$Q_{\mathrm{in}} = 10\ \mathrm{m}^{3}\ \mathrm{s}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='grey')
ax14.text(1.5, 0.3, r"$Q_{\mathrm{in}} = 5\ \mathrm{m}^{3}\ \mathrm{s}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='blue')


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
print('Making supplementary version of figure 4!')


fig4b = plt.figure(figsize=(7.04724, (8.75/7)*5))  # J.Glac dual column figure spec
gs = gridspec.GridSpec(5, 3, figure=fig4b, hspace=0.35, wspace=0.15, width_ratios=[0.4, 0.4, 0.2])
fig4b.subplots_adjust(left=0.075, right=0.925, top=0.98, bottom=0.075)

# Load default results once
default_dir = '../experiments/RES/output_run_1_Default'
lh_default = np.load(os.path.join(default_dir, 'l_h.npy'))
Qr_default = np.load(os.path.join(default_dir, 'Qr.npy'))
tt_default = np.load(os.path.join(default_dir, 'tt.npy'))

# --- Ax1-2: Englacial void ratio
HiEVR_dir = '../experiments/RES/output_run_6_HiEVR'
LoEVR_dir = '../experiments/RES/output_run_7_LoEVR'
lh_HiEVR = np.load(os.path.join(HiEVR_dir, 'l_h.npy'))
lh_LoEVR = np.load(os.path.join(LoEVR_dir, 'l_h.npy'))
Qr_HiEVR = np.load(os.path.join(HiEVR_dir, 'Qr.npy'))
Qr_LoEVR = np.load(os.path.join(LoEVR_dir, 'Qr.npy'))
tt_HiEVR = np.load(os.path.join(HiEVR_dir, 'tt.npy'))
tt_LoEVR = np.load(os.path.join(LoEVR_dir, 'tt.npy'))

ax1 = fig4b.add_subplot(gs[0, 0])
ax1.plot(tt_default, lh_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax1.plot(tt_HiEVR, lh_HiEVR[lakepos, :], color='red', linestyle='-', linewidth=1.25)
ax1.plot(tt_LoEVR, lh_LoEVR[lakepos, :], color='blue', linestyle='-', linewidth=1.25)
ax1.set_xlim(0, 15)
ax1.set_ylim(-5, 125)
ax1.margins(x=0, y=0)
ax1.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax1.xaxis.set_minor_locator(MultipleLocator(1))

ax2 = fig4b.add_subplot(gs[0, 1])
ax2.plot(tt_default, Qr_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax2.plot(tt_HiEVR, Qr_HiEVR[lakepos, :], color='red', linestyle='-', linewidth=1.25)
ax2.plot(tt_LoEVR, Qr_LoEVR[lakepos, :], color='blue', linestyle='-', linewidth=1.25)
ax2.set_xlim(0, 15)
ax2.set_ylim(-5, 150)
ax2.margins(x=0, y=0)
ax2.yaxis.tick_right()
ax2.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax2.xaxis.set_minor_locator(MultipleLocator(1))
ax2.text(1.5, 0.7, r"$e_{\mathrm{v}} = 1\mathrm{e}^{-2}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='red')
ax2.text(1.5, 0.5, r"$e_{\mathrm{v}} = 1\mathrm{e}^{-4}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='grey')
ax2.text(1.5, 0.3, r"$e_{\mathrm{v}} = 1\mathrm{e}^{-5}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='blue')

# --- Ax1-2: Friction coefficient
HiFricC_dir = '../experiments/RES/output_run_14_HiFricC'
LoFricC_dir = '../experiments/RES/output_run_15_LoFricC'
lh_HiFricC = np.load(os.path.join(HiFricC_dir, 'l_h.npy'))
lh_LoFricC = np.load(os.path.join(LoFricC_dir, 'l_h.npy'))
Qr_HiFricC = np.load(os.path.join(HiFricC_dir, 'Qr.npy'))
Qr_LoFricC = np.load(os.path.join(LoFricC_dir, 'Qr.npy'))
tt_HiFricC = np.load(os.path.join(HiFricC_dir, 'tt.npy'))
tt_LoFricC = np.load(os.path.join(LoFricC_dir, 'tt.npy'))

ax3 = fig4b.add_subplot(gs[1, 0])
ax3.plot(tt_default, lh_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax3.plot(tt_HiFricC, lh_HiFricC[lakepos, :], color='red', linestyle='-', linewidth=1.25)
ax3.plot(tt_LoFricC, lh_LoFricC[lakepos, :], color='blue', linestyle='-', linewidth=1.25)
ax3.set_xlim(0, 15)
ax3.set_ylim(-5, 125)
ax3.margins(x=0, y=0)
ax3.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax3.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax3.xaxis.set_minor_locator(MultipleLocator(1))

ax4 = fig4b.add_subplot(gs[1, 1])
ax4.plot(tt_default, Qr_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax4.plot(tt_HiFricC, Qr_HiFricC[lakepos, :], color='red', linestyle='-', linewidth=1.25)
ax4.plot(tt_LoFricC, Qr_LoFricC[lakepos, :], color='blue', linestyle='-', linewidth=1.25)
ax4.set_xlim(0, 15)
ax4.set_ylim(-5, 150)
ax4.margins(x=0, y=0)
ax4.yaxis.tick_right()
ax4.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax4.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax4.xaxis.set_minor_locator(MultipleLocator(1))
ax4.text(1.5, 0.7, r"$C = 200$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='red')
ax4.text(1.5, 0.5, r"$C = 100$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='grey')
ax4.text(1.5, 0.3, r"$C = 50$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='blue')

# --- Ax5-6: Friction p and q 
HiFricQ_dir = '../experiments/RES/output_run_17_HiFricq'
LoFricP_dir = '../experiments/RES/output_run_16_LoFricp'
lh_HiFricQ = np.load(os.path.join(HiFricQ_dir, 'l_h.npy'))
lh_LoFricP = np.load(os.path.join(LoFricP_dir, 'l_h.npy'))
Qr_HiFricQ = np.load(os.path.join(HiFricQ_dir, 'Qr.npy'))
Qr_LoFricP = np.load(os.path.join(LoFricP_dir, 'Qr.npy'))
tt_HiFricQ = np.load(os.path.join(HiFricQ_dir, 'tt.npy'))
tt_LoFricP = np.load(os.path.join(LoFricP_dir, 'tt.npy'))

ax5 = fig4b.add_subplot(gs[2, 0])
ax5.plot(tt_default, lh_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax5.plot(tt_HiFricQ, lh_HiFricQ[lakepos, :], color='red', linestyle='-', linewidth=1.25)
ax5.plot(tt_LoFricP, lh_LoFricP[lakepos, :], color='blue', linestyle='-', linewidth=1.25)
ax5.set_xlim(0, 15)
ax5.set_ylim(-5, 125)
ax5.margins(x=0, y=0)
ax5.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax5.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax5.xaxis.set_minor_locator(MultipleLocator(1))
ax5.set_ylabel('$l_{\mathrm{h}}$ [m]', labelpad=0.05)  # Reduce padding between label and tick labels


ax6 = fig4b.add_subplot(gs[2, 1])
ax6.plot(tt_default, Qr_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax6.plot(tt_HiFricQ, Qr_HiFricQ[lakepos, :], color='red', linestyle='-', linewidth=1.25)
ax6.plot(tt_LoFricP, Qr_LoFricP[lakepos, :], color='blue', linestyle='-', linewidth=1.25)
ax6.set_xlim(0, 15)
ax6.set_ylim(-5, 150)
ax6.margins(x=0, y=0)
ax6.yaxis.tick_right()
ax6.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax6.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax6.xaxis.set_minor_locator(MultipleLocator(1))
ax6.set_ylabel('$Q_{\mathrm{r}}$ [m$^3$s$^{-1}$]',rotation=270, labelpad=15)  # Reduce padding between label and tick labels
ax6.yaxis.set_label_position("right")
ax6.text(1.5, 0.7, r"$q = 2\ p = 4$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='red')
ax4.text(1.5, 0.5, r"$q = 1\ p = 4$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='grey')
ax4.text(1.5, 0.3, r"$q = 1\ p = 1$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='blue')

# --- Ax7-8: Bed slope
HiBSin_dir = '../experiments/RES/output_run_18_HiBSin'
LoBSin_dir = '../experiments/RES/output_run_19_LoBSin'
lh_HiBSin = np.load(os.path.join(HiBSin_dir, 'l_h.npy'))
lh_LoBSin = np.load(os.path.join(LoBSin_dir, 'l_h.npy'))
Qr_HiBSin = np.load(os.path.join(HiBSin_dir, 'Qr.npy'))
Qr_LoBSin = np.load(os.path.join(LoBSin_dir, 'Qr.npy'))
tt_HiBSin = np.load(os.path.join(HiBSin_dir, 'tt.npy'))
tt_LoBSin = np.load(os.path.join(LoBSin_dir, 'tt.npy'))

ax7 = fig4b.add_subplot(gs[3, 0])
ax7.plot(tt_default, lh_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax7.plot(tt_HiBSin, lh_HiBSin[lakepos, :], color='red', linestyle='-', linewidth=1.25)
ax7.plot(tt_LoBSin, lh_LoBSin[lakepos, :], color='blue', linestyle='-', linewidth=1.25)
ax7.set_xlim(0, 15)
ax7.set_ylim(-5, 125)
ax7.margins(x=0, y=0)
ax7.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax7.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax7.xaxis.set_minor_locator(MultipleLocator(1))

ax8 = fig4b.add_subplot(gs[3, 1])
ax8.plot(tt_default, Qr_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax8.plot(tt_HiBSin, Qr_HiBSin[lakepos, :], color='red', linestyle='-', linewidth=1.25)
ax8.plot(tt_LoBSin, Qr_LoBSin[lakepos, :], color='blue', linestyle='-', linewidth=1.25)
ax8.set_xlim(0, 15)
ax8.set_ylim(-5, 150)
ax8.margins(x=0, y=0)
ax8.yaxis.tick_right()
ax8.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax8.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax8.xaxis.set_minor_locator(MultipleLocator(1))
ax8.text(1.5, 0.7, r"$z_{\mathrm{b}}= X sin(0.01)$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='red')
ax8.text(1.5, 0.5, r"$z_{\mathrm{b}}= X sin(0.0075)$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='grey')
ax8.text(1.5, 0.3, r"$z_{\mathrm{b}}= X sin(0.0005)$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='blue')

# --- Ax9-10: Surface slope
HiHPar_dir = '../experiments/RES/output_run_20_HiHPar'
LoHPar_dir = '../experiments/RES/output_run_21_LoHPar'
lh_HiHPar = np.load(os.path.join(HiHPar_dir, 'l_h.npy'))
lh_LoHPar = np.load(os.path.join(LoHPar_dir, 'l_h.npy'))
Qr_HiHPar = np.load(os.path.join(HiHPar_dir, 'Qr.npy'))
Qr_LoHPar = np.load(os.path.join(LoHPar_dir, 'Qr.npy'))
tt_HiHPar = np.load(os.path.join(HiHPar_dir, 'tt.npy'))
tt_LoHPar = np.load(os.path.join(LoHPar_dir, 'tt.npy'))

ax9 = fig4b.add_subplot(gs[4, 0])
ax9.plot(tt_default, lh_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax9.plot(tt_HiHPar, lh_HiHPar[lakepos, :], color='red', linestyle='-', linewidth=1.25)
ax9.plot(tt_LoHPar, lh_LoHPar[lakepos, :], color='blue', linestyle='-', linewidth=1.25)
ax9.set_xlim(0, 15)
ax9.set_ylim(-5, 125)
ax9.margins(x=0, y=0)
ax9.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax9.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax9.xaxis.set_minor_locator(MultipleLocator(1))
ax9.set_xlabel('$t$ [yrs]')

ax10 = fig4b.add_subplot(gs[4, 1])
ax10.plot(tt_default, Qr_default[lakepos, :], color='gray', linestyle='-', linewidth=1.25,label='Default')
ax10.plot(tt_HiHPar, Qr_HiHPar[lakepos, :], color='red', linestyle='-', linewidth=1.25)
ax10.plot(tt_LoHPar, Qr_LoHPar[lakepos, :], color='blue', linestyle='-', linewidth=1.25)
ax10.set_xlim(0, 15)
ax10.set_ylim(-5, 150)
ax10.margins(x=0, y=0)
ax10.yaxis.tick_right()
ax10.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax10.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax10.xaxis.set_minor_locator(MultipleLocator(1))
ax10.set_xlabel('$t$ [yrs]')
ax10.text(1.5, 0.7, r"$H = 2\sqrt{X}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='red')
ax10.text(1.5, 0.5, r"$H = 1.1\sqrt{X}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='grey')
ax10.text(1.5, 0.3, r"$H = 0.5\sqrt{X}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='blue')

# Labels for subplots: (a) to (g)
labels = ['a','i', 'ii', 'b','i','ii', 'c','i','ii', 'd','i','ii', 'e','i','ii']
axes = [ax1,ax1,ax2, ax3, ax3, ax4, ax5, ax5, ax6, ax7, ax7, ax8, ax9, ax9, ax10]

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
fig4b.savefig(os.path.join(filedir, 'figure4/SUPPfigure4.png'), dpi=300)
fig4b.savefig(os.path.join(filedir, 'figure4/SUPPfigure4.pdf'), dpi=300)
fig4b.savefig(os.path.join(filedir, 'figure4/SUPPfigure4.eps'), dpi=300)
fig4b.savefig(os.path.join(filedir, 'figure4/SUPPfigure4.svg'), dpi=300)


############################################################

print('creating figure 5!')

fig5 = plt.figure(figsize=(7.04724, 5))  # J.Glac dual column figure spec
gs = gridspec.GridSpec(2, 3, figure=fig5, hspace=0.65, wspace=0.43)
fig5.subplots_adjust(left=0.06, right=0.918, top=0.87, bottom=0.088)

# ax1,ax2 = channel conductivity
Vel_HiKc = np.load(os.path.join(HiKc_dir, 'vel.npy'))
Vel_LoKc = np.load(os.path.join(LoKc_dir, 'vel.npy'))

ax1 = fig5.add_subplot(gs[0, 0])
ax1.set_ylabel('$X$ [km]', labelpad=0.001)
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
ax1r.set_ylim(-5, 150)
ax1.set_ylim(0.25, 10)  # Convert to km
ticks_X = [5, 10]
ax1.set_yticks(ticks_X)
ax1.xaxis.set_minor_locator(MultipleLocator(1))
ax1.yaxis.set_minor_locator(MultipleLocator(1))

# Set up divider for ax1
divider1 = make_axes_locatable(ax1)
divider2 = make_axes_locatable(ax1r)
# Add new Axes above ax1 for the colorbar
cax1 = divider1.append_axes("top", size="5%", pad=0.05)
cax2 = divider2.append_axes("top", size="5%", pad=0.05)
# Create horizontal colorbar in that new axis
cbar1 = fig5.colorbar(cf_HiKc, cax=cax1, orientation='horizontal')
cbar2 = fig5.colorbar(cf_HiKc, cax=cax2, orientation='horizontal')
cbar2.set_label('$U_{\mathrm{b}}$ [m a$^{-1}$]')
# Move ticks and label to top
cax1.xaxis.set_ticks_position('top')
cax1.xaxis.set_label_position('top')
cax2.xaxis.set_ticks_position('top')
cax2.xaxis.set_label_position('top')
# Set tick labels every 100 units based on colorbar limits
vmin, vmax = cf_HiKc.get_clim()  # Get the color limits of the plot
ticks = np.arange(np.ceil(vmin / 100) * 50, vmax + 1, 50)

cbar1.set_ticks(ticks)
cbar2.set_ticks(ticks)

ax1.text(0.5, 9.5, r"$k_{\mathrm{c}} = 0.25\ \mathrm{m}^{3/2}\ \mathrm{kg}^{-1/2}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='black')

ax2 = fig5.add_subplot(gs[1, 0])
ax2.set_ylabel('$X$ [km]', labelpad=0.001)
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
ax2.set_xlabel('$t$ [yrs]')
ax2r.set_ylim(0, 20)
ticks_X = [5, 10]
ax2.set_yticks(ticks_X)
ax2.xaxis.set_minor_locator(MultipleLocator(1))
ax2.yaxis.set_minor_locator(MultipleLocator(1))

# Set up divider for ax1
divider3 = make_axes_locatable(ax2)
divider4 = make_axes_locatable(ax2r)
# Add new Axes above ax1 for the colorbar
cax3 = divider3.append_axes("top", size="5%", pad=0.05)
cax4 = divider4.append_axes("top", size="5%", pad=0.05)
# Create horizontal colorbar in that new axis
cbar3 = fig5.colorbar(cf_LoKc, cax=cax3, orientation='horizontal')
cbar4 = fig5.colorbar(cf_LoKc, cax=cax4, orientation='horizontal')
cbar4.set_label('$U_{\mathrm{b}}$ [m a$^{-1}$]')
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

ax2.text(0.5, 9.5, r"$k_{\mathrm{c}} = 0.01\ \mathrm{m}^{3/2}\ \mathrm{kg}^{-1/2}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='black')

# ax3 and ax4: sheet conductivity

Vel_HiKs = np.load(os.path.join(HiKs_dir, 'vel.npy'))
Vel_LoKs = np.load(os.path.join(LoKs_dir, 'vel.npy'))

ax3 = fig5.add_subplot(gs[0, 1])
#ax3.set_ylabel('$X$ [km]', labelpad=0.001)
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
ticks_X = [5, 10]
ax3.set_yticks(ticks_X)
ax3.xaxis.set_minor_locator(MultipleLocator(1))
ax3.yaxis.set_minor_locator(MultipleLocator(1))



# Set up divider for ax3
divider3 = make_axes_locatable(ax3)
divider4 = make_axes_locatable(ax3r)
# Add new Axes above ax3 for the colorbar
cax5 = divider3.append_axes("top", size="5%", pad=0.05)
cax6 = divider4.append_axes("top", size="5%", pad=0.05)
# Create horizontal colorbar in that new axis
cbar5 = fig5.colorbar(cf_HiKs, cax=cax5, orientation='horizontal')
cbar6 = fig5.colorbar(cf_HiKs, cax=cax6, orientation='horizontal')
cbar6.set_label('$U_{\mathrm{b}}$ [m a$^{-1}$]')
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

ax3.text(0.5, 9.5, r"$k_{\mathrm{s}} = 0.1\ \mathrm{Pa}\ \mathrm{s}^{-1}$", 
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
ax4.set_xlabel('$t$ [yrs]')
ticks_X = [5, 10]
ax4.set_yticks(ticks_X)
ax4.xaxis.set_minor_locator(MultipleLocator(1))
ax4.yaxis.set_minor_locator(MultipleLocator(1))

# Set up divider for ax1
divider7 = make_axes_locatable(ax4)
divider8 = make_axes_locatable(ax4r)
# Add new Axes above ax1 for the colorbar
cax7 = divider7.append_axes("top", size="5%", pad=0.05)
cax8 = divider8.append_axes("top", size="5%", pad=0.05)
# Create horizontal colorbar in that new axis
cbar7 = fig5.colorbar(cf_LoKs, cax=cax7, orientation='horizontal')
cbar8 = fig5.colorbar(cf_LoKs, cax=cax8, orientation='horizontal')
cbar8.set_label('$U_{\mathrm{b}}$ [m a$^{-1}$]')
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

ax4.text(0.5, 9.5, r"$k_{\mathrm{s}} = 0.01\ \mathrm{Pa}\ \mathrm{s}^{-1}$", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='black')
         #ha='center', va='center', fontsize=9,color='black',bbox=dict(edgecolor='blue', alpha=1, facecolor='white'))


# ax5 and ax6: bed bump height

Vel_HiBh = np.load(os.path.join(HiBh_dir, 'vel.npy'))
Vel_LoBh = np.load(os.path.join(LoBh_dir, 'vel.npy'))

ax5 = fig5.add_subplot(gs[0, 2])
#ax3.set_ylabel('$X$ [km]', labelpad=0.001)
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
ax5r.set_ylabel('$Q_{\mathrm{r}}$ [m$^3$ s$^{-1}$]', color='k', rotation=270, labelpad=15)  # Rotate label and adjust padding
ax5r.tick_params(axis='y', labelcolor='k')
ax5.set_xlim(7, 11)
ax5.set_ylim(0.25, 10)  # Convert to km
ticks_X = [5, 10]
ax5.set_yticks(ticks_X)
ax5.xaxis.set_minor_locator(MultipleLocator(1))
ax5.yaxis.set_minor_locator(MultipleLocator(1))

# Set up divider for ax3
divider9 = make_axes_locatable(ax5)
divider10 = make_axes_locatable(ax5r)
# Add new Axes above ax3 for the colorbar
cax9 = divider9.append_axes("top", size="5%", pad=0.05)
cax10 = divider10.append_axes("top", size="5%", pad=0.05)
# Create horizontal colorbar in that new axis
cbar9 = fig5.colorbar(cf_HiBh, cax=cax9, orientation='horizontal')
cbar10 = fig5.colorbar(cf_HiBh, cax=cax10, orientation='horizontal')
cbar10.set_label('$U_{\mathrm{b}}$ [m a$^{-1}$]')
# Move ticks and label to top
cax9.xaxis.set_ticks_position('top')
cax9.xaxis.set_label_position('top')
cax10.xaxis.set_ticks_position('top')
cax10.xaxis.set_label_position('top')
# Set tick labels every 100 units based on colorbar limits
vmin, vmax = cf_HiBh.get_clim()  # Get the color limits of the plot
ticks = np.arange(np.ceil(vmin / 100) * 50, vmax + 1, 50)

cbar9.set_ticks(ticks)
cbar10.set_ticks(ticks)

ax5.text(0.5, 9.5, r"$h_{\mathrm{b}} = 1\ \mathrm{m} $", 
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
ax6r.set_ylabel('$Q_{\mathrm{r}}$ [m$^3$ s$^{-1}$]', color='k', rotation=270, labelpad=15)  # Rotate label and adjust padding
ax6r.tick_params(axis='y', labelcolor='k')
ax6.set_xlim(7, 11)
ax6.set_ylim(0.25, 10)  # Convert to km
ax6.set_xlabel('$t$ [yrs]')
ticks_X = [5, 10]
ax6.set_yticks(ticks_X)
ax6.xaxis.set_minor_locator(MultipleLocator(1))
ax6.yaxis.set_minor_locator(MultipleLocator(1))

# Set up divider for ax1
divider11 = make_axes_locatable(ax6)
divider12 = make_axes_locatable(ax6r)
# Add new Axes above ax1 for the colorbar
cax11 = divider11.append_axes("top", size="5%", pad=0.05)
cax12 = divider12.append_axes("top", size="5%", pad=0.05)
# Create horizontal colorbar in that new axis
cbar11 = fig5.colorbar(cf_LoBh, cax=cax11, orientation='horizontal')
cbar12 = fig5.colorbar(cf_LoBh, cax=cax12, orientation='horizontal')
cbar12.set_label('$U_{\mathrm{b}}$ [m a$^{-1}$]')
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

ax6.text(0.5, 9.5, r"$h_{\mathrm{b}} = 0.01\ \mathrm{m} $", 
         transform=plt.gca().transAxes,
         ha='center', va='center', fontsize=9,color='black')
         #ha='center', va='center', fontsize=9,color='black',bbox=dict(edgecolor='blue', alpha=1, facecolor='white'))
 # ax3 and ax4: sheet conductivity

 # Labels for subplots: (a) to (g)
labels = ['a', 'b', 'c', 'd', 'e','f']
axes = [ax1, ax2, ax3, ax4, ax5, ax6]

# Custom (x, y) positions for each label in axis coordinates
label_positions = [
    (-0.12, 1.5),  # ax1
    (-0.12, 1.5),   # ax2
    (-0.12, 1.5), # ax3
    (-0.12, 1.5), # ax4
    (-0.12, 1.5), # ax5
    (-0.12, 1.5),  # ax6
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



"""
CODE SNIPPETS:

Old channel plots in fig3
# --- Ax3: second row plot (Q_c time i)
lscale_plot = 3  # Scale for the quiver plot
Qcmin = 0.5
Qcmax = 50

ax3 = fig3.add_subplot(gs[2, :])
#ax3, sm1 = plotchannels(mesh,np.abs(Qc[:,idx1]),contours=True,phi=phi[:,idx1]/1e6,ax=ax3,min=0.5,quiver=False)
ax3, sm1  = plotchannels2(mesh, Qc[:, idx1], contours=True, phi=phi[:, idx1]/1e6, ax=ax3, vmin=Qcmin, vmax=Qcmax,lscale=lscale_plot)
ax3.set_xlim([0, 10e3])
ax3.set_ylim([0, 3e3])
ax3.set_xticks([])  # Hide the x-axis ticks
ax3.set_xlabel('')  # Hide the x-axis label
ax3.set_yticks(ax3.get_yticks())
ax3.set_yticklabels([f"{y / 1e3:.0f}" for y in ax3.get_yticks()])
ax3.text(0.01, 0.9, f"{tt[idx1]:.2f} yrs", transform=plt.gca().transAxes,
    ha='left', va='center', fontsize=10)
divider1 = make_axes_locatable(ax3)
cax1 = divider1.append_axes("right", size="3%", pad=0.25)  # Reduce size and padding
cbar1 = fig3.colorbar(sm1, cax=cax1)
cax1.set_visible(False)


# --- Ax4: third row plot (Q_c time ii)
ax4 = fig3.add_subplot(gs[4, :])  # Extend ax4 across both columns
#ax4, sm2 = plotchannels(mesh, np.abs(Qc[:, idx2]),contours=True,phi=phi[:,idx2]/1e6, ax=ax4, min=0.5, quiver=False)
ax4, sm2  = plotchannels2(mesh, Qc[:, idx2], contours=True, phi=phi[:, idx2]/1e6, ax=ax4, vmin=Qcmin, vmax=Qcmax,lscale=lscale_plot)
ax4.set_xlim([0, 10e3])
ax4.set_ylim([0, 3e3])
ax4.text(0.01, 0.9, f"{tt[idx2]:.2f} yrs", transform=plt.gca().transAxes,
         ha='left', va='center', fontsize=10)
ax4.set_xticks([])  # Hide the x-axis ticks
ax4.set_xlabel('')  # Hide the x-axis label
ax4.set_ylabel('$Y$ [km]')  # Set the y-axis label
ax4.set_yticks(ax4.get_yticks())
ax4.set_yticklabels([f"{y / 1e3:.0f}" for y in ax4.get_yticks()])
divider2 = make_axes_locatable(ax4)
cax2 = divider2.append_axes("right", size="3%", pad=0.25)

# Add colorbar
cbar2 = fig3.colorbar(sm2, cax=cax2,shrink=1.2, aspect=20)


cbar2.set_label('$Q_{\mathrm{c}}$ [m$^3$s$^{-1}$]', rotation=270, labelpad=15)  # Rotate label and adjust padding
# Add tick labels every 10
cbar2.set_ticks([1,10,20,30,40,50])



# --- Ax5: fourth row plot (Q_c time iii)
ax5 = fig3.add_subplot(gs[6, :])
#ax5, sm3 = plotchannels(mesh, np.abs(Qc[:, idx3]),contours=True,phi=phi[:,idx3]/1e6, ax=ax5, min=0.5, quiver=False)
ax5, sm3  = plotchannels2(mesh, Qc[:, idx3], contours=True, phi=phi[:, idx3]/1e6, ax=ax5, vmin=Qcmin, vmax=Qcmax,lscale=lscale_plot)
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

ax5.set_xlabel('$X$ [km]',labelpad=0.02)  # Set the x-axis label

divider3 = make_axes_locatable(ax5)
cax3 = divider3.append_axes("right", size="3%", pad=0.25)  # Reduce size and padding
cbar3 = fig3.colorbar(sm3, cax=cax3)
cax3.set_visible(False)
"""