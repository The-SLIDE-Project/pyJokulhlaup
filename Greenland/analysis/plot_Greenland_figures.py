"""
Plot the (Greenland relevant) figures for the main paper
========================================================

    Usage: python plot_Greenland_figures.py

    This script will plot:
        figure 1: Bed topography, 

"""
import sys
import os
import pickle
import textwrap
import cmocean
import imageio
from datetime import datetime
import matplotlib
import matplotlib.image 
import matplotlib.patches
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection, PolyCollection, PatchCollection
import matplotlib.gridspec as gridspec
matplotlib.use('TkAgg')
from matplotlib.colors import Normalize
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D
from src.utils import *
ISSM_DIR = os.getenv('ISSM_DIR')
sys.path.append(os.path.join(ISSM_DIR, 'src/m/dev/'))
import devpath
from issmversion import issmversion
from model import model
from meshconvert import meshconvert
#from read_netCDF import *

#sys.path.insert(0, os.path.join(ISSM_DIR, 'src/m/netcdf/'))

from read_netCDF import read_netCDF

from src.utils import *

#Set model path
IS_dir = '../models/23-HiKcZeroChSS-transient-29-09-2025-11-53'
print(os.getcwd())
Fig_dir = 'figures'
if not os.path.exists(Fig_dir):
    os.makedirs(Fig_dir)


# load results
md = read_netCDF(os.path.join(IS_dir, 'output-nc/output.nc'))
#x = md.mesh.x/1e3
#y = md.mesh.y/1e3
x = md.mesh.x
y = md.mesh.y

md.mesh.elements = md.mesh.elements.astype(int)

connect_edge = reorder_edges(md)
mesh = {
    'x': md.mesh.x,
    'y': md.mesh.y,
    'elements': md.mesh.elements,
    'connect_edge': connect_edge,
}

# load the hydrology params
lakemask_path = os.path.join(IS_dir, 'hydrology/lake_mask.npy')
Qin_path = os.path.join(IS_dir, 'hydrology/lake_Qin.npy')
spc_path = os.path.join(IS_dir, 'hydrology/spcphi.npy')
lakemask = np.load(lakemask_path)
Qin = np.load(Qin_path)
spc = np.load(spc_path)

# Geom
thickness = md.geometry.thickness
bed = md.geometry.bed
surface = bed + thickness

# load results
time_path = os.path.join(IS_dir, 'results/time.npy')
lh_path = os.path.join(IS_dir, 'results/HydrologyLakeHeight.npy')
Qr_path = os.path.join(IS_dir, 'results/HydrologyLakeOutletQr.npy')
ff_path = os.path.join(IS_dir, 'results/ff.npy')
phi_path = os.path.join(IS_dir, 'results/HydraulicPotential.npy')
N_path = os.path.join(IS_dir, 'results/EffectivePressure.npy')
Qc_path = os.path.join(IS_dir, 'results/ChannelDischarge.npy')

runoff    = md.smb.runoff[:-1]

runoff_tt = md.smb.runoff[-1]

tt = np.load(time_path)
lh = np.load(lh_path)
Qr = np.load(Qr_path)
ff = np.load(ff_path)
phi = np.load(phi_path)
N = np.load(N_path)
Qc = np.load(Qc_path)

# Create a mesh triangulation
meshtri = Triangulation(x, y, md.mesh.elements-1)

# Set lake and outlet positions
# Add lake outlet and spc outlet
lakepos1 = 6168
lakepos2 = 8733
spcpos  = 10655

# Set time span for  shading in fig 1 and span of fig 2
time_to_show_1 = 1710
time_to_show_2 = 1765
time_to_show_3 = 1850
time_to_show_4 = 1925
time_to_show_5 = 2000

# Create a figure with GridSpec
fig1 = plt.figure(figsize=(7.04724, 7))
gs1 =gridspec.GridSpec(5,3,
                       hspace=0.35, wspace=0.05,
                       left=0.07, bottom=0.062, right=0.98, top=0.975,
                       width_ratios=[0.01, 1, 0.01],
                       height_ratios=[0.05,1, 0.4,0.4, 0.4],
                       )


ax1 = fig1.add_subplot(gs1[1, :])
inset_width = 0.475
inset_height = 0.475
ax2 = ax1.inset_axes([-0.16, 0.8, inset_width, inset_height])  # top-left corner of ax1
ax3 = ax1.inset_axes([0.23, 0.73, 0.57, 0.57])

ax4 = fig1.add_subplot(gs1[2, 1])
ax5 = fig1.add_subplot(gs1[3, 1])
ax6 = fig1.add_subplot(gs1[4, 1])

# --- Top plot: Bed topography ---

tric1 = ax1.tripcolor(meshtri, bed, cmap=cmocean.cm.gray_r, edgecolors='w', linewidths=0.01, alpha=0.75)
#ax1, sm1 = plotchannels(mesh, np.abs(Qc[:,2000]), ax=ax1, min=1,quiver=False,linewidth=0.5)
ax1.set_aspect('equal')
ax1.set_xlim([-232.5*1e3, -194*1e3])
ax1.set_ylim([-2502*1e3, -2488*1e3])
ax1.set_axis_off()

Qcmin = 1
Qcmax = 100
cnorm = matplotlib.colors.Normalize(vmin=Qcmin, vmax=Qcmax)
time_to_show = 2000
Q_arr = Qc[:,time_to_show]

lscale = 1.5 
qlist = np.where(np.abs(Q_arr[:])>Qcmin)[0]
lc_colors = []
lc_lw = []
lc_xy = []
for i in qlist:
    Qi = np.abs(Q_arr[i])
    # if Qi>Smin:
    # ax.plot(mesh['x'][mesh['connect_edge'][i,:]]/1e3,
    #     mesh['y'][mesh['connect_edge'][i,:]]/1e3,
    #     linewidth=lscale*(0.25+1.25*cnorm(Qi)), 
    #     color=cmocean.cm.turbid(cnorm(Qi)))
    x0,x1 = mesh['x'][mesh['connect_edge'][i,:]]
    y0,y1 =  mesh['y'][mesh['connect_edge'][i,:]]
    lc_xy.append([(x0, y0), (x1, y1)])
    lc_lw.append(lscale*(0.25+1.25*cnorm(Qi)))
    lc_colors.append(cmocean.cm.ice_r(cnorm(Qi)))
lc = LineCollection(lc_xy, colors=lc_colors, linewidths=lc_lw,
    capstyle='round')
lc.set(rasterized=True)
ax1.add_collection(lc)
cax2 = ax1.inset_axes((0.4, 0.065, 0.25, 0.03))
cb2 = fig1.colorbar(matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmocean.cm.ice_r),
    cax=cax2, orientation='horizontal')
cb2.set_ticks([Qcmin, 50, 100, 150])
cb2.ax.xaxis.set_label_position('top')
cb2.set_label('$Q_{\mathrm{c}}$ [m$^3$ s$^{-1}$]')
# Coordinates
xmin = -232.5 * 1e3
ymin = -2502 * 1e3
ymax = -2488 * 1e3

# Scale bar parameters
bar_length = 5e3    # 50 km in meters
bar_height = 0.5e3     # 1 km thick (so visible on plot)
bar_x = xmin    # offset from left edge
bar_y = ymin + 0.01 * (ymax - ymin)  # offset from bottom

# Create rectangle
scale_rect = matplotlib.patches.Rectangle((bar_x, bar_y), bar_length, bar_height, zorder=15)
spc = PatchCollection([scale_rect], facecolor='k', clip_on=False)
ax1.add_collection(spc)

# Add text label
ax1.text(bar_x + bar_length/2, bar_y + bar_height + 0.075*(ymax - ymin),  # a little below bar
         '5 km', ha='center', va='top', fontsize=10, color='k')

lakepos = [lakepos1, lakepos2]  # Combine lakepos1 and lakepos2 into a single list

for idx in lakepos:  # Iterate through both lake positions
    ax1.plot(x[idx], y[idx], marker='*', color='fuchsia', markeredgecolor='black', markersize=11)

ax1.plot(x[spcpos], y[spcpos], marker='^', color='turquoise',markeredgecolor='black', markersize=9)

# Create dummy handles for legend
lake_handle = plt.Line2D([], [], marker='*', color='fuchsia',markeredgecolor='black', linestyle='none',
                         markersize=10, label='Lake outlet')
spc_handle  = plt.Line2D([], [], marker='^', color='turquoise',markeredgecolor='black', linestyle='none',
                         markersize=9, label='Glacier outlet')

# Add legend above scale bar
legend_x = bar_x + bar_length / 2
legend_y = bar_y + bar_height + 0.2 * (ymax - ymin)

ax1.legend(
    handles=[lake_handle, spc_handle],
    loc='center',
    bbox_to_anchor=(legend_x, legend_y),
    bbox_transform=ax1.transData,
    frameon=False,
    fontsize=10
)

# Get coordinates for all lake outlets
lake_coords = (x[lakepos1], y[lakepos1])
lake_coord2 = (x[lakepos2], y[lakepos2])

# Combine lake coordinates into a list
lake_coords = [lake_coords, lake_coord2]



# Sort by x-coordinate to determine west vs east
lake_coords_sorted = sorted(lake_coords, key=lambda p: p[0])  # (west, east)
west_out_coords, east_out_coords = lake_coords_sorted

# Labels with wrapping
west_label = "\n".join(textwrap.wrap("West lake outlet", width=12))
east_label = "\n".join(textwrap.wrap("East lake outlet", width=12))

# Annotate west lake
ax1.annotate(
    west_label,
    xy=west_out_coords, xycoords='data',
    xytext=(west_out_coords[0] + 2.5e3, west_out_coords[1] + 2.1e3),  # offset position
    arrowprops=dict(facecolor='black', arrowstyle="-", lw=0.8),
    ha='center', va='bottom', fontsize=9
)

# Annotate east lake
ax1.annotate(
    east_label,
    xy=east_out_coords, xycoords='data',
    xytext=(east_out_coords[0] + 4e3, east_out_coords[1] + 1e3),
    arrowprops=dict(facecolor='black', arrowstyle="-", lw=0.8),
    ha='center', va='bottom', fontsize=9
)

# Greenland map
img = matplotlib.image.imread('Greenland_IS_Location.png')
# Display in ax2
ax2.imshow(img)
ax2.axis('off')  # Optional: hide axis ticks and labels
ax2.set_facecolor('white')
ax2.text(200, 200, 'b', fontsize=10,
         verticalalignment='top', horizontalalignment='right',
         color='black', bbox=dict(facecolor='none', edgecolor='none'))


tric3 = ax3.tripcolor(meshtri, bed, cmap=cmocean.cm.gray_r,edgecolors='w', linewidths=0.0005)
ax3.set_aspect('equal')
ax3.set_axis_off()
# Get axis limits of ax1
x0, x1 = ax1.get_xlim()
y0, y1 = ax1.get_ylim()
cax = ax3.inset_axes((1, 0.35, 0.025, 0.5))
cbar = fig1.colorbar(tric3, shrink=0.5, pad=0.02, cax=cax, orientation='vertical')
cbar.set_label('$z_{\mathrm{b}}$ [m a.s.l]', fontsize=10,labelpad=12.0, rotation=270)
cbar.set_ticks([500, 0, -300])

# Change tick label font size
cbar.ax.tick_params(labelsize=10)

# Create a rectangle using ax1's extent in ax3
rect = matplotlib.patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                 edgecolor='black', facecolor='none', linewidth=1, linestyle='-')
ax3.add_patch(rect)
# Add label 'c' at top-left of the extent box
ax3.text(x0-1500, y1+1500, 'c', fontsize=10,
         verticalalignment='top', horizontalalignment='right',
         color='black', bbox=dict(facecolor='none', edgecolor='none'))

# --- thrid row, SMB ---
percentile_runoff = np.percentile(runoff, 95, axis=0)


ax4.plot(runoff_tt,percentile_runoff, color='black', linewidth=.25)
ax4.set_ylabel('P95 Runoff \n[m w.e. a$^{-1}$]',fontsize=10,labelpad=0.5)
ax4.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax4.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax4.xaxis.set_minor_locator(MultipleLocator(1))
ax4.set_xlabel('')  # Hide the x-axis label
ax4.margins(x=0,y=0)
ax4.axvspan(
    tt[time_to_show_1], tt[time_to_show_5],alpha=0.15,color='grey')

# --- fourth row, lh ---
# Load lake height record
datapath = '/Volumes/ajh24/GlaDS/GREENLAND_LAKES/Data/LakeMeasurements/IceMarginalLakeHeight.csv'

# Load CSV manually with numpy.genfromtxt
# Assuming the CSV has headers and columns: Medc, Datec, StDevc, Source (Datec format: yyyy-mm-dd)
data = np.genfromtxt(datapath, delimiter=',', names=True, dtype=None, encoding=None)

# Extract columns
lhm = data['Medc']
minlhm = np.min(lhm)
date_strings = data['Datec']  # strings like 'YYYY-MM-DD'
stdevc = data['StDevc']

# Parse dates into datetime objects
dates = np.array([datetime.strptime(d, '%Y-%m-%d') for d in date_strings])

# Extract years
years = np.array([d.year for d in dates])

# Compute start and end of each year
start_of_year = np.array([datetime(y, 1, 1) for y in years])
end_of_year = np.array([datetime(y + 1, 1, 1) for y in years])

# Calculate days in year
days_in_year = np.array([(e - s).days for s, e in zip(start_of_year, end_of_year)])

# Calculate decimal time
days_elapsed = np.array([(d - s).days for d, s in zip(dates, start_of_year)])
dectime = years + days_elapsed / days_in_year

# Calculate error (half standard deviation)
err = stdevc / 2


ax5.plot(dectime, lhm-minlhm, color='grey', linewidth=0.5)
ax5.fill_between(dectime, lhm-minlhm-err, lhm-minlhm+err, color='grey', alpha=0.5, label='$l_{\mathrm{h}}$ (meas.)')


ax5.plot(tt,np.max(lh,axis=0), color='black', linewidth=1, label='$l_{\mathrm{h}}$ (model)')
ax5.set_ylabel('$l_\mathrm{h}$ [m]',fontsize=10,labelpad=0.5)
#ax5.set_xticklabels([])  # Hide the x-axis ticks
ax5.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax5.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax5.xaxis.set_minor_locator(MultipleLocator(1))
ax5.set_xlim([np.floor(np.min(runoff_tt)), np.ceil(np.max(runoff_tt))])
ax5.margins(x=0,y=0.05)
ax5.legend(loc='upper right', fontsize=6, frameon=True, handlelength=0.5, handletextpad=0.2,
           labelspacing=0.1, borderpad=0.1, borderaxespad=0.1,facecolor='white',edgecolor='none')
ax5.axvspan(
    tt[time_to_show_1], tt[time_to_show_5],alpha=0.15,color='grey')

# --- fifth row, Qr ---
ax6.plot(tt,np.sum(Qr,axis=0), color='black', linewidth=1)
ax6.set_ylabel('$Q_\mathrm{r}$ [m$^3$ s$^{-1}$]',fontsize=10,labelpad=0.5)
ax6.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax6.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax6.xaxis.set_minor_locator(MultipleLocator(1))
ax6.set_xlabel('Model time [yrs]',fontsize=10)
ax6.set_xlim([np.floor(np.min(runoff_tt)), np.ceil(np.max(runoff_tt))])
ax6.margins(x=0,y=0.05)
ax6.axvspan(
    tt[time_to_show_1], tt[time_to_show_5],alpha=0.15,color='grey')

# Add vertical line
ax4.axvline(tt[time_to_show], color='gray', linestyle='--', linewidth=1)
ax5.axvline(tt[time_to_show], color='gray', linestyle='--', linewidth=1)
ax6.axvline(tt[time_to_show], color='gray', linestyle='--', linewidth=1)
# Add label 'c' slightly above the line
ax4.text(
    tt[time_to_show]+0.2, 0.83,
    'c',
    ha='center', va='bottom',
    fontsize=10,
    transform=matplotlib.transforms.blended_transform_factory(ax4.transData, ax4.transAxes)    
)
ax5.text(
    tt[time_to_show]+0.2, 0.83,
    'c',
    ha='center', va='bottom',
    fontsize=10,
    transform=matplotlib.transforms.blended_transform_factory(ax5.transData, ax5.transAxes)    
)
ax6.text(
    tt[time_to_show]+0.2, 0.83,
    'c',
    ha='center', va='bottom',
    fontsize=10,
    transform=matplotlib.transforms.blended_transform_factory(ax6.transData, ax6.transAxes)    
)

mid_tt = tt[time_to_show_1] + (tt[time_to_show_5] - tt[time_to_show_1]) / 2

# Label span of figure 2
ax4.text(
    mid_tt, 0.83,
    'Fig.X',
    ha='center', va='bottom',
    fontsize=8,
    color='gray',
    transform=matplotlib.transforms.blended_transform_factory(ax4.transData, ax4.transAxes)    
)
ax5.text(
    mid_tt, 0.83,
    'Fig.X',
    ha='center', va='bottom',
    fontsize=8,
    color='gray',
    transform=matplotlib.transforms.blended_transform_factory(ax5.transData, ax5.transAxes)    
)
ax6.text(
    mid_tt, 0.83,
    'Fig.X',
    ha='center', va='bottom',
    fontsize=8,
    color='gray',
    transform=matplotlib.transforms.blended_transform_factory(ax6.transData, ax6.transAxes)    
)

# Labels for subplots: (a) to (g)
labels = ['c', 'a', 'b', 'd', 'e', 'f']
axes = [ax1, ax1, ax1, ax4, ax5, ax6]

# Custom (x, y) positions for each label in axis coordinates
label_positions = [
    (-0.02, 0.8),  # ax1
    (-0.02, 1.28),   # ax2
    (0.275, 1.28),  # ax3
    (-0.085, 1.15),  # ax4
    (-0.085, 1.15),  # ax5
    (-0.085, 1.15),  # ax6
]

for ax, label, (x, y) in zip(axes, labels, label_positions):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=12,
        fontweight='bold'
    )

os.makedirs(os.path.join(Fig_dir, 'figure1'), exist_ok=True)
fig1.savefig(os.path.join(Fig_dir, 'figure1/figure1.png'), dpi=300)
fig1.savefig(os.path.join(Fig_dir, 'figure1/figure1.pdf'), dpi=300)
fig1.savefig(os.path.join(Fig_dir, 'figure1/figure1.eps'), dpi=300)
fig1.savefig(os.path.join(Fig_dir, 'figure1/figure1.svg'), dpi=300)

# Create a figure with GridSpec
fig2 = plt.figure(figsize=(7.04724/2, 3.5))
gs2 =gridspec.GridSpec(4,2,
                       hspace=0.25, wspace=0.2,
                       left=0.15, bottom=0.062, right=0.85, top=0.985,
                       height_ratios=[0.2, 0.01, 0.2, 0.2],
                       )

ax1 = fig2.add_subplot(gs2[0, :])
inset_width = 1.1
inset_height =1.1
ax2 = ax1.inset_axes([-0.42, -1.6, inset_width, inset_height]) 
ax3 = ax1.inset_axes([0.3, -1.6, inset_width, inset_height])  
ax5 = ax1.inset_axes([-0.42, -2.62, inset_width, inset_height]) 
ax6 = ax1.inset_axes([0.3, -2.62, inset_width, inset_height]) 

#ax2 = fig2.add_subplot(gs2[2, 0])
#ax3 = fig2.add_subplot(gs2[2, 1])
#ax4 = fig2.add_subplot(gs2[4, 0])
#ax5 = fig2.add_subplot(gs2[3, 0])
#ax6 = fig2.add_subplot(gs2[3, 1])

# --- Top plot: lh Qr time series ---
ax1.plot(tt, np.max(lh, axis=0), color='black', linewidth=1, label='$l_{\mathrm{h}}$ [m]')
ax1.set_ylabel('$l_\mathrm{h}$ [m]', fontsize=10, labelpad=0.5)
ax1.xaxis.set_major_locator(MultipleLocator(1))         # Tick every 3 years
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax1.xaxis.set_minor_locator(MultipleLocator(1/12))
ax1.set_xlabel('Model time [yrs]', fontsize=10)
ax1_secondary = ax1.twinx()
ax1_secondary.plot(tt, np.sum(Qr, axis=0), color='gray', linewidth=1, label='$Q_\mathrm{r}$ [m$^3$ s$^{-1}$]')
ax1_secondary.set_ylabel('$Q_\mathrm{r}$ [m$^3$ s$^{-1}$]', fontsize=10,rotation=270, labelpad=15,color='gray')
ax1.set_xlim(2017.25,2019.25)
ax1.margins(x=0, y=0)
ax1.set_ylim(0, 120)
ax1_secondary.set_ylim(-40, 80)


ax1.axvline(tt[time_to_show_1], color='gray', linestyle='--', linewidth=1)
ax1.axvline(tt[time_to_show_2], color='gray', linestyle='--', linewidth=1)
#ax1.axvline(tt[time_to_show_3], color='gray', linestyle='--', linewidth=1)
ax1.axvline(tt[time_to_show_4], color='gray', linestyle='--', linewidth=1)
ax1.axvline(tt[time_to_show_5], color='gray', linestyle='--', linewidth=1)
# Add label 'c' slightly above the line
ax1.text(
    tt[time_to_show_1]+0.05, 0.83,
    'b',
    ha='center', va='bottom',
    fontsize=10,
    transform=matplotlib.transforms.blended_transform_factory(ax1.transData, ax1.transAxes)    
)
ax1.text(
    tt[time_to_show_2]+0.05, 0.83,
    'c',
    ha='center', va='bottom',
    fontsize=10,
    transform=matplotlib.transforms.blended_transform_factory(ax1.transData, ax1.transAxes)    
)
#ax1.text(
#    tt[time_to_show_3]+0.05, 0.83,
#    'd',
#    ha='center', va='bottom',
#    fontsize=10,
#    transform=matplotlib.transforms.blended_transform_factory(ax1.transData, ax1.transAxes)    
#)
ax1.text(
    tt[time_to_show_4]+0.05, 0.83,
    'd',
    ha='center', va='bottom',
    fontsize=10,
    transform=matplotlib.transforms.blended_transform_factory(ax1.transData, ax1.transAxes)    
)
ax1.text(
    tt[time_to_show_5]+0.05, 0.83,
    'e',
    ha='center', va='bottom',
    fontsize=10,
    transform=matplotlib.transforms.blended_transform_factory(ax1.transData, ax1.transAxes)    
)



# --- Second plot: Qc time series 1
tric1 = ax2.tripcolor(meshtri, bed, cmap=cmocean.cm.gray_r, edgecolors='w', linewidths=0.01, alpha=0.75)
#ax1, sm1 = plotchannels(mesh, np.abs(Qc[:,2000]), ax=ax1, min=1,quiver=False,linewidth=0.5)
ax2.set_aspect('equal')
xmin = -232.5 * 1e3
xmax = -217 * 1e3
ymin = -2498 * 1e3
ymax = -2489 * 1e3
ax2.set_xlim([xmin, xmax])
ax2.set_ylim([ymin, ymax])
ax2.set_axis_off()

Qcmin = 1
Qcmax = 300
cnorm = matplotlib.colors.Normalize(vmin=Qcmin, vmax=Qcmax)
Q_arr = Qc[:,time_to_show_1]

lscale = 2 
qlist = np.where(np.abs(Q_arr[:])>Qcmin)[0]
lc_colors1 = []
lc_lw1 = []
lc_xy1 = []
for i in qlist:
    Qi = np.abs(Q_arr[i])
    # if Qi>Smin:
    # ax.plot(mesh['x'][mesh['connect_edge'][i,:]]/1e3,
    #     mesh['y'][mesh['connect_edge'][i,:]]/1e3,
    #     linewidth=lscale*(0.25+1.25*cnorm(Qi)), 
    #     color=cmocean.cm.turbid(cnorm(Qi)))
    x0,x1 = mesh['x'][mesh['connect_edge'][i,:]]
    y0,y1 =  mesh['y'][mesh['connect_edge'][i,:]]
    lc_xy1.append([(x0, y0), (x1, y1)])
    lc_lw1.append(lscale*(0.25+1.25*cnorm(Qi)))
    lc_colors1.append(cmocean.cm.ice_r(cnorm(Qi)))
lc1 = LineCollection(lc_xy1, colors=lc_colors1, linewidths=lc_lw1,
    capstyle='round')
lc1.set(rasterized=True)
ax2.add_collection(lc1)
ax2.plot(mesh['x'][(lakepos1)], mesh['y'][(lakepos1)], marker='*', color='fuchsia', markeredgecolor='black', markersize=8, markeredgewidth=0.5)
ax2.plot(mesh['x'][(lakepos2)], mesh['y'][(lakepos2)], marker='*', color='fuchsia', markeredgecolor='black', markersize=8, markeredgewidth=0.5)
# Plot the glacier outlet position
ax2.plot(mesh['x'][spcpos], mesh['y'][spcpos], marker='^', color='turquoise', markeredgecolor='black', markersize=6, markeredgewidth=0.5)
#cax2 = ax2.inset_axes((1.01, 0, 0.03, 1))
#cb1 = fig1.colorbar(matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmocean.cm.ice_r),
#    cax=cax2, orientation='vertical')
#cb1.set_ticks([Qcmin, 50, 100, 150])
#cb1.ax.xaxis.set_label_position('top')
#cb1.set_label('$Q_{\mathrm{c}}$ [m$^3$ s$^{-1}$]',rotation=270, labelpad=15, fontsize=10)
#
## Scale bar parameters
#bar_length = 5e3    # 50 km in meters
#bar_height = 0.5e3     # 1 km thick (so visible on plot)
#bar_x = xmin    # offset from left edge
#bar_y = ymin + 0.0 * (ymax - ymin)  # offset from bottom
#
## Create rectangle
#scale_rect = matplotlib.patches.Rectangle((bar_x, bar_y), bar_length, bar_height, zorder=15)
#spc = PatchCollection([scale_rect], facecolor='k', clip_on=False)
#ax2.add_collection(spc)
#
## Add text label
#ax2.text(bar_x + bar_length/2, bar_y + bar_height + 0.75*(ymax - ymin),  # a little below bar
#         '5 km', ha='center', va='top', fontsize=10, color='k')


#for idx in lakepos[0]:  # use lakepos[0] because np.where returns a tuple
#    ax2.plot(x[idx], y[idx], marker='*', color='fuchsia',markeredgecolor='black', markersize=11)

#ax2.plot(x[spcpos], y[spcpos], marker='^', color='turquoise',markeredgecolor='black', markersize=9)

# --- third plot: Qc time series 2
tric2 = ax3.tripcolor(meshtri, bed, cmap=cmocean.cm.gray_r, edgecolors='w', linewidths=0.01, alpha=0.75)
#ax1, sm1 = plotchannels(mesh, np.abs(Qc[:,2000]), ax=ax1, min=1,quiver=False,linewidth=0.5)
ax3.set_aspect('equal')
ax3.set_xlim([xmin, xmax])
ax3.set_ylim([ymin, ymax])
ax3.set_axis_off()


cnorm = matplotlib.colors.Normalize(vmin=Qcmin, vmax=Qcmax)
Q_arr = Qc[:,time_to_show_2]

lscale = 2
qlist = np.where(np.abs(Q_arr[:])>Qcmin)[0]
lc_colors2 = []
lc_lw2 = []
lc_xy2 = []
for i in qlist:
    Qi = np.abs(Q_arr[i])
    # if Qi>Smin:
    # ax.plot(mesh['x'][mesh['connect_edge'][i,:]]/1e3,
    #     mesh['y'][mesh['connect_edge'][i,:]]/1e3,
    #     linewidth=lscale*(0.25+1.25*cnorm(Qi)), 
    #     color=cmocean.cm.turbid(cnorm(Qi)))
    x0,x1 = mesh['x'][mesh['connect_edge'][i,:]]
    y0,y1 =  mesh['y'][mesh['connect_edge'][i,:]]
    lc_xy2.append([(x0, y0), (x1, y1)])
    lc_lw2.append(lscale*(0.25+1.25*cnorm(Qi)))
    lc_colors2.append(cmocean.cm.ice_r(cnorm(Qi)))
lc2 = LineCollection(lc_xy2, colors=lc_colors2, linewidths=lc_lw2,
    capstyle='round')
lc2.set(rasterized=True)
ax3.add_collection(lc2)
cax3 = ax3.inset_axes((0.1, -0.9, 0.7, 0.06))
ax3.plot(mesh['x'][(lakepos1)], mesh['y'][(lakepos1)], marker='*', color='fuchsia', markeredgecolor='black', markersize=8, markeredgewidth=0.5)
ax3.plot(mesh['x'][(lakepos2)], mesh['y'][(lakepos2)], marker='*', color='fuchsia', markeredgecolor='black', markersize=8, markeredgewidth=0.5)
# Plot the glacier outlet position
ax3.plot(mesh['x'][spcpos], mesh['y'][spcpos], marker='^', color='turquoise', markeredgecolor='black', markersize=6, markeredgewidth=0.5)

# --- fourth plot: Qc time series 3
#tric3 = ax4.tripcolor(meshtri, bed, cmap=cmocean.cm.gray_r, edgecolors='w', linewidths=0.01, alpha=0.75)
##ax1, sm1 = plotchannels(mesh, np.abs(Qc[:,2000]), ax=ax1, min=1,quiver=False,linewidth=0.5)
#ax4.set_aspect('equal')
#ax4.set_xlim([xmin, xmax])
#ax4.set_ylim([ymin, ymax])
#ax4.set_axis_off()
#
#
#cnorm = matplotlib.colors.Normalize(vmin=Qcmin, vmax=Qcmax)
#Q_arr = Qc[:,time_to_show_3]
#
#lscale = 1.5 
#qlist = np.where(np.abs(Q_arr[:])>Qcmin)[0]
#lc_colors3 = []
#lc_lw3 = []
#lc_xy3 = []
#for i in qlist:
#    Qi = np.abs(Q_arr[i])
#    # if Qi>Smin:
#    # ax.plot(mesh['x'][mesh['connect_edge'][i,:]]/1e3,
#    #     mesh['y'][mesh['connect_edge'][i,:]]/1e3,
#    #     linewidth=lscale*(0.25+1.25*cnorm(Qi)), 
#    #     color=cmocean.cm.turbid(cnorm(Qi)))
#    x0,x1 = mesh['x'][mesh['connect_edge'][i,:]]
#    y0,y1 =  mesh['y'][mesh['connect_edge'][i,:]]
#    lc_xy3.append([(x0, y0), (x1, y1)])
#    lc_lw3.append(lscale*(0.25+1.25*cnorm(Qi)))
#    lc_colors3.append(cmocean.cm.ice_r(cnorm(Qi)))
#lc3 = LineCollection(lc_xy3, colors=lc_colors3, linewidths=lc_lw3,
#    capstyle='round')
#lc3.set(rasterized=True)
#ax4.add_collection(lc3)
##cax4 = ax4.inset_axes((1.01, 0, 0.03, 1))
#cb3 = fig1.colorbar(matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmocean.cm.ice_r),
#    cax=cax3, orientation='vertical')
#cb3.set_ticks([Qcmin, 50, 100, 150, 200, 250])
#cb3.ax.xaxis.set_label_position('top')
#cb3.set_label('$Q_{\mathrm{c}}$ [m$^3$ s$^{-1}$]',rotation=270, labelpad=15, fontsize=10)


# --- fifth plot: Qc time series 4
tric4 = ax5.tripcolor(meshtri, bed, cmap=cmocean.cm.gray_r, edgecolors='w', linewidths=0.01, alpha=0.75)
#ax1, sm1 = plotchannels(mesh, np.abs(Qc[:,2000]), ax=ax1, min=1,quiver=False,linewidth=0.5)
ax5.set_aspect('equal')
ax5.set_xlim([xmin, xmax])
ax5.set_ylim([ymin, ymax])
ax5.set_axis_off()


cnorm = matplotlib.colors.Normalize(vmin=Qcmin, vmax=Qcmax)
Q_arr = Qc[:,time_to_show_4]

lscale = 2 
qlist = np.where(np.abs(Q_arr[:])>Qcmin)[0]
lc_colors4 = []
lc_lw4 = []
lc_xy4 = []
for i in qlist:
    Qi = np.abs(Q_arr[i])
    # if Qi>Smin:
    # ax.plot(mesh['x'][mesh['connect_edge'][i,:]]/1e3,
    #     mesh['y'][mesh['connect_edge'][i,:]]/1e3,
    #     linewidth=lscale*(0.25+1.25*cnorm(Qi)), 
    #     color=cmocean.cm.turbid(cnorm(Qi)))
    x0,x1 = mesh['x'][mesh['connect_edge'][i,:]]
    y0,y1 =  mesh['y'][mesh['connect_edge'][i,:]]
    lc_xy4.append([(x0, y0), (x1, y1)])
    lc_lw4.append(lscale*(0.25+1.25*cnorm(Qi)))
    lc_colors4.append(cmocean.cm.ice_r(cnorm(Qi)))
lc4 = LineCollection(lc_xy4, colors=lc_colors4, linewidths=lc_lw4,
    capstyle='round')
lc4.set(rasterized=True)
ax5.add_collection(lc4)

# Scale bar parameters
bar_length = 5e3    # 50 km in meters
bar_height = 0.5e3     # 1 km thick (so visible on plot)
bar_x = xmin+0.5e3    # offset from left edge
bar_y = ymin + 0.0 * (ymax - ymin)  # offset from bottom

# Create rectangle
scale_rect = matplotlib.patches.Rectangle((bar_x, bar_y), bar_length, bar_height, zorder=15)
spc = PatchCollection([scale_rect], facecolor='k', clip_on=False)
ax5.add_collection(spc)

# Add text label
ax5.text(bar_x + bar_length/2, bar_y + bar_height + 0.15*(ymax - ymin),  # a little below bar
         '5 km', ha='center', va='top', fontsize=10, color='k')


ax5.plot(mesh['x'][(lakepos1)], mesh['y'][(lakepos1)], marker='*', color='fuchsia', markeredgecolor='black', markersize=8, markeredgewidth=0.5)
ax5.plot(mesh['x'][(lakepos2)], mesh['y'][(lakepos2)], marker='*', color='fuchsia', markeredgecolor='black', markersize=8, markeredgewidth=0.5)
# Plot the glacier outlet position
ax5.plot(mesh['x'][spcpos], mesh['y'][spcpos], marker='^', color='turquoise', markeredgecolor='black', markersize=6, markeredgewidth=0.5)

# Create dummy handles for legend
lake_handle = plt.Line2D([], [], marker='*', color='fuchsia',markeredgecolor='black', linestyle='none',
                         markersize=9, label='Lake outlet', markeredgewidth=0.5)
spc_handle  = plt.Line2D([], [], marker='^', color='turquoise',markeredgecolor='black', linestyle='none',
                         markersize=8, label='Glacier outlet', markeredgewidth=0.5)

# Legend position just to the right of the scale bar
legend_x = bar_x + bar_length + 0.25e3   # 0.5 km gap to the right
legend_y = (bar_y + bar_height /2) - 0.6e3       # vertically centered on bar

ax5.legend(
    handles=[lake_handle, spc_handle],
    loc='center left',                   # anchor legend's left edge to this point
    bbox_to_anchor=(legend_x, legend_y),
    bbox_transform=ax5.transData,        # match data coords of scale bar
    frameon=False,
    fontsize=8
)




# --- sixth plot: Qc time series 5
tric5 = ax6.tripcolor(meshtri, bed, cmap=cmocean.cm.gray_r, edgecolors='w', linewidths=0.01, alpha=0.75)
#ax1, sm1 = plotchannels(mesh, np.abs(Qc[:,2000]), ax=ax1, min=1,quiver=False,linewidth=0.5)
ax6.set_aspect('equal')
ax6.set_xlim([xmin, xmax])
ax6.set_ylim([ymin, ymax])
ax6.set_axis_off()


cnorm = matplotlib.colors.Normalize(vmin=Qcmin, vmax=Qcmax)
Q_arr = Qc[:,time_to_show_5]

lscale = 2
qlist = np.where(np.abs(Q_arr[:])>Qcmin)[0]
lc_colors5 = []
lc_lw5 = []
lc_xy5 = []
for i in qlist:
    Qi = np.abs(Q_arr[i])
    # if Qi>Smin:
    # ax.plot(mesh['x'][mesh['connect_edge'][i,:]]/1e3,
    #     mesh['y'][mesh['connect_edge'][i,:]]/1e3,
    #     linewidth=lscale*(0.25+1.25*cnorm(Qi)), 
    #     color=cmocean.cm.turbid(cnorm(Qi)))
    x0,x1 = mesh['x'][mesh['connect_edge'][i,:]]
    y0,y1 =  mesh['y'][mesh['connect_edge'][i,:]]
    lc_xy5.append([(x0, y0), (x1, y1)])
    lc_lw5.append(lscale*(0.25+1.25*cnorm(Qi)))
    lc_colors5.append(cmocean.cm.ice_r(cnorm(Qi)))
lc5 = LineCollection(lc_xy5, colors=lc_colors5, linewidths=lc_lw5,
    capstyle='round')
lc5.set(rasterized=True)
ax6.add_collection(lc5)
ax6.plot(mesh['x'][(lakepos1)], mesh['y'][(lakepos1)], marker='*', color='fuchsia', markeredgecolor='black', markersize=8, markeredgewidth=0.5)
ax6.plot(mesh['x'][(lakepos2)], mesh['y'][(lakepos2)], marker='*', color='fuchsia', markeredgecolor='black', markersize=8, markeredgewidth=0.5)
# Plot the glacier outlet position
ax6.plot(mesh['x'][spcpos], mesh['y'][spcpos], marker='^', color='turquoise', markeredgecolor='black', markersize=6, markeredgewidth=0.5)

#cax1 = ax6.inset_axes((0.1, -0.2, 1, 0.03))
cb1 = fig1.colorbar(matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmocean.cm.ice_r),
    cax=cax3, orientation='horizontal')
cb1.set_ticks([Qcmin, 100, 200, 300])
cb1.ax.tick_params(labelsize=8)
cb1.ax.xaxis.set_label_position('top')
cb1.set_label('$Q_{\mathrm{c}}$ [m$^3$ s$^{-1}$]', fontsize=8)

# Labels for time
ax2.text(0.37, 0.91, f"{tt[time_to_show_1]:.2f} yrs", transform=ax2.transAxes,
         ha='center', va='center', fontsize=8)
# Labels for time
ax3.text(0.37, 0.91, f"{tt[time_to_show_2]:.2f} yrs", transform=ax3.transAxes,
         ha='center', va='center', fontsize=8)
# Labels for time
ax5.text(0.37, 0.91, f"{tt[time_to_show_4]:.2f} yrs", transform=ax5.transAxes,
         ha='center', va='center', fontsize=8)
# Labels for time
ax6.text(0.37, 0.91, f"{tt[time_to_show_5]:.2f} yrs", transform=ax6.transAxes,
         ha='center', va='center', fontsize=8)

# Labels for subplots: (a) to (g)
labels = ['a', 'b', 'c', 'd', 'e']
axes = [ax1, ax2, ax3, ax5, ax6]

# Custom (x, y) positions for each label in axis coordinates
label_positions = [
    (-0.15, 1.075),  # ax1
    (0.1, 1),   # ax2
    (0.1, 1),  # ax3
    (0.1, 1),  # ax4
    (0.1, 1),  # ax5
    (0.1, 1),  # ax6
]

for ax, label, (x, y) in zip(axes, labels, label_positions):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=12,
        fontweight='bold'
    )




os.makedirs(os.path.join(Fig_dir, 'figure2'), exist_ok=True)
fig2.savefig(os.path.join(Fig_dir, 'figure2/figure2.png'), dpi=300)
fig2.savefig(os.path.join(Fig_dir, 'figure2/figure2.pdf'), dpi=300)
fig2.savefig(os.path.join(Fig_dir, 'figure2/figure2.eps'), dpi=300)
fig2.savefig(os.path.join(Fig_dir, 'figure2/figure2.svg'), dpi=300)

##################################################################

print('making figure 3')

# Create a figure with GridSpec
fig3 = plt.figure(figsize=(7.04724/2, 8))
gs3 =gridspec.GridSpec(4,2,
                       hspace=0.005, wspace=0.2,
                       left=0.05, bottom=0.062, right=0.99, top=0.985,
                       )


# Load model vel results
vx_path = os.path.join(IS_dir, 'results/Vx.npy')
vy_path = os.path.join(IS_dir, 'results/Vy.npy')
vx = np.load(vx_path)
vy = np.load(vy_path)
mean_vx = np.mean(vx, axis=1)
mean_vy = np.mean(vy, axis=1)
# idx the correct observation period
#idx1 = np.argmin(np.abs(tt - 2018.5))
#idx2 = np.argmin(np.abs(tt - 2019))
idx1 = 1903
idx2 = 1950
anomaly_vx = np.mean(vx[:, idx1:idx2+1], axis=1) - mean_vx
anomaly_vy = np.mean(vy[:, idx1:idx2+1], axis=1) - mean_vy

# Load observed vel results
Obs_dir = '../models/velanom'
obs_mean_vx_path = os.path.join(Obs_dir, 'Uavg.npy')
obs_lake_vx_path = os.path.join(Obs_dir, 'Ulake.npy')
obs_mean_vy_path = os.path.join(Obs_dir, 'Vavg.npy')
obs_lake_vy_path = os.path.join(Obs_dir, 'Vlake.npy')
obs_mean_vx = np.load(obs_mean_vx_path)
obs_lake_vx = np.load(obs_lake_vx_path)
obs_mean_vy = np.load(obs_mean_vy_path)
obs_lake_vy = np.load(obs_lake_vy_path)


# set view
xmin = -232.5 * 1e3
xmax = -220 * 1e3
ymin = -2498 * 1e3
ymax = -2489 * 1e3

# Modelled data
# First plot
ax1 = fig3.add_subplot(gs3[0, 0])

tric1 = ax1.tripcolor(meshtri, mean_vx, cmap=cmocean.cm.tempo)
ax1.set_xlim([xmin, xmax])
ax1.set_ylim([ymin, ymax])
ax1.set_axis_off()
ax1.set_aspect('equal')

# Add colorbar
cax1 = ax1.inset_axes((0, 0.1, 1, 0.05))
cbar1 = fig3.colorbar(tric1, shrink=0.5, pad=0.02, cax=cax1, orientation='horizontal')
cbar1.set_label('mean $u_{\mathrm{b}}$ [m a$^{-1}$]', fontsize=10)

# Second plot
ax2 = fig3.add_subplot(gs3[0, 1])

tric2 = ax2.tripcolor(meshtri, mean_vy, cmap=cmocean.cm.tempo)
ax2.set_xlim([xmin, xmax])
ax2.set_ylim([ymin, ymax])
ax2.set_axis_off()
ax2.set_aspect('equal')

# Add colorbar
cax2 = ax2.inset_axes((0, 0.1, 1, 0.05))
cbar2 = fig3.colorbar(tric2, shrink=0.5, pad=0.02, cax=cax2, orientation='horizontal')
cbar2.set_label('mean $v_{\mathrm{b}}$ [m a$^{-1}$]', fontsize=10)

# Third plot
ax3 = fig3.add_subplot(gs3[1, 0])

# Center the colormap on 0 by setting vmin and vmax symmetrically
umin = -max(abs(anomaly_vx.min()), abs(anomaly_vx.max()))
umax = max(abs(anomaly_vx.min()), abs(anomaly_vx.max()))
tric3 = ax3.tripcolor(meshtri, anomaly_vx, cmap=cmocean.cm.balance, vmin=umin, vmax=umax)
ax3.set_xlim([xmin, xmax])
ax3.set_ylim([ymin, ymax])
ax3.set_axis_off()
ax3.set_aspect('equal')

# Add colorbar
cax3 = ax3.inset_axes((0, 0.1, 1, 0.05))
cbar3 = fig3.colorbar(tric3, shrink=0.5, pad=0.02, cax=cax3, orientation='horizontal')
cbar3.set_label('Lake drain anomaly $u_{\mathrm{b}}$ [m a$^{-1}$]', fontsize=10)

# Fourth plot
ax4 = fig3.add_subplot(gs3[1, 1])

# Center the colormap on 0 by setting vmin and vmax symmetrically
vmin = -max(abs(anomaly_vy.min()), abs(anomaly_vy.max()))
vmax = max(abs(anomaly_vy.min()), abs(anomaly_vy.max()))
tric4 = ax4.tripcolor(meshtri, anomaly_vy, cmap=cmocean.cm.balance, vmin=vmin, vmax=vmax)

ax4.set_xlim([xmin, xmax])
ax4.set_ylim([ymin, ymax])
ax4.set_axis_off()
ax4.set_aspect('equal')

# Add colorbar
cax4 = ax4.inset_axes((0, 0.1, 1, 0.05))
cbar4 = fig3.colorbar(tric4, shrink=0.5, pad=0.02, cax=cax4, orientation='horizontal')
cbar4.set_label('Lake drain anomaly $v_{\mathrm{b}}$ [m a$^{-1}$]', fontsize=10)

# Observed data
# First plot
ax5 = fig3.add_subplot(gs3[2, 0])

tric5 = ax5.tripcolor(meshtri, obs_mean_vx, cmap=cmocean.cm.tempo)
ax5.set_xlim([xmin, xmax])
ax5.set_ylim([ymin, ymax])
ax5.set_axis_off()
ax5.set_aspect('equal')

# Add colorbar
cax5 = ax5.inset_axes((0, 0.1, 1, 0.05))
cbar5 = fig3.colorbar(tric5, shrink=0.5, pad=0.02, cax=cax5, orientation='horizontal')
cbar5.set_label('mean $u_{\mathrm{b}}$ [m a$^{-1}$]', fontsize=10)

# Second plot
ax6 = fig3.add_subplot(gs3[2, 1])

tric6 = ax6.tripcolor(meshtri, obs_mean_vy, cmap=cmocean.cm.tempo)
ax6.set_xlim([xmin, xmax])
ax6.set_ylim([ymin, ymax])
ax6.set_axis_off()
ax6.set_aspect('equal')

# Add colorbar
cax6 = ax6.inset_axes((0, 0.1, 1, 0.05))
cbar6 = fig3.colorbar(tric6, shrink=0.5, pad=0.02, cax=cax6, orientation='horizontal')
cbar6.set_label('mean $v_{\mathrm{b}}$ [m a$^{-1}$]', fontsize=10)

# Third plot
ax7 = fig3.add_subplot(gs3[3, 0])

# Center the colormap on 0 by setting vmin and vmax symmetrically
anomaly_vx_obs = obs_lake_vx - obs_mean_vx
umin_obs = -max(abs(anomaly_vx_obs.min()), abs(anomaly_vx_obs.max()))
umax_obs = max(abs(anomaly_vx_obs.min()), abs(anomaly_vx_obs.max()))
tric7 = ax7.tripcolor(meshtri, anomaly_vx_obs, cmap=cmocean.cm.balance, vmin=umin_obs, vmax=umax_obs)
ax7.set_xlim([xmin, xmax])
ax7.set_ylim([ymin, ymax])
ax7.set_axis_off()
ax7.set_aspect('equal')

# Add colorbar
cax7 = ax7.inset_axes((0, 0.1, 1, 0.05))
cbar7 = fig3.colorbar(tric7, shrink=0.5, pad=0.02, cax=cax7, orientation='horizontal')
cbar3.set_label('Lake drain anomaly $u_{\mathrm{b}}$ [m a$^{-1}$]', fontsize=10)

# Fourth plot
ax8 = fig3.add_subplot(gs3[3, 1])

# Center the colormap on 0 by setting vmin and vmax symmetrically
anomaly_vy_obs = obs_lake_vy - obs_mean_vy
vmin_obs = -max(abs(anomaly_vy_obs.min()), abs(anomaly_vy_obs.max()))
vmax_obs = max(abs(anomaly_vy_obs.min()), abs(anomaly_vy_obs.max()))
tric8 = ax8.tripcolor(meshtri, anomaly_vy_obs, cmap=cmocean.cm.balance, vmin=vmin_obs, vmax=vmax_obs)

ax8.set_xlim([xmin, xmax])
ax8.set_ylim([ymin, ymax])
ax8.set_axis_off()
ax8.set_aspect('equal')

# Add colorbar
cax8 = ax8.inset_axes((0, 0.1, 1, 0.05))
cbar8 = fig3.colorbar(tric8, shrink=0.5, pad=0.02, cax=cax8, orientation='horizontal')
cbar8.set_label('Lake drain anomaly $v_{\mathrm{b}}$ [m a$^{-1}$]', fontsize=10)


os.makedirs(os.path.join(Fig_dir, 'figure3'), exist_ok=True)
fig3.savefig(os.path.join(Fig_dir, 'figure3/figure3.png'), dpi=300)
fig3.savefig(os.path.join(Fig_dir, 'figure3/figure3.pdf'), dpi=300)
fig3.savefig(os.path.join(Fig_dir, 'figure3/figure3.eps'), dpi=300)
fig3.savefig(os.path.join(Fig_dir, 'figure3/figure3.svg'), dpi=300)