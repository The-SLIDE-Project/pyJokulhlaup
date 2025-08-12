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

IS_dir = '../models/6-TrQin-transient-20-06-2025-08-19'
print(os.getcwd())

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

# Create a figure with GridSpec
fig1 = plt.figure(figsize=(7.04724, 8))
gs1 =gridspec.GridSpec(5,3,
                       hspace=0.35, wspace=0.05,
                       left=0.1, bottom=0.095, right=0.95, top=0.975,
                       width_ratios=[0.01, 1, 0.01],
                       height_ratios=[0.05,1, 0.4,0.4, 0.4],
                       )


ax1 = fig1.add_subplot(gs1[1, :])
inset_width = 0.475
inset_height = 0.475
ax2 = ax1.inset_axes([-0.14, 0.78, inset_width, inset_height])  # top-left corner of ax1
ax3 = ax1.inset_axes([0.37, 0.72, 0.57, 0.57])

ax4 = fig1.add_subplot(gs1[2, 1])
ax5 = fig1.add_subplot(gs1[3, 1])
ax6 = fig1.add_subplot(gs1[4, 1])

# --- Top plot: Bed topography ---

tric1 = ax1.tripcolor(meshtri, bed, cmap=cmocean.cm.deep,edgecolors='w', linewidths=0.01)
#ax1, sm1 = plotchannels(mesh, np.abs(Qc[:,2000]), ax=ax1, min=1,quiver=False,linewidth=0.5)
ax3.set_aspect('equal')
ax1.set_xlim([-232.5*1e3, -200*1e3])
ax1.set_ylim([-2502*1e3, -2488*1e3])
ax1.set_axis_off()

Qcmin = 1
Qcmax = 100
cnorm = matplotlib.colors.Normalize(vmin=Qcmin, vmax=Qcmax)
Q_arr = Qc[:,2000]

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
    lc_colors.append(cmocean.cm.gray_r(cnorm(Qi)))
lc = LineCollection(lc_xy, colors=lc_colors, linewidths=lc_lw,
    capstyle='round')
lc.set(rasterized=True)
ax1.add_collection(lc)
cax2 = ax1.inset_axes((0.4, 0.065, 0.25, 0.03))
cb2 = fig1.colorbar(matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmocean.cm.gray_r),
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

# Add lake outlet and spc outlet
lakepos = np.where(lakemask == 1)

spcpos  = 10655

for idx in lakepos[0]:  # use lakepos[0] because np.where returns a tuple
    ax1.plot(x[idx], y[idx], marker='*', color='blue', markersize=5)

ax1.plot(x[spcpos], y[spcpos], marker='o', color='red', markersize=5)

# Create dummy handles for legend
lake_handle = plt.Line2D([], [], marker='*', color='blue', linestyle='None',
                         markersize=6, label='Lake outlet')
spc_handle  = plt.Line2D([], [], marker='o', color='red', linestyle='None',
                         markersize=6, label='SPC outlet')

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




# Greenland map
img = matplotlib.image.imread('Greenland_IS_Location.png')
# Display in ax2
ax2.imshow(img)
ax2.axis('off')  # Optional: hide axis ticks and labels
ax2.set_facecolor('white')


tric3 = ax3.tripcolor(meshtri, bed, cmap=cmocean.cm.deep,edgecolors='w', linewidths=0.0005)
ax3.set_aspect('equal')
ax3.set_axis_off()
# Get axis limits of ax1
x0, x1 = ax1.get_xlim()
y0, y1 = ax1.get_ylim()
cax = ax3.inset_axes((1, 0.35, 0.025, 0.5))
cbar = fig1.colorbar(tric3, shrink=0.5, pad=0.02, cax=cax, orientation='vertical')
cbar.set_label('$z_{\mathrm{b}}$ [m]', fontsize=8, labelpad=0.5)
cbar.set_ticks([500, 0, -300])

# Change tick label font size
cbar.ax.tick_params(labelsize=8)

# Create a rectangle using ax1's extent in ax3
rect = matplotlib.patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                 edgecolor='black', facecolor='none', linewidth=1, linestyle='-')
ax3.add_patch(rect)
# Add label 'c' at top-left of the extent box
ax3.text(x0-1500, y1+1500, 'c', fontsize=8,
         verticalalignment='top', horizontalalignment='right',
         color='black', bbox=dict(facecolor='none', edgecolor='none'))

# --- thrid row, SMB ---
percentile_runoff = np.percentile(runoff, 95, axis=0)


ax4.plot(runoff_tt,percentile_runoff, color='black', linewidth=.25)
ax4.set_ylabel('P95 Runoff \n[m w.e. a$^{-1}$]',fontsize=8,labelpad=0.5)
ax4.set_xticklabels([])  # Hide the x-axis ticks
xticks = np.arange(np.floor(np.min(runoff_tt)), np.ceil(np.max(runoff_tt)) + 1, 1)
ax4.set_xticks(xticks)
ax4.set_xlabel('')  # Hide the x-axis label
ax4.margins(x=0,y=0)

# --- fourth row, lh ---
# Load lake height record
datapath = '/Volumes/ajh24/GlaDS/GREENLAND_LAKES/Data/LakeMeasurements/IceMarginalLakeHeight.csv'

# Load CSV manually with numpy.genfromtxt
# Assuming the CSV has headers and columns: Medc, Datec, StDevc, Source (Datec format: yyyy-mm-dd)
data = np.genfromtxt(datapath, delimiter=',', names=True, dtype=None, encoding=None)

# Extract columns
lhm = data['Medc']
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


ax5.plot(dectime, lhm-266, color='grey', linewidth=0.5)
ax5.fill_between(dectime, lhm-266-err, lhm-266+err, color='grey', alpha=0.5, label='$l_{\mathrm{h}}$ (meas.)')


ax5.plot(tt,np.max(lh,axis=0), color='black', linewidth=1, label='$l_{\mathrm{h}}$ (model)')
ax5.set_ylabel('$l_\mathrm{h}$ [m]',fontsize=8,labelpad=0.5)
ax5.set_xticklabels([])  # Hide the x-axis ticks
xticks = np.arange(np.floor(np.min(runoff_tt)), np.ceil(np.max(runoff_tt)) + 1, 1)
ax5.set_xticks(xticks)
ax5.set_xlabel('')  # Hide the x-axis label
ax5.set_xlim([np.floor(np.min(runoff_tt)), np.ceil(np.max(runoff_tt))])
xticks = np.arange(np.floor(np.min(runoff_tt)), np.ceil(np.max(runoff_tt)) + 1, 1)
ax5.margins(x=0,y=0.05)
ax5.legend(loc='upper right', fontsize=6, frameon=True, handlelength=0.5, handletextpad=0.2,
           labelspacing=0.1, borderpad=0.1, borderaxespad=0.1,facecolor='white',edgecolor='none')

# --- fifth row, Qr ---
ax6.plot(tt,np.sum(Qr,axis=0), color='black', linewidth=1)
ax6.set_ylabel('$Q_\mathrm{r}$ [m]',fontsize=8,labelpad=0.5)
ax6.xaxis.set_major_locator(MultipleLocator(3))         # Tick every 3 years
ax6.xaxis.set_major_formatter(FormatStrFormatter('%d')) # No decimal places
ax6.xaxis.set_minor_locator(MultipleLocator(1))
ax6.set_xlabel('Model time [yrs]',fontsize=8,labelpad=0.5)
ax6.set_xlim([np.floor(np.min(runoff_tt)), np.ceil(np.max(runoff_tt))])
ax6.margins(x=0,y=0.05)

os.makedirs(os.path.join(IS_dir, 'figures/figure1'), exist_ok=True)
fig1.savefig(os.path.join(IS_dir, 'figures/figure1/figure1.png'), dpi=300)
fig1.savefig(os.path.join(IS_dir, 'figures/figure1/figure1.pdf'), dpi=300)
fig1.savefig(os.path.join(IS_dir, 'figures/figure1/figure1.eps'), dpi=300)
fig1.savefig(os.path.join(IS_dir, 'figures/figure1/figure1.svg'), dpi=300)

