""" 
Compute synthetic surface and bed elevations
"""

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec
import cmocean

with open('synthetic_mesh.pkl', 'rb') as meshin:
    mesh = pickle.load(meshin)

#thickness = 25+(1.2*( np.sqrt(mesh['x'])))
thickness = 25+np.sqrt(1.1*mesh['x'])
bed = (mesh['x'] * np.sin(0.005))
surf = bed + thickness
print('Min surface is ~', f"{np.min(surf):.2f}")
print('Mean surface ~', f"{np.mean(surf):.2f}")    
print('Max surface ~', f"{np.max(surf):.2f}")

slope_rad = np.arctan((np.max(surf) - np.min(surf)) / np.max(mesh['x']))
slope_deg = np.rad2deg(slope_rad)
print("Average surface slope is ~", f"{slope_deg:.2f}", "degrees")

print('Min thickness is ~', f"{np.min(surf-bed):.2f}")
print('Mean thickness ~', f"{np.mean(surf-bed):.2f}")    
print('Max thickness ~', f"{np.max(surf-bed):.2f}")


np.save('synthetic_surface.npy', surf)
np.save('synthetic_bed.npy', bed)


mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
# Create figure with GridSpec
fig = plt.figure(figsize=(9, 9))
gs = GridSpec(3, 2, width_ratios=[20, 1], height_ratios=[1, 1, 1], wspace=0.05, hspace=0.15)

# --- Top plot: Surface ---
ax1 = fig.add_subplot(gs[0, 0])
tric1 = ax1.tripcolor(mtri, surf, shading='gouraud', cmap=cmocean.cm.rain)
ax1.set_aspect('equal')
ax1.set_xlim([0, 10e3])
ax1.set_ylim([0, 3e3])
ax1.set_xticks(np.linspace(0, 10e3, 7))
ax1.set_xticklabels((ax1.get_xticks() / 1e3).astype(int))
ax1.set_yticks(np.linspace(0, 3e3, 5))
ax1.set_yticklabels((ax1.get_yticks() / 1e3).astype(int))
ax1.set_ylabel('Y (km)')

# Surface colorbar
cax1 = fig.add_subplot(gs[0, 1])
cbar1 = fig.colorbar(tric1, cax=cax1)
cbar1.set_label('Surface Elevation (m)')

# --- Bottom plot: Thickness ---
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
tric2 = ax2.tripcolor(mtri, surf-bed, shading='gouraud', cmap=cmocean.cm.ice_r)
ax2.set_aspect('equal')
ax2.set_xlim([0, 10e3])
ax2.set_ylim([0, 3e3])
ax2.set_xticks(np.linspace(0, 10e3, 7))
ax2.set_xticklabels((ax2.get_xticks() / 1e3).astype(int))
ax2.set_yticks(np.linspace(0, 3e3, 5))
ax2.set_yticklabels((ax2.get_yticks() / 1e3).astype(int))

ax2.set_ylabel('Y (km)')


# Thickness colorbar
cax2 = fig.add_subplot(gs[1, 1])
cbar2 = fig.colorbar(tric2, cax=cax2)
cbar2.set_label('Ice Thickness (m)')

# --- Bottom plot: bed ---
ax3 = fig.add_subplot(gs[2, 0])
tric3 = ax3.tripcolor(mtri, bed, shading='gouraud', cmap=cmocean.cm.deep)
ax3.set_aspect('equal')
ax3.set_xlim([0, 10e3])
ax3.set_ylim([0, 3e3])
ax3.set_xticks(np.linspace(0, 10e3, 7))
ax3.set_xticklabels((ax3.get_xticks() / 1e3).astype(int))
ax3.set_yticks(np.linspace(0, 3e3, 5))
ax3.set_yticklabels((ax3.get_yticks() / 1e3).astype(int))
ax3.set_ylabel('Y (km)')
ax3.set_xlabel('X (km)')

# Surface colorbar
cax3 = fig.add_subplot(gs[2, 1])
cbar3 = fig.colorbar(tric3, cax=cax3)
cbar3.set_label('Bed Elevation (m)')

# Save figure
fig.savefig('synthetic_surface_thickness_bed.png', dpi=300)