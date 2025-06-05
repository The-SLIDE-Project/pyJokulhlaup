"""
Make ISSM meshes
"""
import os
import sys
import pickle
ISSM_DIR = os.getenv('ISSM_DIR')
sys.path.append(os.path.join(ISSM_DIR, 'src/m/dev/'))
import devpath
from issmversion import issmversion
from model import model
from meshconvert import meshconvert
from solve import solve
from setmask import setmask
from parameterize import parameterize
from bamg import *
from triangle import *
from plotmodel import *
from GetAreas import GetAreas

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec

import cmocean

from src.utils import reorder_edges

## Point to domain outline file
outline ='square_domain.exp'
meshfile = 'synthetic_mesh.pkl'
min_length = 100

## Mesh characteristics
md = model()
md = triangle(md, outline, min_length)
print('Made mesh with numberofvertices:', md.mesh.numberofvertices)

# Compute the nodes connected to each edge
connect_edge = reorder_edges(md)

# Compute edge lengths
x0 = md.mesh.x[connect_edge[:,0]]
x1 = md.mesh.x[connect_edge[:,1]]
dx = x1 - x0
y0 = md.mesh.y[connect_edge[:,0]]
y1 = md.mesh.y[connect_edge[:,1]]
dy = y1 - y0
edge_length = np.sqrt(dx**2 + dy**2)

pos = np.where(np.logical_and(
    md.mesh.vertexonboundary,
    md.mesh.x==np.max(md.mesh.x, axis=-1)))

# Print the position of the vertices on the boundary where x is at its maximum
print(pos)
print(md.mesh.y[pos])

# Find the position of the lake connection
lakepos = 77


print(md.mesh)
areas = GetAreas(md.mesh.elements, md.mesh.x, md.mesh.y)


if os.path.exists(meshfile):
    os.remove(meshfile)

meshdict = {}
meshdict['x'] = md.mesh.x
meshdict['y'] = md.mesh.y
meshdict['elements'] = md.mesh.elements
meshdict['connect_edge'] = connect_edge
meshdict['edge_length'] = edge_length
meshdict['area'] = areas
meshdict['vertexonboundary'] = md.mesh.vertexonboundary
meshdict['numberofelements'] = md.mesh.numberofelements
meshdict['numberofvertices'] = md.mesh.numberofvertices
meshdict['lakepos'] = lakepos

# Write dictionary
with open(meshfile, 'wb') as mesh:
    pickle.dump(meshdict, mesh)


fig, ax = plt.subplots(figsize=(8, 3))
mtri = Triangulation(md.mesh.x, md.mesh.y, md.mesh.elements-1)
ax.tripcolor(mtri, 0*md.mesh.x, facecolor='none', edgecolor='k')

x_pos = md.mesh.x[lakepos]
y_pos = md.mesh.y[lakepos]
# Plot red dot at the lake position
ax.plot(x_pos, y_pos, 'blue', markersize=5, label='Lake center')

x_pos = md.mesh.x[lakepos]
y_pos = md.mesh.y[lakepos]
# Annotate with text *outside* the plot to the right
ax.annotate(
    'Lake \noutlet',
    xy=(x_pos, y_pos),  # Point to annotate
    xycoords='data',
    xytext=(1.02, 0.5),  # Position outside to the right, in axes fraction
    textcoords='axes fraction',
    arrowprops=dict(facecolor='black', arrowstyle='->'),
    fontsize=10,
    color='black',
    ha='left',
    va='center',
    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='white', lw=1)
)

ax.set_aspect('equal')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Synthetic Mesh')
ax.set_xlim([0, 10.e3])
ax.set_ylim([0, 3e3])
fig.savefig('synthetic_mesh.png', dpi=300)

