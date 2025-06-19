"""
Extra tools and utils 
"""

from pydoc import importfile

import time
import csv

import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from cmocean import cm
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt


def import_table(table_path):

    """ import a csv table given path/to/table.csv
    Parameters
    ----------
    table_path : str
        Path to the table file, e.g. 'path/to/table.csv'
    Returns
    -------
    table : dict
        Dictionary with column names as keys and lists of column values
        as values.

    Raises
    -------
        ValueError: If any row has a different number of columns than the header.
    """
    n_column = 13 # number of columns in the table

    filepath = table_path  # Define filepath as table_path
    expected_columns = 13  # Define expected_columns

    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        if len(header) != expected_columns:
            raise ValueError(f"Expected {expected_columns} columns, but found {len(header)} in header.")

        data = {col: [] for col in header}
        
        for line_number, row in enumerate(reader, start=2):
            if len(row) != expected_columns:
                raise ValueError(f"Row {line_number} has {len(row)} columns, expected {expected_columns}.")
            for col, val in zip(header, row):
                data[col].append(val)

    return data



def width_average(mesh, x, dx=2):
    """Width-average values defined on a triangular mesh.
    
    For array x with shape (nx, nt), average in specified dx bins
    to leave the spatial dimension representing x only.
    
    Parameters
    ----------
    mesh : dictionary with keys 'x' and 'y' providing lists of
           x and y coordinates
    
    x : array with shape (nx, nt)
        Target to average over y
    
    dx : float, optional (default 2)
         Bin width in first dimension
    
    Returns
    -------
    array averaged over y in dx-width bins
    """
    xedge = np.arange(0, 100+dx, dx)
    xmid = 0.5*(xedge[1:] + xedge[:-1])
    xavg = np.zeros((len(xmid), x.shape[1]))
    for i in range(len(xavg)):
        xi = xmid[i]
        mask = np.abs(mesh['x']/1e3 - xi)<dx/2
        xavg[i] = np.nanmean(x[mask,:],axis=0)
    return xavg, xedge

def reorder_edges(md):
    """
    Reorder edges from ISSM mesh md to make it easier to relate 
    channel discharge to the nodes of the mesh. Critical for plotting
    channel discharge and computing discharge across fluxgates.

    Based on plotchannels.m from ISSM model source.

    Parameters
    ----------
    md : model instance

    Returns:
    edges : (md.mesh.numberofedges, 2) array specifying
            ordered nodes connected to each edge
    """
    maxnbf = 3*md.mesh.numberofelements
    edges = np.zeros((maxnbf, 3)).astype(int)
    exchange = np.zeros(maxnbf).astype(int)

    head_minv = -1*np.ones(md.mesh.numberofvertices).astype(int)
    next_face = np.zeros(maxnbf).astype(int)
    nbf = 0
    for i in range(md.mesh.numberofelements):
        for j in range(3):
            v1 = md.mesh.elements[i,j]-1
            if j==2:
                v2 = md.mesh.elements[i,0]-1
            else:
                v2 = md.mesh.elements[i,j+1]-1
            
            if v2<v1:
                v3 = v2
                v2 = v1
                v1 = v3
            
            exists = False
            e = head_minv[v1]
            while e!=-1:
                if edges[e, 1]==v2:
                    exists=True
                    break
                e = next_face[e]
            
            if not exists:
                edges[nbf,0] = v1
                edges[nbf,1] = v2
                edges[nbf,2] = i
                if v1!=md.mesh.elements[i,j]-1:
                    exchange[nbf] = 1
                
                next_face[nbf] = head_minv[v1]
                head_minv[v1] = nbf
                nbf = nbf+1

    edges = edges[:nbf]
    pos = np.where(exchange==1)[0]
    v3 = edges[pos, 0]
    edges[pos, 0] = edges[pos, 1]
    edges[pos, 1] = v3

    edges = edges[:, :2]
    return edges

def plotchannels(mesh,x, **kwargs):
    """
    Emulates the functionality of MATLAB plotchannels.m function to plot GlaDS channel data on a 2D mesh

    Parameters: 
        mesh: dict with 'x', 'y', and 'connect_edge' keys
        x: values on edges, shape (n_edges, )
        **kwargs:
            ax: matplotlib axis to plot on (optional)
            min: minimum value to display (optional, default 1)
            max: maximum value to display (optional, default to the max of the data)
            colormap: str or matplotlib colormap (optional, default cmocean.cm.ice_r)
            linewidth: width of the edges in the plot (optional, default 1.0)
            quiver: boolean, if True, plot arrows along edges (optional, default False)
            arrow_scale: scale for arrows if quiver is True (optional, default 1.0)

    Returns:
        tuple: A tuple containing:
            - matplotlib.axes.Axes: The axes object containing the plot.
            - matplotlib.cm.ScalarMappable: The ScalarMappable object that contains
              the colormap and normalization, used for creating the colorbar.
    
    """

    # Process options from kwargs
    ax = kwargs.get('ax', None)
    is_quiver = kwargs.get('quiver', False)
    linewidth = kwargs.get('linewidth', 1)
    cmap_name = kwargs.get('colormap', cm.ice_r)
    arrow_scale = kwargs.get('arrow_scale', 1.0)

    # Use absolute values for colormap if plotting discharge with arrows
    level = np.abs(x) if is_quiver else x

    # Get min/max for color normalization
    vmin = kwargs.get('min', np.min(level))
    vmax = kwargs.get('max', np.max(level))
    # Create mask for values above or equal to vmin
    valid_mask = level >= vmin

    # Setup Axes and Colormap
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal')
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        ax.set_title("Channel Plot")

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Prepare edges for plotting
    edges = mesh['connect_edge']
    x_coords = mesh['x']
    y_coords = mesh['y']
    # Get vertex coordinates for each edge
    v1_idx, v2_idx = edges[:, 0], edges[:, 1]

    x1, x2 = x_coords[v1_idx], x_coords[v2_idx]
    y1, y2 = y_coords[v1_idx], y_coords[v2_idx]

    # Create full segments
    segments_all = np.array([[x1, y1], [x2, y2]]).transpose(2, 0, 1)

    # Apply mask to segments and levels
    segments = segments_all[valid_mask]
    level_valid = level[valid_mask]

    # Create a LineCollection only for valid segments
    line_collection = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth)
    line_collection.set_array(level_valid)

    ax.add_collection(line_collection)

    # Plot Quivers (Arrows) if requested ---
    if is_quiver:
        arrow_verts = []
        # Filter out zero-length edges to avoid division by zero
        edge_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        valid_edges = edge_len > 1e-9

        # Midpoints of valid edges
        xq = np.mean([x1[valid_edges], x2[valid_edges]], axis=0)
        yq = np.mean([y1[valid_edges], y2[valid_edges]], axis=0)
        
        # Direction vectors (normalized)
        tx = (x2[valid_edges] - x1[valid_edges]) / edge_len[valid_edges]
        ty = (y2[valid_edges] - y1[valid_edges]) / edge_len[valid_edges]
        
        # Apply data sign for flow direction and scale for arrow size
        arrow_size = edge_len[valid_edges].mean() * 0.1 * arrow_scale
        flow_sign = np.sign(x[valid_edges])
        tx = tx * flow_sign * arrow_size
        ty = ty * flow_sign * arrow_size
        
        # Perpendicular vectors for arrowhead base
        px = -ty
        py = tx

        # Define vertices for each arrow (as a triangle)
        # Vertices: [tip, base_left, base_right]
        v1 = np.array([xq + tx, yq + ty]).T
        v2 = np.array([xq - tx + 0.5 * px, yq - ty + 0.5 * py]).T
        v3 = np.array([xq - tx - 0.5 * px, yq - ty - 0.5 * py]).T
        arrow_verts = np.dstack([v1, v2, v3]).transpose(0, 2, 1)

        # Create a PolyCollection for efficient arrow plotting
        arrow_collection = PolyCollection(
            arrow_verts, 
            cmap=cmap, 
            norm=norm, 
            edgecolors='none'
        )
        # Set the data values to be mapped to colors
        arrow_collection.set_array(level[valid_edges][level[valid_edges] >= vmin])
        ax.add_collection(arrow_collection)

    # Add a colorbar to the plot
    ax.set_aspect('equal')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(level)
    #plt.colorbar(sm, ax=ax, label='Channel Area / Discharge Magnitude')
    
    ax.autoscale_view()
    return ax,sm

def lakeheightminmax(x):
    """
    Given a 1D np.array describing an evolving quantity X over time,
    return the index of peaks and troughs
    Parameters
    ----------
    X : np.array of time-evolving parameter
    
    Returns
    -------
    peaks : np.array
        Indices of local maxima in X
    troughs : np.array
        Indices of local minima in X
    """

    peaks, properties = find_peaks(x, distance=5, prominence=1)

    if len(peaks) == 0:
        print("No peaks found in the data.")
        peaks = np.argmax(x) # fallback to the maximum index

    print("Peak indices:", peaks)


    troughs, properties = find_peaks(-x, distance=5, prominence=1)
    if len(troughs) == 0:
        print("no troughs detected.")
        troughs = [np.argmin(x[peaks[0]:])] # fallback to the end of the array

    print("Trough indices:", troughs)
    return peaks, troughs

    
