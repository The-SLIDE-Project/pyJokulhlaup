"""
Extra tools and utils 
"""

from pydoc import importfile

import time
import csv

import numpy as np
from scipy import stats

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