"""
Run an ISSM-GlaDS(JOKULHLAUP) job from the command line given a run table and job id.

    Usage: python run_job.py <run_table> <job_id>

run_table is a path to a CSV file containing job submission information.
job_id correponds to a row in the supplied job submission table
"""

"""
Run ISSM-GlaDS simultions using experiment config files from command-line

Usage
-----

    python -m src.run_job [config] [id]

config : path to experiment configuration file
id : integer job number
"""

import os
import sys
import argparse
import numpy as np
import pickle
import socket

ISSM_DIR = os.getenv('ISSM_DIR')
sys.path.append(os.path.join(ISSM_DIR, 'src/m/dev/'))
import devpath
from issmversion import issmversion

from model import model
from meshconvert import meshconvert
from hydrologyglads import *
from SetIceSheetBC import SetIceSheetBC
from setflowequation import setflowequation
from setmask import setmask
from solve import solve
from paterson import paterson
from generic import generic
from bcgslbjacobioptions import bcgslbjacobioptions
from mumpsoptions import mumpsoptions
import matplotlib
import matplotlib.gridspec as gridspec
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from parameterize import parameterize

from src.utils import *

def run_job(run_table, job_id):
    """
    Run a single ISSM-GlaDS(JOKULHLAUP) from the run table at the specified job id.

    Parameters
    ----------
    run_table : str
        Path to the run table CSV file.
    job_id : int
        Job ID corresponding to a row in the run table.

    Returns
    -------
    md: model
        The solved ISSM model object after running the job.
    """
    # Load the run table
    run_table = import_table(run_table)

    # Resolve the requested job against the table ID column (preferred)
    # and fall back to legacy 1-based row indexing when needed.
    nrows = len(run_table['ID'])
    row_idx = None
    if 'ID' in run_table:
        table_ids = np.asarray(run_table['ID']).astype(int)
        matches = np.where(table_ids == int(job_id))[0]
        if matches.size > 0:
            row_idx = int(matches[0])

    if row_idx is None:
        if 1 <= int(job_id) <= nrows:
            row_idx = int(job_id) - 1
            print(f"[WARN] Requested job_id={job_id} not found in 'ID' column; using legacy row index {row_idx + 1}.")
        else:
            raise ValueError(
                f"job_id={job_id} not found in 'ID' column and out of row-index range 1..{nrows}."
            )

    # Get the job parameters for the resolved row
    paramsdict = {}
    paramsdict['k_c'] = float(run_table['$k_{c}$'][row_idx])
    paramsdict['k_s'] = float(run_table['$k_{s}$'][row_idx])
    paramsdict['evr'] = float(run_table['$evr$'][row_idx])
    paramsdict['h_b'] = float(run_table['$h_{b}$'][row_idx])
    paramsdict['l_c'] = float(run_table['$l_{c}$'][row_idx])
    paramsdict['l_s'] = float(run_table['$l_{s}$'][row_idx])
    paramsdict['C'] = float(run_table['friction C'][row_idx])
    paramsdict['p'] = float(run_table['friction p'][row_idx])
    paramsdict['q'] = float(run_table['friction q'][row_idx])
    paramsdict['bed_angle'] = float(run_table['bed angle'][row_idx])
    paramsdict['surface_parabola'] = float(run_table['surface parabola'][row_idx])
    paramsdict['melt_rate'] = float(run_table['$melt rate$'][row_idx])
    paramsdict['Q_in'] = float(run_table['$Q_{in}$'][row_idx])
    paramsdict['s_melt_flag'] = int(run_table['seasonal melt flag'][row_idx])
    paramsdict['Name'] = run_table['Name'][row_idx]
    paramsdict['run_id'] = run_table['ID'][row_idx]
    paramsdict['Notes'] = run_table['Notes'][row_idx]

    # initialize the model
    md = model()
    # Set model name
    md.miscellaneous.name = 'output_run_{}_{}'.format(paramsdict['run_id'], paramsdict['Name'])
    print('Model name:', md.miscellaneous.name)
    

    # read in mesh and pass to ISSM
    meshfile = '../data/geometry/synthetic_mesh.pkl'
    with open(meshfile, 'rb') as meshin:
        mesh = pickle.load(meshin)
    md = meshconvert(md, mesh['elements'], mesh['x'], mesh['y'])
    lakepos = mesh['lakepos']

    # Vectors for convenience
    onevec = np.ones((md.mesh.numberofvertices))

    # Set the geometry
    md.geometry.bed = mesh['x'] * np.sin(paramsdict['bed_angle'])
    md.geometry.thickness = 50 + np.sqrt(paramsdict['surface_parabola'] * mesh['x'])
    md.geometry.surface = md.geometry.bed + md.geometry.thickness
    md.geometry.base = md.geometry.bed      

    # parameterise the model
    print('parameterising model')
    md = parameterize(md, '../default.py')

    #Set model parameters
    md.basalforcings.groundedice_melting_rate = paramsdict['melt_rate'] * onevec
    md.hydrology.elastic_sheet_flag = 0
    md.hydrology.channel_sheet_width = paramsdict['l_c']
    md.hydrology.cavity_spacing = paramsdict['l_s']
    md.hydrology.sheet_conductivity = paramsdict['k_s'] * onevec
    md.hydrology.channel_conductivity = paramsdict['k_c'] * onevec
    md.hydrology.bump_height = paramsdict['h_b'] * onevec
    md.hydrology.englacial_void_ratio = paramsdict['evr'] * onevec
    md.hydrology.lake_Qin[lakepos] = paramsdict['Q_in']


    # Set friction params 
    md.friction.coefficient = paramsdict['C'] * onevec
    md.friction.p = paramsdict['p'] * np.ones((md.mesh.numberofelements))
    md.friction.q = paramsdict['q'] * np.ones((md.mesh.numberofelements))



    # Store the parameters in the model
    resdir = f"RES/output_run_{paramsdict['run_id']}_{paramsdict['Name']}"

    if not os.path.exists(resdir):
        os.makedirs(resdir)
    
    with open(os.path.join(resdir, 'paramsdict.pkl'), 'wb') as paramsinfo:
        pickle.dump(paramsdict, paramsinfo)
    
    # Solve the model
    md = solve(md,'Transient')
    requested_outputs = extract_requested_outputs(md)
    for field in requested_outputs.keys():
        np.save(os.path.join(resdir, '{}.npy'.format(field)), 
           requested_outputs[field])
        
    plot_requested_outputs(requested_outputs,md,paramsdict,resdir)    
    return md,requested_outputs,resdir


def extract_requested_outputs(md):
    """
    Extract arrays of model output fields from ISSM outputs

    Construct a dictionary where the values are ( - , n_timesteps)
    arrays by iterating over the md.results.TransitionSolution
    struct array.

    Parameters
    ----------
    md : model
         Solved ISSM model instance
    
    Returns
    -------
    dict of model output fields
    """
    # Determine the size of TransientSolution dynamically
    imin = 0
    imax = len(md.results.TransientSolution)

    phi_bed = np.vstack(md.materials.rho_freshwater * md.constants.g * md.geometry.bed)
    phi = np.array([ts.HydraulicPotential[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T
    N = np.array([ts.EffectivePressure[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T
    pw = phi - phi_bed
    ff = pw/(N + pw)
    outputs = dict(
        phi = phi,
        N = N,
        ff = ff,
        l_h = np.array([ts.HydrologyLakeHeight[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        Qr = np.array([ts.HydrologyLakeOutletQr[:,0] for ts in md.results.TransientSolution[imin:imax]]).T,
        Qrc = np.array([ts.HydrologyLakeChannelQr[:,0] for ts in md.results.TransientSolution[imin:imax]]).T,
        h_s = np.array([ts.HydrologySheetThickness[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        S = np.array([ts.ChannelArea[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        Qc = np.array([ts.ChannelDischarge[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        qs = np.array([ts.HydrologySheetDischarge[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        tt = np.array([ts.time for ts in md.results.TransientSolution[imin:imax]]).T,
        hydrovx = np.array([ts.HydrologyWaterVx[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        hydrovy = np.array([ts.HydrologyWaterVy[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        vel = np.array([ts.Vel[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        vx = np.array([ts.Vx[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        vy = np.array([ts.Vy[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
    )
    return outputs

def plot_requested_outputs(outputs,md,paramsdict,resdir):
    """
    Plot simple timeseries outputs from the ISSM model.

    Parameters
    ----------
    outputs : dict
        Dictionary containing model output fields.
    """
    # read in mesh
    meshfile = '../data/geometry/synthetic_mesh.pkl'
    with open(meshfile, 'rb') as meshin:
        mesh = pickle.load(meshin)
    lakepos = mesh['lakepos']
    tt = outputs['tt']
    Qc = outputs['Qc']
    Qrc = outputs['Qrc']
    lh = outputs['l_h']
    Qr = outputs['Qr']
    ff = outputs['ff']
    vel = outputs['vel']
    N = outputs['N']

    fig = plt.figure(figsize=(8.27,11.69))
    gs = gridspec.GridSpec(4, 2, figure=fig)

    # --- AX1: Top-left plot (Lake Height and Flux) ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('Lake height')
    ax1.set_xlabel('Time')

    ax1.set_ylabel('Lake Height (m)')
    ax1.plot(tt, lh[lakepos, :], color='black', linestyle='-', label='$l_h$ (Lake Height)')

    # Create a twin axis for Lake Height
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('$Q_{r}$ (m$^3$s$^{-1}$)')
    ax1_twin.plot(tt, Qr[lakepos, :], color='gray', linestyle='-', label='$Q_r$ (Outlet Flux)')

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # --- AX2: Bottom left plot (Flotation Fraction) ---
    ax2 = fig.add_subplot(gs[1, :])  # The ':' makes this subplot span the entire row
    ax2.set_title('Median Flotation Fraction and Ice Velocity')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Flotation Fraction')
    ax2.plot(tt, np.median(ff, axis=0), color='black', linestyle='-', label='Median Flotation Fraction')

    # Create a twin axis for Qr
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel('$Q_{r}$ (m$^3$s$^{-1}$)')
    ax2_twin.plot(tt, Qr[lakepos, :], color='gray', linestyle='-', label='$Q_r$ (Outlet Flux)')

    # Combine legends from both axes
    lines3, labels3 = ax2.get_legend_handles_labels()
    lines4, labels4 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines3 + lines4, labels3 + labels4, loc='upper left')

    # --- AX3: Bottom right plot (Velocity) ---
    ax3 = fig.add_subplot(gs[2, :])
    ax3.set_title('Ice Velocity')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Velocity (m a⁻¹)')
    ax3.plot(tt, np.median(outputs['vel'], axis=0), color='black', linestyle='-', label='Median Velocity')
    
    # Create a twin axis for Qr
    ax3_twin = ax3.twinx()
    ax3_twin.set_ylabel('$Q_{r}$ (m$^3$s$^{-1}$)')
    ax3_twin.plot(tt, Qr[lakepos, :], color='gray', linestyle='-', label='$Q_r$ (Outlet Flux)')


    # Combine legends from both axes
    lines5, labels5 = ax3.get_legend_handles_labels()
    lines6, labels6 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines5 + lines6, labels5 + labels6,loc='upper left')

    # --- AX4: Bottom-right plot (Flux vs Effective Pressure) ---
    ax4 = fig.add_subplot(gs[3:4, 0])
    QrLake = Qr[lakepos, :]
    NLake = N[lakepos, :] / 1e6  # Convert to MPa
    ax4.set_title('Qr vs. N')
    ax4.set_xlabel('Effective Pressure, $N$ (MPa)')
    ax4.set_ylabel('$Q_{r}$ (m$^3$s$^{-1}$)')

    # Plot line in the background
    ax4.plot(NLake, QrLake, alpha=0.5, color='gray')
    # Plot scatter points on top, using a greyscale colormap
    scatter = ax4.scatter(NLake, QrLake, c=tt, cmap='Greys', s=15, edgecolor='none', zorder=10)
    cbar = fig.colorbar(scatter, ax=ax4)
    cbar.set_label('Time')
    # Add vertical dashed line for "Lake empty"
    lake_empty_N = 9.81 * md.materials.rho_ice * md.geometry.thickness[lakepos] / 1e6  # Convert to MPa
    ax4.axvline(x=lake_empty_N, linestyle='--', color='black', label='Lake empty')

    # Add horizontal dashed line for Qr = Qin
    Qr_line = paramsdict['Q_in']
    ax4.axhline(y=Qr_line, linestyle='--', color='black')

    # Annotate "lake draining" and "lake filling"
    mid_x = (NLake.min() + NLake.max()) / 2  # Calculate the middle of the x-axis span
    ax4.text(mid_x, Qr_line + 0.05 * (QrLake.max() - QrLake.min()), 'lake draining', 
        verticalalignment='bottom', horizontalalignment='center')
    ax4.text(mid_x, Qr_line - 0.05 * (QrLake.max() - QrLake.min()), 'lake filling', 
         verticalalignment='top', horizontalalignment='center')



    num_points = len(NLake)
    arrow_indices = [int(num_points * 0.25), int(num_points * 0.5), int(num_points * 0.75)]

    for i in arrow_indices:
        # Get the point for the arrow's head
        x_head, y_head = NLake[i], QrLake[i]
        # Get a preceding point for the arrow's tail to indicate direction
        # A small step-back (e.g., 5 points) makes the arrow clear
        step_back = 5
        if i > step_back:
            x_tail, y_tail = NLake[i - step_back], QrLake[i - step_back]
            ax4.annotate(
                '', # No text is needed for the arrow
                xy=(x_head, y_head),
                xytext=(x_tail, y_tail),
                arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
                zorder=20 # Ensure arrows are drawn on top
            )

    # --- AX5: Bottom-right plot (Veocity distance from lake) ---
    ax5 = fig.add_subplot(gs[3:4, 1])
    ax5.set_title('Distance from Lake and Velocity')
    ax5.set_xlabel('Time [yrs]')
    ax5.set_ylabel('Velocity [m a-1]')

    # Define grayscale colors from black to light gray
    grays = ['0.0', '0.25', '0.5', '0.7', '0.85']  # from black to light gray

    # Width-averaged velocities at different distances
    xavg, xedge = width_average(mesh, vel, 1)

    # Plot from furthest (9–10 km) to closest (1–0 km)
    ax5.plot(tt, xavg[0, :], color=grays[0], linestyle='-', label='9–10 km from lake')
    #ax5.plot(tt, xavg[2, :], color=grays[1], linestyle='-', label='7–8 km from lake')
    ax5.plot(tt, xavg[4, :], color=grays[2], linestyle='-', label='5–6 km from lake')
    #ax5.plot(tt, xavg[6, :], color=grays[3], linestyle='-', label='3–4 km from lake')
    ax5.plot(tt, xavg[9, :], color=grays[4], linestyle='-', label='1–0 km from lake')

    # Combine legends from both axes
    lines7, labels7 = ax5.get_legend_handles_labels()
    ax5.legend(lines7, labels7,loc='upper left')
    
    axes = [ax1, ax2, ax3, ax4, ax5]  # Your existing axes
    labels = ['a', 'b', 'c','d','e']
    
    for ax, label in zip(axes, labels):
        # Panel label in upper-left corner
        ax.text(-0.05, 1, label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='bottom', ha='right')
        
    # Adjust layout automatically to prevent labels/titles from overlapping
    fig.tight_layout()

    # Save figure
    fig.savefig(os.path.join(resdir, 'Summary.png'), dpi=300)
    print(f"Saved figure to {os.path.join(resdir, 'Summary.png')}")

    fig2 = plt.figure(figsize=(8.27,11.69/4))

    ax1 = fig2.add_subplot()
    ax1.set_title('Qr partition')
    ax1.set_xlabel('Time [yrs]')
    ax1.set_ylabel('Flux (m$^3$s$^{-1}$)')
    ax1.plot(tt, Qrc[lakepos, :], color='red', linestyle='-', label='$Q_{rc}$ (Channel Flux)')
    ax1.plot(tt, Qr[lakepos,:] - Qrc[lakepos, :], color='blue', linestyle='-', label='$Q_{rs}$ (Sheet Flux)')
    ax1.plot(tt, Qr[lakepos, :], color='black', linestyle='-', label='$Q_{r}$ (Total Flux)')
    ax1.legend(loc='upper left')
    fig2.tight_layout()
    # Set log scale
    ax1.set_yscale('log')

    # Save figure
    fig2.savefig(os.path.join(resdir, 'Qr_partition.png'), dpi=300)
    print(f"Saved figure to {os.path.join(resdir, 'Qr_partition.png')}")



    """
    # Plot channels during a lake drainage cycle
    # Get index of channel peaks and troughs
    lh_lake = lh[lakepos, :]
    peaks,troughs = lakeheightminmax(lh_lake)
    if len(peaks) == 1 or len(troughs) == 1:
        idx3 = troughs[-1]  # Last trough
        # Find the mid lake height between max lake height and drainage
        mid_value = (lh_lake[peaks[-1]] + lh_lake[troughs[-1]]) / 2
        start = np.min([peaks[-1], troughs[-1]])
        end = np.max([peaks[-1], troughs[-1]])
        idx2 = min(range(start, end + 1), key=lambda i: abs(lh_lake[i] - mid_value))
        idx1 = peaks[-1]  # Last peak

        fig = plt.figure(figsize=(7, 8))
        gs = gridspec.GridSpec(4, 2, width_ratios=[20,1], wspace=0.1, hspace=0.35)
        # --- Top plot: Channel 1 ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1, sm1 = plotchannels(mesh,np.abs(outputs['Qc'][:,idx1]),ax=ax1,min=0.5,quiver=False)
        ax1.set_xlim([0, 10e3])
        ax1.set_ylim([0, 3e3])
        ax1.text(1.01, 0.5, f"{tt[idx1]:.2f} yrs", transform=plt.gca().transAxes,
             ha='left', va='center', fontsize=10)
        cax1 = fig.add_subplot(gs[0, 1])
        cbar1 = fig.colorbar(sm1, cax=cax1)
        #cbar1.set_label('Channel Discharge (m$^3$ s$^{-1}$)')
        # --- Middle plot: Channel 2 ---
        ax2 = fig.add_subplot(gs[1, 0])
        ax2, sm2 = plotchannels(mesh,np.abs(outputs['Qc'][:,idx2]),ax=ax2,min=0.5,quiver=False)
        ax2.set_xlim([0, 10e3])
        ax2.set_ylim([0, 3e3])
        ax2.text(1.01, 0.5, f"{tt[idx2]:.2f} yrs", transform=plt.gca().transAxes,
             ha='left', va='center', fontsize=10)
        ax2.set_ylabel('Y [m]')
        cax2 = fig.add_subplot(gs[1, 1])
        cbar2 = fig.colorbar(sm2, cax=cax2)
        cbar2.set_label('Channel Discharge (m$^3$ s$^{-1}$)')
        # --- Bottom plot: Channel 3 ---
        ax3 = fig.add_subplot(gs[2, 0])
        ax3, sm3 = plotchannels(mesh,np.abs(outputs['Qc'][:,idx3]),ax=ax3,min=0.5,quiver=False)
        ax3.set_xlim([0, 10e3])
        ax3.set_ylim([0, 3e3])
        ax3.text(1.01, 0.5, f"{tt[idx3]:.2f} yrs", transform=plt.gca().transAxes,
             ha='left', va='center', fontsize=10)
        ax3.set_xlabel('X [m]')
        cax3 = fig.add_subplot(gs[2, 1])
        cbar3 = fig.colorbar(sm3, cax=cax3)
        #cbar3.set_label('Channel Discharge (m$^3$ s$^{-1}$)')
        # --- Fourth plot: lake height in idx range
        ax4 = fig.add_subplot(gs[3, 0])
        ax4.plot(tt, lh_lake, color='black', linestyle='-', label='Lake Height')
        ax4.set_xlim([tt[idx1]-1.5, tt[idx3]+1.5])
        ax4.set_ylabel('Lake Height [m]')
        ax4.set_xlabel('model time [yrs]')
        #ax4.set_adjustable('datalim')
        ax4.set_ylim([0, np.max(lh_lake)+50])
        ax4.vlines([tt[idx1], tt[idx2], tt[idx3]], color='gray', linestyles='--', ymin=0, ymax=np.max(lh_lake)+38)
        for i, label in enumerate(['a', 'b', 'c']):
            ax4.text(tt[[idx1, idx2, idx3][i]], np.max(lh_lake)+48, label, color='black', 
                 fontsize=10, fontweight='bold', ha='center', va='top')

        axes = [ax1, ax2, ax3, ax4]  # Your existing axes
        labels = ['a', 'b', 'c', 'd']
        

        for ax, label in zip(axes, labels):
            # Panel label in upper-left corner
            ax.text(-0.1, 1.1, label, transform=ax.transAxes,
                    fontsize=12, fontweight='bold', va='bottom', ha='right')




        # Save figure
        fig.tight_layout()
        fig.savefig(os.path.join(resdir, 'Channel_plot.png'), dpi=300)
    else:
        idx1 = troughs[-2]  # Second last trough
        # Find the mid lake height between index 1 and 2
        mid_value = (lh_lake[peaks[-1]] + lh_lake[troughs[-2]]) / 2
        start = np.min([peaks[-1], troughs[-2]])
        end = np.max([peaks[-1], troughs[-2]])
        idx2 = min(range(start, end + 1), key=lambda i: abs(lh_lake[i] - mid_value))
        idx3 = peaks[-1]  # Last peak
        # Find the mid lake height between index 3 and 5
        mid_value = (lh_lake[peaks[-1]] + lh_lake[troughs[-1]]) / 2
        start = np.min([peaks[-1], troughs[-1]])
        end = np.max([peaks[-1], troughs[-1]])
        idx4 = min(range(start, end + 1), key=lambda i: abs(lh_lake[i] - mid_value))
        idx5 = troughs[-1]  # Last trough

        # Plot channel evolution through a complete lake drainage cycle
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(6, 2, width_ratios=[20,1], wspace=0.1, hspace=0.35)
        # --- Top plot: Channel 1 ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1, sm1 = plotchannels(mesh,np.abs(outputs['Qc'][:,idx1]),ax=ax1,min=0.5,quiver=False)
        ax1.set_xlim([0, 10e3])
        ax1.set_ylim([0, 3e3])
        ax1.text(1.01, 0.5, f"{tt[idx1]:.2f} yrs", transform=plt.gca().transAxes,
             ha='left', va='center', fontsize=10)
        cax1 = fig.add_subplot(gs[0, 1])
        cbar1 = fig.colorbar(sm1, cax=cax1)
        #cbar1.set_label('Channel Discharge (m$^3$ s$^{-1}$)')
        # --- Second plot: Channel 2 ---                                       
        ax2 = fig.add_subplot(gs[1, 0])
        ax2, sm2 = plotchannels(mesh,np.abs(outputs['Qc'][:,idx2]),ax=ax2,min=0.5,quiver=False)
        ax2.set_xlim([0, 10e3])
        ax2.set_ylim([0, 3e3])
        ax2.text(1.01, 0.5, f"{tt[idx2]:.2f} yrs", transform=plt.gca().transAxes,
             ha='left', va='center', fontsize=10)
        #ax2.set_ylabel('Y [m]')
        cax2 = fig.add_subplot(gs[1, 1])
        cbar2 = fig.colorbar(sm2, cax=cax2)
        #cbar2.set_label('Channel Discharge (m$^3$ s$^{-1}$)')
        # --- Third plot: Channel 3 ---                                       
        ax3 = fig.add_subplot(gs[2, 0])
        ax3, sm3 = plotchannels(mesh,np.abs(outputs['Qc'][:,idx3]),ax=ax3,min=0.5,quiver=False)
        ax3.set_xlim([0, 10e3])
        ax3.set_ylim([0, 3e3])
        ax3.text(1.01, 0.5, f"{tt[idx3]:.2f} yrs", transform=plt.gca().transAxes,
             ha='left', va='center', fontsize=10)
        #ax3.set_ylabel('Y [m]')
        cax3 = fig.add_subplot(gs[2, 1])
        cbar3 = fig.colorbar(sm3, cax=cax3)
        cbar3.set_label('Channel Discharge (m$^3$ s$^{-1}$)')
        # --- Fourth plot: Channel 4 ---                                       
        ax4 = fig.add_subplot(gs[3, 0])
        ax4, sm4 = plotchannels(mesh,np.abs(outputs['Qc'][:,idx4]),ax=ax4,min=0.5,quiver=False)
        ax4.set_xlim([0, 10e3])
        ax4.set_ylim([0, 3e3])
        ax4.text(1.01, 0.5, f"{tt[idx4]:.2f} yrs", transform=plt.gca().transAxes,
             ha='left', va='center', fontsize=10)
        ax4.set_ylabel('Y [m]')
        cax4 = fig.add_subplot(gs[3, 1])
        cbar4 = fig.colorbar(sm4, cax=cax4)
        #cbar4.set_label('Channel Discharge (m$^3$ s$^{-1}$)')
        # --- Fifth plot: Channel 5 ---                                       
        ax5 = fig.add_subplot(gs[4, 0])
        ax5, sm5 = plotchannels(mesh,np.abs(outputs['Qc'][:,idx5]),ax=ax5,min=0.5,quiver=False)
        ax5.set_xlim([0, 10e3])
        ax5.set_ylim([0, 3e3])
        ax5.set_ylabel('Y [m]')
        ax5.text(1.01, 0.5, f"{tt[idx5]:.2f} yrs", transform=plt.gca().transAxes,
             ha='left', va='center', fontsize=10)
        ax5.set_ylabel('Y [m]')
        cax5 = fig.add_subplot(gs[4, 1])
        cbar5 = fig.colorbar(sm5, cax=cax5)
        #cbar5.set_label('Channel Discharge (m$^3$ s$^{-1}$)')
        # --- Sixth plot: lake height in idx range
        ax6 = fig.add_subplot(gs[5, 0])
        ax6.plot(tt, lh_lake, color='black', linestyle='-', label='Lake Height')
        ax6.set_xlim([tt[idx1]-1.5, tt[idx5]+1.5])
        ax6.set_ylabel('Lake Height [m]')
        ax6.set_xlabel('model time [yrs]')
        ax6.set_ylim([0, np.max(lh_lake)+50])
        ax4.vlines([tt[idx1], tt[idx2], tt[idx3],tt[idx4],tt[idx5]], color='gray', linestyles='--', ymin=0, ymax=np.max(lh_lake)+38)
        for i, label in enumerate(['a', 'b', 'c','d', 'e']):
            ax4.text(tt[[idx1, idx2, idx3][i]], np.max(lh_lake)+48, label, color='black', 
                 fontsize=10, fontweight='bold', ha='center', va='top')

        axes = [ax1, ax2, ax3, ax4, ax5, ax6]  # Your existing axes
        labels = ['a', 'b', 'c', 'd','e', 'f']
        

        for ax, label in zip(axes, labels):
            # Panel label in upper-left corner
            ax.text(-0.1, 1.1, label, transform=ax.transAxes,
                    fontsize=12, fontweight='bold', va='bottom', ha='right')

        # Save figure
        fig.savefig(os.path.join(resdir, 'Channel_plot.png'), dpi=300)
"""
        
    

    





def main():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('job_table', type=str, help='Path to the run table CSV file')
    parser.add_argument('jobid', type=int, help='Job ID corresponding to a row in the run table')
    args = parser.parse_args()
    md = run_job(args.job_table, args.jobid)

if __name__=='__main__':
    main()



