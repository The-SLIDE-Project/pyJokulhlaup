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
matplotlib.use('TkAgg')
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

    # Get the job parameters for the specified job_id
    paramsdict = {}
    paramsdict['k_c'] = float(run_table['$k_{c}$'][job_id-1])
    paramsdict['k_s'] = float(run_table['$k_{s}$'][job_id-1])
    paramsdict['evr'] = float(run_table['$evr$'][job_id-1])
    paramsdict['h_b'] = float(run_table['$h_{b}$'][job_id-1])
    paramsdict['l_c'] = float(run_table['$l_{c}$'][job_id-1])
    paramsdict['l_s'] = float(run_table['$l_{s}$'][job_id-1])
    paramsdict['melt_rate'] = float(run_table['$melt rate$'][job_id-1])
    paramsdict['Q_in'] = float(run_table['$Q_{in}$'][job_id-1])
    paramsdict['s_melt_flag'] = int(run_table['seasonal melt flag'][job_id-1])
    paramsdict['h_el_flag'] = int(run_table['$h_{el}$ flag'][job_id-1])
    paramsdict['Name'] = run_table['Name'][job_id-1]
    paramsdict['Notes'] = run_table['Notes'][job_id-1]

    # initialize the model
    md = model()

    # read in mesh and pass to ISSM
    meshfile = '../data/geometry/synthetic_mesh.pkl'
    with open(meshfile, 'rb') as meshin:
        mesh = pickle.load(meshin)
    md = meshconvert(md, mesh['elements'], mesh['x'], mesh['y'])

    # Vectors for convenience
    xvec = md.mesh.x
    onevec = 0*xvec + 1
    oneelem = np.ones((md.mesh.numberofelements))
    oneedges = np.ones((md.mesh.numberofedges))

    # parameterise the model
    print('parameterising model')
    md = parameterize(md, '../defaults.py')

    # Forcing setup 
    print('Setting up forcing')
    md.basalforcings.groundedice_melting_rate = paramsdict['melt_rate'] * onevec
    md.basalforcings.geothermalflux = (68./1000.) * onevec

    # Hydrology
    print('Setting up hydrology')
    md.hydrology = hydrologyglads()

    lakepos = mesh['lakepos']

    # flags
    md.hydrology.ischannels = 1
    md.hydrology.isincludesheetthickness = 1
    md.hydrology.creep_open_flag = 0
    md.hydrology.melt_flag = 1
    md.hydrology.istransition = 1
    md.hydrology.elastic_sheet_flag = paramsdict['h_el_flag']

    # constants
    md.hydrology.channel_sheet_width = paramsdict['l_c']
    md.hydrology.cavity_spacing = paramsdict['l_s']
    md.hydrology.omega = 1./2000.
    md.hydrology.sheet_alpha = 5./4.
    md.hydrology.sheet_beta = 3./2.
    md.hydrology.rheology_B_base = paterson(273.15)*onevec
    md.hydrology.moulin_input = 0.0*onevec

    # system conductivity
    md.hydrology.sheet_conductivity = paramsdict['k_s'] * onevec
    md.hydrology.channel_conductivity = paramsdict['k_c'] * onevec
    md.hydrology.bump_height = paramsdict['h_b'] * onevec
    md.hydrology.englacial_void_ratio = paramsdict['evr'] * onevec

    # BC
    md.hydrology.neumannflux = 0. * oneelem
    md.hydrology.spcphi = np.nan * onevec
    #md.stressbalance.spcvx = np.nan * onevec
    #md.stressbalance.spcvy = np.nan * onevec
    pos = np.where(np.logical_and(
        md.mesh.vertexonboundary,
        md.mesh.x == np.min(md.mesh.x, axis=-1)
    ))
    md.hydrology.spcphi[pos] = 5.e5
    #md.stressbalance.spcvx[pos] = 0.0
    #md.stressbalance.spcvy[pos] = 0.0 
    phi_bed = md.constants.g * md.materials.rho_freshwater * md.geometry.bed
    p_ice = md.constants.g * md.materials.rho_ice * md.geometry.thickness

    # Lakes
    md.hydrology.islakes = 1
    md.hydrology.lake_mask = 0* onevec
    md.hydrology.lake_mask[lakepos] = 1
    md.hydrology.num_lakes = np.max(md.hydrology.lake_mask)
    md.hydrology.lake_area = 0. * onevec
    md.hydrology.lake_area[lakepos] = 5e6  # m^2
    md.hydrology.lake_Qin = 0. * onevec
    md.hydrology.lake_Qin[lakepos] = paramsdict['Q_in']

    # Initialization
    md.initialization.lake_depth = 0. * onevec
    md.initialization.lake_depth[lakepos] = 30.  # m
    md.initialization.lake_outletQr = 0. * onevec
    md.initialization.watercolumn = 0.5 * md.hydrology.bump_height
    md.initialization.elastic_sheet = 0. * onevec
    md.initialization.hydraulic_potential = phi_bed + (0.85 * p_ice)
    md.initialization.sheet_discharge = 0. * onevec
    md.initialization.channelarea = 0. * oneedges
    md.initialization.channeldischarge = 0. * oneedges
    
    # requested outputs
    md.hydrology.requested_outputs = [
        'HydraulicPotential',
        'EffectivePressure',
        'HydrologySheetThickness',
        'HydrologySheetDischarge',
        'HydrologyElasticSheetThickness',
        'ChannelDischarge',
        'ChannelArea',
        'HydrologyWaterVx',
        'HydrologyWaterVy',
        'HydrologyLakeHeight',
        'HydrologyLakeOutletQr'
    ]

    # Timestepping
    nyears = 3
    hour = 3600 # seconds
    day = 86400 # seconds
    dt_hours = 1
    outhours = 72
    md.timestepping.time_step = dt_hours * hour/md.constants.yts  # seconds
    md.timestepping.final_time = nyears  # years
    md.settings.output_frequency = outhours / dt_hours  # output every outhours hours

    # Transient 
    md.transient.deactivateall()
    md.transient.ishydrology = True
    md.transient.isstressbalance = True

    # Execution path
    md.cluster = generic('name',socket.gethostname(),'np', 5)
    cwd = os.getcwd()
    expath = os.path.join(cwd, 'TMP/')
    if not os.path.exists(expath):
        os.makedirs(expath)
    md.cluster.executionpath = expath

    print(md.cluster.executionpath)

    # Solver
    md.stressbalance.maxiter = 50
    md.stressbalance.abstol = np.nan
    md.stressbalance.reltol = np.nan
    md.stressbalance.restol = 1.e-3
    md.debug.gprof = 0
    md.debug.profiling = 1
    md.toolkits.DefaultAnalysis = bcgslbjacobioptions()
    md.toolkits.RecoveryAnalysis = mumpsoptions()
    md.verbose.solution = True
    md.miscellaneous.name = 'output_run_{}_{}'.format(job_id, paramsdict['Name'])

    print(md.hydrology)

    # Store the parameters in the model
    resdir = f"RES/output_run_{job_id}_{paramsdict['Name']}"

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
        
    #plot_requested_outputs(requested_outputs,md,paramsdict,resdir)    
    return md,requested_outputs,resdir












    """
    # parameterize the model

    
    md = setmask(md,'','')
    md = setflowequation(md, 'SSA', 'all')
    md = parameterize(md, '../defaults.py')
    md = SetIceSheetBC(md)
    



    md.miscellaneous.name = f"output_run_{job_id}_{paramsdict['Name']}"
    print('Model name:', md.miscellaneous.name)


    # set the parameters
    vertices = np.ones((md.mesh.numberofvertices, 1))
    
    print('Setting GlaDS-Lakes parameters')
    
    lakepos = mesh['lakepos']
    md.hydrology.lake_mask = np.zeros((md.mesh.numberofvertices, 1))
    md.hydrology.lake_mask[lakepos] = 1
    md.hydrology.lake_area = np.zeros((md.mesh.numberofvertices, 1))
    md.hydrology.lake_area[lakepos] = 5e6  # m^2
    md.hydrology.lake_Qin[lakepos] = paramsdict['Q_in']
    md.hydrology.num_lakes = 1

    print('Setting other GlaDS parameters')
    
    md.hydrology.channel_conductivity = paramsdict['k_c'] * vertices
    md.hydrology.sheet_conductivity = paramsdict['k_s'] * vertices
    md.hydrology.englacial_void_ratio = paramsdict['evr'] * vertices
    md.hydrology.bump_height = paramsdict['h_b'] *vertices
    md.hydrology.cavity_spacing = paramsdict['l_s']
    md.hydrology.channel_sheet_width = paramsdict['l_c']
    md.basalforcings.groundedice_melting_rate = paramsdict['melt_rate'] * vertices
    md.hydrology.elastic_sheet_flag = paramsdict['h_el_flag']
    if md.hydrology.elastic_sheet_flag == 1:
        md.hydrology.requested_outputs.append('HydrologyElasticSheetThickness')


    # Set the initialization parameters
    print('Setting initialization parameters')
    md.initialization.watercolumn = 0.5*md.hydrology.bump_height*vertices
    md.initialization.elastic_sheet = 0.*vertices
    md.initialization.channelarea = 0.*np.zeros((md.mesh.numberofedges, 1))
    md.initialization.channeldischarge = 0.*np.zeros((md.mesh.numberofedges, 1))
    md.initialization.sheet_discharge = 0.*vertices
    md.initialization.lake_outletQr = 0.*vertices
    md.initialization.lake_depth = 0.*vertices
    md.initialization.lake_depth[lakepos] = 30.
    phi_bed = md.constants.g*md.materials.rho_freshwater*md.geometry.base
    p_ice = md.constants.g*md.materials.rho_ice*md.geometry.thickness
    md.initialization.hydraulic_potential = phi_bed + 0.85*p_ice

    print(md.verbose)
    print(md.stressbalance)
    print(md.transient)
    print(md.toolkits)

    # Store the parameters in the model    

    resdir = f"RES/output_run_{job_id}_{paramsdict['Name']}"

    if not os.path.exists(resdir):
        os.makedirs(resdir)

    #hydro = md.hydrology
    hydro = md.hydrology
    with open(os.path.join(resdir, 'md.hydrology.pkl'), 'wb') as modelinfo:
        pickle.dump(hydro, modelinfo)
    initialization = md.initialization
    with open(os.path.join(resdir, 'md.initialization.pkl'), 'wb') as initinfo:
        pickle.dump(initialization, initinfo)
    friction = md.friction
    with open(os.path.join(resdir, 'md.friction.pkl'), 'wb') as frictioninfo:
        pickle.dump(friction, frictioninfo)    

    with open(os.path.join(resdir, 'paramsdict.pkl'), 'wb') as paramsinfo:
        pickle.dump(paramsdict, paramsinfo)
    
    # solve the model
    md = solve(md,'Transient')
    requested_outputs = extract_requested_outputs(md)
    for field in requested_outputs.keys():
        np.save(os.path.join(resdir, '{}.npy'.format(field)), 
           requested_outputs[field])
        
    plot_requested_outputs(requested_outputs, resdir)    
    return md,requested_outputs,resdir
    """

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
        h_s = np.array([ts.HydrologySheetThickness[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        S = np.array([ts.ChannelArea[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        Qc = np.array([ts.ChannelDischarge[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        tt = np.array([ts.time for ts in md.results.TransientSolution[imin:imax]]).T,
        hydrovx = np.array([ts.HydrologyWaterVx[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        hydrovy = np.array([ts.HydrologyWaterVy[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        vel = np.array([ts.Vel[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        vx = np.array([ts.Vx[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        vy = np.array([ts.Vy[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
    )
    if md.hydrology.elastic_sheet_flag == 1:
        outputs['h_el'] = np.array([ts.HydrologyElasticSheetThickness[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T
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
    lh = outputs['l_h']
    Qr = outputs['Qr']
    ff = outputs['ff']
    vel = outputs['vel']
    N = outputs['N']

    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # --- AX1: Top-left plot (Lake Height and Flux) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Lake Height and Outlet Flux')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('$Q_{r}$ (m$^3$s$^{-1}$)')
    ax1.plot(tt, Qr[lakepos, :], color='black', linestyle='-', label='$Q_r$ (Outlet Flux)')

    # Create a twin axis for Lake Height
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('Lake Height (m)')
    ax1_twin.plot(tt, lh[lakepos, :], color='black', linestyle='--', label='$l_h$ (Lake Height)')

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')


    # --- AX2: Top-right plot (Flux vs Effective Pressure) ---
    ax2 = fig.add_subplot(gs[0, 1])
    QrLake = Qr[lakepos, :]
    NLake = N[lakepos, :] / 1e6  # Convert to MPa
    ax2.set_title('Lake Outlet Flux vs. Effective Pressure')
    ax2.set_xlabel('Effective Pressure, $N$ (MPa)')
    ax2.set_ylabel('$Q_{r}$ (m$^3$s$^{-1}$)')

    # Plot line in the background
    ax2.plot(NLake, QrLake, alpha=0.5, color='gray')
    # Plot scatter points on top, using a greyscale colormap
    scatter = ax2.scatter(NLake, QrLake, c=tt, cmap='Greys', s=15, edgecolor='none', zorder=10)
    cbar = fig.colorbar(scatter, ax=ax2)
    cbar.set_label('Time')
    # Add vertical dashed line for "Lake empty"
    lake_empty_N = 9.81 * md.materials.rho_ice * md.geometry.thickness[lakepos] / 1e6  # Convert to MPa
    ax2.axvline(x=lake_empty_N, linestyle='--', color='black', label='Lake empty')

    # Add horizontal dashed line for Qr = Qin
    Qr_line = paramsdict['Q_in']
    ax2.axhline(y=Qr_line, linestyle='--', color='black')

    # Annotate "lake draining" and "lake filling"
    mid_x = (NLake.min() + NLake.max()) / 2  # Calculate the middle of the x-axis span
    ax2.text(mid_x, Qr_line + 0.1 * (QrLake.max() - QrLake.min()), 'lake draining', 
        verticalalignment='bottom', horizontalalignment='center')
    ax2.text(mid_x, Qr_line - 0.1 * (QrLake.max() - QrLake.min()), 'lake filling', 
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
            ax2.annotate(
                '', # No text is needed for the arrow
                xy=(x_head, y_head),
                xytext=(x_tail, y_tail),
                arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
                zorder=20 # Ensure arrows are drawn on top
            )


    # --- AX3: Bottom plot (Flotation and Velocity) ---
    ax3 = fig.add_subplot(gs[1, :])  # The ':' makes this subplot span the entire row
    ax3.set_title('Median Flotation Fraction and Ice Velocity')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Flotation Fraction')
    ax3.plot(tt, np.median(ff, axis=0), color='black', linestyle='-', label='Median Flotation Fraction')

    # Create a twin axis for Velocity
    ax3_twin = ax3.twinx()
    ax3_twin.set_ylabel('Velocity (m a⁻¹)')
    ax3_twin.plot(tt, np.median(vel, axis=0), color='black', linestyle='--', label='Median Velocity')

    # Combine legends from both axes
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4)

    # Adjust layout automatically to prevent labels/titles from overlapping
    fig.tight_layout()

    # Save figure
    fig.savefig(os.path.join(resdir, 'Summary.png'), dpi=300)
    print(f"Saved figure to {os.path.join(resdir, 'Summary.png')}")

    # Plot channels during a lake drainage cycle
    # Get index of channel peaks and troughs
    lh_lake = lh[lakepos, :]
    peaks,troughs = lakeheightminmax(lh_lake)
    if len(peaks) == 1:
        idx1 = troughs[-1]  # Last trough
        # Find the mid lake height between max lake height and drainage
        mid_value = (lh_lake[peaks[-1]] + lh_lake[troughs[-1]]) / 2
        start = np.min([peaks[-1], troughs[-1]])
        end = np.max([peaks[-1], troughs[-1]])
        idx2 = min(range(start, end + 1), key=lambda i: abs(lh_lake[i] - mid_value))
        idx3 = peaks[-1]  # Last peak

        fig = plt.figure(figsize=(7, 8))
        gs = gridspec.GridSpec(4, 2, width_ratios=[20,1], wspace=0.2, hspace=0.35)
        # --- Top plot: Channel 1 ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1, sm1 = plotchannels(mesh,np.abs(outputs['Qc'][:,idx1]),ax=ax1,min=0.1,quiver=False)
        ax1.set_xlim([0, 10e3])
        ax1.set_ylim([0, 3e3])
        ax1.text(0.95, 0.95, f"{tt[idx1]:.2f} years", transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
        cax1 = fig.add_subplot(gs[0, 1])
        cbar1 = fig.colorbar(sm1, cax=cax1)
        #cbar1.set_label('Channel Discharge (m$^3$ s$^{-1}$)')
        # --- Middle plot: Channel 2 ---
        ax2 = fig.add_subplot(gs[1, 0])
        ax2, sm2 = plotchannels(mesh,np.abs(outputs['Qc'][:,idx2]),ax=ax2,min=0.1,quiver=False)
        ax2.set_xlim([0, 10e3])
        ax2.set_ylim([0, 3e3])
        ax2.text(0.95, 0.95, f"{tt[idx2]:.2f} years", transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
        ax2.set_ylabel('Y [m]')
        cax2 = fig.add_subplot(gs[1, 1])
        cbar2 = fig.colorbar(sm2, cax=cax2)
        cbar2.set_label('Channel Discharge (m$^3$ s$^{-1}$)')
        # --- Bottom plot: Channel 3 ---
        ax3 = fig.add_subplot(gs[2, 0])
        ax3, sm3 = plotchannels(mesh,np.abs(outputs['Qc'][:,idx3]),ax=ax3,min=0.1,quiver=False)
        ax3.set_xlim([0, 10e3])
        ax3.set_ylim([0, 3e3])
        ax3.text(0.95, 0.95, f"{tt[idx3]:.2f} years", transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
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
        ax4.vlines([tt[idx1], tt[idx2], tt[idx3]],color='gray',linestyles='--',ymin=0,ymax=np.max(lh_lake)+50)

        # Save figure
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
        fig = plt.figure(figsize=(7, 10))
        gs = gridspec.GridSpec(6, 2, width_ratios=[20,1], wspace=0.2, hspace=0.35)
        # --- Top plot: Channel 1 ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1, sm1 = plotchannels(mesh,np.abs(outputs['Qc'][:,idx1]),ax=ax1,min=0.1,quiver=False)
        ax1.set_xlim([0, 10e3])
        ax1.set_ylim([0, 3e3])
        ax1.text(0.95, 0.95, f"{tt[idx1]:.2f} years", transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
        cax1 = fig.add_subplot(gs[0, 1])
        cbar1 = fig.colorbar(sm1, cax=cax1)
        #cbar1.set_label('Channel Discharge (m$^3$ s$^{-1}$)')
        # --- Second plot: Channel 2 ---                                       
        ax2 = fig.add_subplot(gs[1, 0])
        ax2, sm2 = plotchannels(mesh,np.abs(outputs['Qc'][:,idx2]),ax=ax2,min=0.1,quiver=False)
        ax2.set_xlim([0, 10e3])
        ax2.set_ylim([0, 3e3])
        ax2.text(0.95, 0.95, f"{tt[idx2]:.2f} years", transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
        #ax2.set_ylabel('Y [m]')
        cax2 = fig.add_subplot(gs[1, 1])
        cbar2 = fig.colorbar(sm2, cax=cax2)
        #cbar2.set_label('Channel Discharge (m$^3$ s$^{-1}$)')
        # --- Third plot: Channel 3 ---                                       
        ax3 = fig.add_subplot(gs[2, 0])
        ax3, sm3 = plotchannels(mesh,np.abs(outputs['Qc'][:,idx3]),ax=ax3,min=0.1,quiver=False)
        ax3.set_xlim([0, 10e3])
        ax3.set_ylim([0, 3e3])
        ax3.text(0.95, 0.95, f"{tt[idx3]:.2f} years", transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
        #ax3.set_ylabel('Y [m]')
        cax3 = fig.add_subplot(gs[2, 1])
        cbar3 = fig.colorbar(sm3, cax=cax3)
        cbar3.set_label('Channel Discharge (m$^3$ s$^{-1}$)')
        # --- Fourth plot: Channel 4 ---                                       
        ax4 = fig.add_subplot(gs[3, 0])
        ax4, sm4 = plotchannels(mesh,np.abs(outputs['Qc'][:,idx4]),ax=ax4,min=0.1,quiver=False)
        ax4.set_xlim([0, 10e3])
        ax4.set_ylim([0, 3e3])
        ax4.text(0.95, 0.95, f"{tt[idx4]:.2f} years", transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
        ax4.set_ylabel('Y [m]')
        cax4 = fig.add_subplot(gs[3, 1])
        cbar4 = fig.colorbar(sm4, cax=cax4)
        #cbar4.set_label('Channel Discharge (m$^3$ s$^{-1}$)')
        # --- Fifth plot: Channel 5 ---                                       
        ax5 = fig.add_subplot(gs[4, 0])
        ax5, sm5 = plotchannels(mesh,np.abs(outputs['Qc'][:,idx5]),ax=ax5,min=0.1,quiver=False)
        ax5.set_xlim([0, 10e3])
        ax5.set_ylim([0, 3e3])
        ax5.set_ylabel('Y [m]')
        ax5.text(0.95, 0.95, f"{tt[idx5]:.2f} years", transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
        ax5.set_ylabel('Y [m]')
        cax5 = fig.add_subplot(gs[4, 1])
        cbar5 = fig.colorbar(sm5, cax=cax5)
        #cbar5.set_label('Channel Discharge (m$^3$ s$^{-1}$)')
        # --- Sixth plot: lake height in idx range
        ax5 = fig.add_subplot(gs[5, 0])
        ax5.plot(tt, lh_lake, color='black', linestyle='-', label='Lake Height')
        ax5.set_xlim([tt[idx1]-1.5, tt[idx5]+1.5])
        ax5.set_ylabel('Lake Height [m]')
        ax5.set_xlabel('model time [yrs]')
        ax5.vlines([tt[idx1], tt[idx2], tt[idx3],tt[idx4],tt[idx5]],color='gray',linestyles='--',ymin=0,ymax=np.max(lh_lake)+50)
        ax5.set_ylim([0, np.max(lh_lake) + 20])
        # Save figure
        fig.savefig(os.path.join(resdir, 'Channel_plot.png'), dpi=300)

        
    

    





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



