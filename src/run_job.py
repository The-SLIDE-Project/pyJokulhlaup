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
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from parameterize import parameterize

from src.utils import import_table

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
    paramsdict['s_melt_flag'] = float(run_table['seasonal melt flag'][job_id-1])
    paramsdict['h_el_flag'] = float(run_table['$h_{el}$ flag'][job_id-1])
    paramsdict['Name'] = run_table['Name'][job_id-1]
    paramsdict['Notes'] = run_table['Notes'][job_id-1]

    # initialize the model
    md = model()

    # read in mesh and pass to ISSM
    meshfile = '../data/geometry/synthetic_mesh.pkl'
    with open(meshfile, 'rb') as meshin:
        mesh = pickle.load(meshin)
    md = meshconvert(md, mesh['elements'], mesh['x'], mesh['y'])

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
        Q = np.array([ts.ChannelDischarge[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        time = np.array([ts.time for ts in md.results.TransientSolution[imin:imax]]).T,
        hydrovx = np.array([ts.HydrologyWaterVx[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        hydrovy = np.array([ts.HydrologyWaterVy[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        vel = np.array([ts.Vel[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        vx = np.array([ts.Vx[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        vy = np.array([ts.Vy[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
    )
    if md.hydrology.elastic_sheet_flag == 1:
        outputs['h_el'] = np.array([ts.HydrologyElasticSheetThickness[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T
    return outputs

def plot_requested_outputs(outputs,resdir):
    """
    Plot simple timeseries outputs from the ISSM model.

    Parameters
    ----------
    outputs : dict
        Dictionary containing model output fields.
    """
    # read in mesh and pass to ISSM
    meshfile = '../data/geometry/synthetic_mesh.pkl'
    with open(meshfile, 'rb') as meshin:
        mesh = pickle.load(meshin)
    lakepos = mesh['lakepos']
    tt = outputs['time']
    lh = outputs['l_h']
    Qr = outputs['Qr']
    ff = outputs['ff']
    vel = outputs['vel']
    N = outputs['N']

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    ax1 = axs[0]
    ax1.set_title('Lake outlet flux and sheet discharge')
    ax1.set_xlabel('time')
    ax1.set_ylabel('$Q_{r}$ [m$^3$s$^-1$]')
    ax1.plot(tt, Qr[lakepos, :], 'k-', label='Q_r')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Lake Height')
    ax2.plot(tt, lh[lakepos, :], 'k--', label='l_h')  

    ax3 = axs[1]
    QrLake = Qr[lakepos, :]
    NLake = N[lakepos, :]/1e6
    t = tt
    ax3.set_title('Lake outlet flux vs Effective Pressure')
    ax3.set_xlabel('N [MPa]')
    ax3.set_ylabel('$Q_{r}$ [m$^3$s$^-1$]')
    scatter = ax3.scatter(NLake,QrLake,c=t,cmap='viridis', s=10,edgecolor='none')
    ax3.plot(NLake, QrLake,alpha=0.5,color='gray')
    cbar = fig.colorbar(scatter, ax=ax3, label='time')


    ax4 = axs[2]
    ax4.set_title('Flotation fraction and Velocity')
    ax4.set_xlabel('time')
    ax4.set_ylabel('Flotation fraction')
    ax4.plot(tt, np.median(ff, axis=0), label='Median flotation fraction')
    ax4 = ax4.twinx()
    ax4.set_ylabel('Velocity (m a⁻¹)')
    ax4.plot(tt, np.median(vel, axis=0), 'g', label='Median velocity')

    lines3, labels3 = ax4.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax4.legend(lines3 + lines4, labels3 + labels4)

    # Save figure
    fig.savefig(os.path.join(resdir, 'Summary.png'), dpi=300)


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



