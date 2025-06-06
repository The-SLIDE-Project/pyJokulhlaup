import pickle

import numpy as np

import os
import sys
ISSM_DIR = os.getenv('ISSM_DIR')
sys.path.append(os.path.join(ISSM_DIR, 'src/m/dev/'))
import devpath
from issmversion import issmversion
from hydrologyglads import hydrologyglads
from generic import generic
from paterson import *
from bcgsbjacobioptions import *
from mumpsoptions import *
#from SetIceSheetBC import *
#from setmask import setmask



# Vectors for convenience
xvec = md.mesh.x
onevec = 0*xvec + 1

# Calving
#md.calving.calvingrate=0*onevec

# Geometry
print(os.getcwd())
bed = np.load('../data/geometry/synthetic_bed.npy')
surf = np.load('../data/geometry/synthetic_surface.npy')
thick = surf - bed
md.geometry.base = bed
md.geometry.bed =bed
md.geometry.surface = surf
md.geometry.thickness = thick

# Flow 


# Velocity
md.initialization.vx = -100.0 * np.ones(md.mesh.numberofvertices)  # Shape: (N,)
md.initialization.vy = np.zeros(md.mesh.numberofvertices)         # Shape: (N,)
md.initialization.vel = np.sqrt(md.initialization.vx**2 + md.initialization.vy**2)

md.inversion.iscontrol = 0
# Friction
md.friction.p = 0.95 * np.ones((md.mesh.numberofelements, 1))
md.friction.q = 1.25 * np.ones((md.mesh.numberofelements, 1))
md.friction.coefficient = np.sqrt(10.**2.) * np.ones((md.mesh.numberofvertices, 1))
N = md.constants.g * md.materials.rho_ice * md.geometry.thickness
md.friction.coupling = 4
md.friction.effective_pressure = N
md.friction.effective_pressure_limit = 0.
md.basalforcings.geothermalflux = (68./1000.) * np.ones((md.mesh.numberofvertices, 1))  # W/m^2

# Constants
md.materials.rheology_B = paterson(273.)*onevec
md.materials.rheology_law = 'Paterson'
md.initialization.temperature = (273.)*onevec
#md.thermal.spctemperature = md.initialization.temperature
md.materials.rheology_n = 3.*np.ones((md.mesh.numberofelements,1))
md.materials.rho_water = 1023.
md.materials.rho_freshwater = 1.e3
md.materials.rho_ice = 917.
md.materials.mu_water = md.materials.rho_freshwater * 1.793e-6
md.constants.g = 9.81

# HYDROLOGY
# parameters
md.hydrology = hydrologyglads()
md.hydrology.sheet_conductivity = 0.05*onevec
md.hydrology.sheet_alpha = 5./4.
md.hydrology.sheet_beta = 3./2.
md.hydrology.cavity_spacing = 10.
md.hydrology.bump_height = 0.5*onevec
md.hydrology.channel_sheet_width = md.hydrology.cavity_spacing
md.hydrology.omega = 1./2000.
md.hydrology.englacial_void_ratio = 1.e-4
md.hydrology.rheology_B_base = paterson(273.)*onevec
md.hydrology.istransition = 1
md.hydrology.ischannels = 1
md.hydrology.islakes = 1
md.hydrology.channel_conductivity = 0.5*onevec
md.hydrology.channel_alpha = 5./4.
md.hydrology.channel_beta = 3./2.
md.hydrology.creep_open_flag = 0
md.hydrology.melt_flag = 1
md.hydrology.elastic_sheet_flag = 0
md.hydrology.elastic_sheet_depth_scale = 0
md.hydrology.elastic_sheet_exponent = 1.
md.hydrology.uplift_reg_rate = 0.01/1.e3/9.81
md.hydrology.reg_pressure = 1.e4
md.hydrology.moulin_input = 0.*onevec


# Lakes
md.hydrology.islakes = 1
md.hydrology.lake_mask = np.zeros((md.mesh.numberofvertices, 1))
md.hydrology.lake_area = np.zeros((md.mesh.numberofvertices, 1))
md.hydrology.lake_Qin = np.zeros((md.mesh.numberofvertices, 1))
md.hydrology.num_lakes = 0

# md.hydrology.requested_outputs = ['default']
md.hydrology.requested_outputs = [
        'HydraulicPotential',
        'EffectivePressure',
        'HydrologySheetThickness',
        'ChannelDischarge',
        'ChannelArea',
        # 'HydrologySheetDischarge',
        'HydrologyWaterVx',
        'HydrologyWaterVy',
        'HydrologyLakeHeight',
        'HydrologyLakeOutletQr'
]



# BOUNDARY CONDITIONS
md.hydrology.spcphi = np.nan*onevec
pos = np.where(np.logical_and(
    md.mesh.vertexonboundary,
    md.mesh.x==np.min(md.mesh.x, axis=-1)))
md.hydrology.spcphi[pos] = 5.e5
md.hydrology.neumannflux = np.zeros((md.mesh.numberofelements, 1))




# TOLERANCES
md.stressbalance.restol = 1.e-3
md.stressbalance.reltol = np.nan
md.stressbalance.abstol = np.nan
md.stressbalance.maxiter = 50
md.debug.gprof = 0
md.debug.profiling =1
md.toolkits.DefaultAnalysis=bcgsbjacobioptions()
md.toolkits.RecoveryAnalysis=mumpsoptions()

# TIMESTEPPING
nyears = 0.1
hour = 3600.
day = 86400.
dt_hours = 1.
out_freq = 72./dt_hours
# out_freq = 1
md.timestepping.time_step = dt_hours*hour/md.constants.yts
md.timestepping.final_time = nyears

md.settings.output_frequency = out_freq


md.transient.deactivateall()
md.transient.ishydrology = True
md.transient.isstressbalance = True

md.verbose.solution = True
md.miscellaneous.name = 'output'

md.cluster = generic('np', 5)

SLURM_TMPDIR = os.getenv('SLURM_TMPDIR')
if SLURM_TMPDIR:
    md.cluster.executionpath = SLURM_TMPDIR
else:
    cwd = os.getcwd()
    expath = os.path.join(cwd, 'TMP/')
    if not os.path.exists(expath):
        os.makedirs(expath)
    md.cluster.executionpath = expath

print(md.cluster.executionpath)



