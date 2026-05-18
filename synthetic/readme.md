# Synthetic experiments for Hepburn et al., 2026

## Simulating jökulhlaups from an ice-marginal lake within a 2D model of subglacial drainage and basal sliding

This subdirectory contains the material necessary to run and analyse the synthetic experiments from Hepburn et al., 2026 (EGUSphere preprint). 

`default.py` contains the generic properties used to setup each of the sensitivity test model runs, and should be modified to adjust the number of cores your machine has.

E.g., `md.cluster = ernie('name',socket.gethostname(),'np', 22)` should become: `md.cluster = generic('name',socket.gethostname(),'np',8)` if your machine has 8 cores. 

### `data/`
 This contains the geometry file (`square_domain.exp`) and scripts (`make_mesh.py`,`make_surface_bed.py`) necessary to generate a model geometry. Running the scripts by:

 `python 3.12 make_mesh.py` 

 and then 

 `python 3.12 make_surface_bed.py`

 generates a model and then stores the result as a `synthetic_mesh.pkl` file, and a `.npy` file for the bed and surface elevation. As well as two `.png` files showing the graphical result. 

 ### `experiments`/

 This contains the code necessary to run the baseline *default* case and the sensitivity tests. It also contains the experiments run and discussed in text stored as `suite_parameters.csv`

 `./run_default.sh` runs the first experiment in `suite_parameters.csv` 

 `./run_suite.sh` runs the experiments sequentially 

 Results are stored in `RES/` and temporary files are stored in `TMP/`, log files can be found in `logs/`

 ### `analysis/`

 Contains the code necessary to plot figures 2-4 in the paper. Run code using:

 `python 3.12 plot_main_figures.py`

 which will stores results in `figures/`