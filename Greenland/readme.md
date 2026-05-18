# Greenland analysiss for Hepburn et al., 2026

## Simulating jökulhlaups from an ice-marginal lake within a 2D model of subglacial drainage and basal sliding

This subdirectory contains the material necessary to analyse the Greenland experiments from Hepburn et al., 2026 (EGUSphere preprint). 
 
### `models/`

Is where the model run data, saved as `.nc` and `.npy` files, avaliable stored seperately at the [online repository linked to this paper](LINK.COM), should be stored. 

The folder `65-LowBh-transient-17-02-2026-13-12` should be stored here.

At the same repository, the velocity anomaly data used to plot Fig.7 is also avaliable in `models/velanom/`

### `analysis/`

contains everything necessary to plot figures 5-7 in the manuscript. 

Run using:

`python 3.12 plot_Greenland.py`