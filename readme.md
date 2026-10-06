# pyJökulhlaup README
## Supplementary material for Simulating jökulhlaups from an ice-marginal lake within a 2D model of subglacial drainage and basal sliding

Adam J. Hepburn<sup>*,1</sup> , Sammie Buzzard<sup>2</sup>, Andrew J. Sole<sup>3</sup>, Stephen J. Livingstone<sup>3</sup>, Felix Ng<sup>3</sup>,
Mathieu Morlighem<sup>4</sup>, Elizabeth A. Bagshaw<sup>5</sup>, Caroline Clason<sup>6</sup>, Lisa Craw<sup>7</sup>, Christine F. Dow<sup>8</sup>,
Samuel Doyle<sup>1</sup>, Jonathan Hawkins<sup>7</sup>, Matthew Peacey<sup>1</sup>, and Robert Storrar<sup>9</sup>

<sup>*</sup>email: `adam.hepburn@aber.ac.uk`

<sup>1</sup> Centre for Glaciology, Department of Geography and Earth Sciences, Aberystwyth University, Aberystwyth, UK

<sup>2</sup> Centre for Polar Observation and Modelling, School of Geography and Natural Sciences, Northumbria University, Newcastle
upon Tyne, UK

<sup>3</sup>
Department of Geography and Planning, The University of Sheffield, Sheffield, UK

<sup>4</sup>
Department of Earth Sciences, Dartmouth College, Hanover, NH, USA

<sup>5</sup>
School of Geographical Sciences, University of Bristol, Bristol, UK

<sup>6</sup>
Department of Geography, Durham University, Durham, UK

<sup>7</sup>
School of Earth and Environmental Sciences, Cardiff University, Cardiff, UK

<sup>8</sup>
Department of Geography and Environmental Management, University of Waterloo, Waterloo

<sup>9</sup>
Department of Natural and Built Environment, Sheffield Hallam University, Sheffield, UK

## File contents

This repo contains everything necessary to run a suite of simulations in ISSM-GlaDS with a fully-coupled ice-marginal lake.

The `synthetic/` folder contains the data, parameterisation files, and analysis files necessary to run experiments on a synthetic domain.

The `Greenland/` folder contains the code necessary to analyse the Greenland model ([data available here](https://dx.doi.org/10.5281/zenodo.20269306))

The `src/` folder contains code and utilities necessary to run a job, read the job table etc.

The `manuscript/` folder contains the paper material (final figures, and `.tex` files)

Note, this does require an ISSM install with the necessary modifications to add ice-marginal lakes. There are two options for this: 

1. A static version of the ISSM code used in our modelling is available [here](https://dx.doi.org/10.5281/zenodo.20269306). If using this version of ISSM, the relevant class can be called using `md.hydrology = hydrologyglads`. 

2. However, we have now merged our code with the current version of ISSM available [here](https://github.com/ISSMteam/ISSM). We strongly recommend the use of this version over the static code, as it restores a version of GlaDS with no lake capability, and instead adds a new hydrology class called IML-GlaDS which can be called using `md.hydrology = hydrologyimlglads`. The repository has been updated to reflect this preference.


## Package install
We suggest running this as an editable pip package.

Python dependancies are included in `requirements.txt` and we suggest installing a python3.12 virtual environment using venv or similar. 

e.g., `python3.12 -m venv issm-jokulhlaup`

will create a virtual environment called `issm-jokulhlaup`

activate this venv using

`source issm-jokulhlaup/bin/activate`

and then install packages using

`python3.12 pip install -r /path/to/this/requirements.txt`

Finally, to use this packages, `cd` into `pyjokulhlaup` and then run:

`python 3.12 pip install -e .`

This will install pyjokulhlaup as an editable package on your local system only. 

## Citation
How to cite. Hepburn, A. J., Buzzard, S., Sole, A., Livingstone, S., Ng, F., Morlighem, M., Bagshaw, E.A., Clason,
C., Craw, L., Dow, C., Doyle, Hawkins, J., S., Peacey, M., Storrar, R., 2026. Simulating jökulhlaups from
an ice-marginal lake within a 2D model of subglacial drainage and basal sliding. The Cryosphere, 20,
5675–5696. [https://doi.org/10.5194/tc-20-5675-2026].
