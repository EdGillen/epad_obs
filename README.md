# EPAD: observing exoplanets

Interactive notebooks on observing exoplanets. Click a badge to open a notebook in your browser via
[Binder](https://mybinder.org); you do not need to install any software.

| Notebook | App: text, sliders and plots (code hidden) | Code: read, run and edit in JupyterLab |
|---|---|---|
| Kepler light curves: finding and fitting transits | [![Kepler light curves: app](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EdGillen/epad_obs/HEAD?urlpath=voila/render/kep_light.ipynb) | [![Kepler light curves: code](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EdGillen/epad_obs/HEAD?urlpath=lab/tree/kep_light.ipynb) |
| Direct imaging: the planet around beta Pictoris | [![Direct imaging: app](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EdGillen/epad_obs/HEAD?urlpath=voila/render/direct_imaging.ipynb) | [![Direct imaging: code](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EdGillen/epad_obs/HEAD?urlpath=lab/tree/direct_imaging.ipynb) |

Notes:
- Starting Binder can take a minute or two (longer the first time after the repository changes).
- The app view runs the whole notebook before it appears (kep_light: about half a minute), and
  changing a setting recomputes all later steps, so please allow a few seconds.
- In the code view the notebook opens without any plots: click inside the notebook, then choose
  **Run > Run All Cells**.
- Binder sessions are temporary: they stop after about 10 minutes of inactivity and any changes you
  make in the code view are not saved. Download a notebook (File > Download) to keep your edits.

## Contents

- `kep_light.ipynb`: the notebook; the analysis functions it uses are in `kepler_tools.py`.
- `direct_imaging.ipynb`: angular differential imaging of beta Pictoris b; the images
  (VLT/NACO, M' band; Bonnefoy et al. 2013) are in `data/betapic_naco_mp.hdf5`.
- `data/kepler/`: Kepler long-cadence light curves (all quarters) of Kepler-7, Kepler-8 and Kepler-10
  from [MAST](https://archive.stsci.edu/kepler/), and `targets.csv` with their published transit
  parameters (Esteves et al. 2015, via the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/))
  and limb-darkening coefficients (Claret & Bloemen 2011).
- `tools/make_kepler_targets.py`: downloads these data again and rewrites `targets.csv`; add a star to
  its list to bundle it as well.

## Running locally

```bash
conda env create -f environment.yml
conda activate epad_obs
jupyter lab              # code view
voila kep_light.ipynb    # app view (likewise for direct_imaging.ipynb)
```

## Credits

Original notebooks by Sijme-Jan Paardekooper ([SijmeJan/epad_obs](https://github.com/SijmeJan/epad_obs)).
`kep_light.ipynb` was rewritten in 2026 for current Jupyter, lightkurve and Binder, with transit
searching, outlier removal and transit fitting added.
`direct_imaging.ipynb` was updated in 2026 for current Jupyter and Binder: the PSF subtraction,
originally done with PynPoint, is now written out with numpy and scipy (same result), with a slider
for the number of principal components and a switch for rotating the images.
