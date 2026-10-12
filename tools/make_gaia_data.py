"""Download the Gaia DR4 pre-release epoch astrometry and write the data used by astrometry.ipynb.

Run from the repository root (needs network access):   python tools/make_gaia_data.py

ESA released Gaia epoch astrometry (the individual one-dimensional, along-scan measurements) for 12
sources in June 2026, ahead of Gaia DR4 (https://www.cosmos.esa.int/web/gaia/dr4-prerelease). This
script keeps the three sources with published orbits (SYSTEMS) and, for each, the CCD measurements
that Gaia's own astrometric solution used (used_by_agis_al), and writes data/gaia/epoch_astrometry.csv:

  system           name of the system (see SYSTEMS)
  transit_id       Gaia field-of-view transit (one transit = up to 9 CCD measurements, a few seconds apart)
  t                barycentric time of the measurement in Julian years since J2017.5 (TCB)
  scan_angle       position angle of the scan direction (deg, east of north)
  parallax_factor  along-scan parallax factor of the transit
  w                along-scan position of the star relative to (ra0, dec0) (mas)
  sigma            formal uncertainty of w (mas): centroiding error plus calibration noise (Gaia's own single-star
                   solution also adds a source excess noise, left out here because the orbit is modelled instead)
and data/gaia/sources.csv with the source_id, reference position (ra0, dec0, deg) and G magnitude of each.

Data: ESA/Gaia/DPAC, licensed under CC BY-NC 3.0 IGO (https://creativecommons.org/licenses/by-nc/3.0/igo/;
https://www.cosmos.esa.int/web/gaia-users/license); selected and reformatted by this script.
"""
import io
import warnings
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io.votable import parse_single_table

URL = ('https://anonftp.cosmos.esa.int/pub/GAIA_PUBLIC_DATA/Gaia_DR4/dr4-prerelease/'
       'gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip')
SYSTEMS = {1457486023639239296: 'Gaia-4', 4318465066420528000: 'Gaia BH3', 3937211745905473024: 'HD 114762'}
OUT = Path('data/gaia')
NS_PER_YEAR = 365.25*86400e9                  # Julian year in nanoseconds
J2010_TO_J2017_5 = 7.5                        # obs_time_tcb counts from J2010.0; the model epoch is J2017.5


def main():
    raw = urllib.request.urlopen(URL, timeout=300).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = [n for n in z.namelist() if n.endswith('.xml')][0]
        table = parse_single_table(io.BytesIO(z.read(name))).to_table()
    rows, info, sources = [], [], []
    warnings.filterwarnings('ignore', message='.*converting a masked element to nan')
    for source_id, system in SYSTEMS.items():
        transits = table[table['source_id'] == source_id]
        for tr in transits:
            get = lambda col: np.ma.filled(np.ma.masked_invalid(np.ma.asarray(tr[col], dtype=float)), np.nan)
            t_ns = np.ma.filled(np.ma.asarray(tr['obs_time_tcb'], dtype=float), np.nan) + float(tr['obs_time_bary_corr'])
            used = np.ma.filled(np.ma.asarray(tr['used_by_agis_al'], dtype=bool), False)
            w, sigma, angle = get('centroid_pos_al'), get('centroid_pos_error_al'), get('scan_pos_angle')
            ok = used & np.isfinite(t_ns) & np.isfinite(w) & np.isfinite(sigma) & np.isfinite(angle) & (np.abs(w) < 1e6)
            for k in np.flatnonzero(ok):
                rows.append(dict(system=system, transit_id=int(tr['transit_id']), t=t_ns[k]/NS_PER_YEAR - J2010_TO_J2017_5,
                                 scan_angle=angle[k], parallax_factor=float(tr['parallax_factor_al']), w=w[k], sigma=sigma[k]))
        sources.append(dict(system=system, source_id=source_id, ra0=float(transits['ra0'][0]), dec0=float(transits['dec0'][0]),
                            g_mag=round(float(np.nanmedian(np.ma.filled(np.ma.asarray(transits['g_mag'], dtype=float), np.nan))), 2)))
        info.append(f"{system}: {len(transits)} transits, {sum(r['system'] == system for r in rows)} CCD measurements used, "
                    f"ra0 = {float(transits['ra0'][0]):.6f}, dec0 = {float(transits['dec0'][0]):.6f}")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(['system', 't'])
    df.to_csv(OUT/'epoch_astrometry.csv', index=False, float_format='%.10g')
    pd.DataFrame(sources).to_csv(OUT/'sources.csv', index=False)          # full precision for ra0, dec0
    print('\n'.join(info))
    print(f"wrote {OUT/'epoch_astrometry.csv'} ({len(df)} rows)")


if __name__ == '__main__':
    main()
