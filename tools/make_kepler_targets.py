"""Download the bundled Kepler data for kep_light.ipynb and write data/kepler/targets.csv.

Run from the repository root (needs network access):   python tools/make_kepler_targets.py

For each target in TARGETS it
  1. downloads all Kepler long-cadence light-curve files (one per quarter) from MAST (via lightkurve),
  2. takes the planet's transit parameters from the NASA Exoplanet Archive (table 'ps', the solution
     published in REFERENCE),
  3. computes quadratic Kepler-band limb-darkening coefficients for the star (Claret & Bloemen 2011,
     A&A 529, A75: ATLAS models, least-squares method, microturbulence 2 km/s; linear interpolation
     in Teff and log g at the nearest tabulated metallicity).
To add a target: add a line to TARGETS and run the script again.
"""
import io
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import lightkurve as lk
from astroquery.vizier import Vizier
from scipy.interpolate import griddata

REFERENCE = 'Esteves et al. 2015'          # one homogeneous analysis of the Kepler data for all three
TARGETS = ['Kepler-7', 'Kepler-8', 'Kepler-10']
OUT = Path('data/kepler')


def archive_rows(planets):
    cols = ('pl_name,hostname,pl_refname,pl_orbper,pl_orbpererr1,pl_orbpererr2,pl_tranmid,pl_tranmiderr1,'
            'pl_tranmiderr2,pl_ratror,pl_ratrorerr1,pl_ratrorerr2,pl_ratdor,pl_ratdorerr1,pl_ratdorerr2,'
            'pl_orbincl,pl_orbinclerr1,pl_orbinclerr2,st_teff,st_logg,st_met')
    names = ','.join(f"'{p}'" for p in planets)
    query = f'select {cols} from ps where pl_name in ({names})'
    url = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?' + urllib.parse.urlencode({'query': query, 'format': 'csv'})
    df = pd.read_csv(io.BytesIO(urllib.request.urlopen(url, timeout=120).read()))
    df['pl_refname'] = df['pl_refname'].str.extract(r'target=ref>([^<]*)<', expand=False)
    return df


def limb_darkening(teff, logg, feh):
    v = Vizier(row_limit=-1, columns=['logg', 'Teff', 'Z', 'a', 'b'])
    t = v.query_constraints(catalog='J/A+A/529/A75/table-af', Filt='Kp', Met='L', Mod='A', xi='2',
                            Teff='3500..8000', logg='2.5..5.0')[0]
    zs = np.array(sorted(set(t['Z'])))
    s = t[t['Z'] == zs[np.argmin(abs(zs - feh))]]
    pts = np.c_[s['Teff'], s['logg']]
    return (float(griddata(pts, s['a'], (teff, logg))), float(griddata(pts, s['b'], (teff, logg))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arch = archive_rows([f'{t} b' for t in TARGETS])
    rows = []
    for target in TARGETS:
        # 1. light curve
        res = lk.search_lightcurve(target, mission='Kepler', author='Kepler', cadence='long')
        (OUT/target).mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            lcs = res.download_all(download_dir=tmp)
            for lc in lcs:
                shutil.copy(lc.meta['FILENAME'], OUT/target/Path(lc.meta['FILENAME']).name)
        quarters = sorted(int(lc.meta['QUARTER']) for lc in lcs)
        size = sum(f.stat().st_size for f in (OUT/target).glob('*.fits'))/1e6
        # 2. transit parameters
        r = arch[(arch.pl_name == f'{target} b') & (arch.pl_refname == REFERENCE)]
        assert len(r) == 1, f'{target} b: no unique solution from {REFERENCE}'
        r = r.iloc[0]
        sym = lambda c: 0.5*(abs(r[c + 'err1']) + abs(r[c + 'err2']))   # symmetrised 1-sigma uncertainty
        # 3. limb darkening
        u1, u2 = limb_darkening(r.st_teff, r.st_logg, r.st_met)
        rows.append(dict(target=target, planet=f'{target} b', kic=lcs[0].meta['KEPLERID'],
                         quarters=' '.join(map(str, quarters)), folder=f'{target}',
                         period=r.pl_orbper, period_err=sym('pl_orbper'),
                         t0_bjd=r.pl_tranmid, t0_err=sym('pl_tranmid'),
                         rp_rs=r.pl_ratror, rp_rs_err=sym('pl_ratror'),
                         a_rs=r.pl_ratdor, a_rs_err=sym('pl_ratdor'),
                         inc=r.pl_orbincl, inc_err=sym('pl_orbincl'),
                         teff=r.st_teff, logg=r.st_logg, feh=r.st_met, u1=round(u1, 3), u2=round(u2, 3),
                         reference=REFERENCE))
        print(f'{target}: {len(lcs)} quarters {quarters} ({size:.1f} MB), u1={u1:.3f}, u2={u2:.3f}')
    pd.DataFrame(rows).to_csv(OUT/'targets.csv', index=False)
    print('wrote', OUT/'targets.csv')


if __name__ == '__main__':
    main()
