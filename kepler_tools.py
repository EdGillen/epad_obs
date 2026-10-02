"""Analysis steps for kep_light.ipynb: Kepler light curves, transit search, outliers and transit fits.

All times are in BKJD (Barycentric Kepler Julian Date = BJD - 2454833), all fluxes are relative.
"""
import glob
import io
import os
import urllib.parse
import urllib.request
import warnings

import numpy as np
import pandas as pd
with warnings.catch_warnings():        # hide an irrelevant warning about an optional lightkurve module
    warnings.simplefilter('ignore')
    import lightkurve as lk
import batman
from astropy.timeseries import BoxLeastSquares
from scipy.ndimage import median_filter
from scipy.optimize import least_squares

BKJD0 = 2454833.0                       # BJD of BKJD = 0
EXPOSURE = 1765.5/86400                 # Kepler long-cadence exposure time (days)
SUPERSAMPLE = 21                        # model points per exposure (see transit_model)
DURATIONS = np.array([0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3])   # trial transit durations (days)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'kepler')


# ----------------------------------------------------------------------------------------- data
def bundled_targets():
    """Table of the bundled targets with their NASA Exoplanet Archive parameters (data/kepler/targets.csv)."""
    return pd.read_csv(os.path.join(DATA, 'targets.csv')).set_index('target')


def load_light_curves(files):
    """Read Kepler light-curve files (one per quarter) and join them.

    Returns a dict of plain numpy arrays: time, quarter, sap_raw (electrons/s), sap and pdc (each
    normalised to its quarter's median) and pdc_err, plus the per-quarter CROWDSAP and FLFRCSAP values.
    """
    parts, info = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for fn in sorted(files):
            lc = lk.read(fn)
            q = int(lc.meta['QUARTER'])
            t = lc.time.value
            sap = np.ma.filled(np.ma.masked_invalid(np.ma.getdata(lc.sap_flux.value)), np.nan).astype(float)
            pdc = np.ma.filled(np.ma.masked_invalid(np.ma.getdata(lc.pdcsap_flux.value)), np.nan).astype(float)
            err = np.ma.filled(np.ma.masked_invalid(np.ma.getdata(lc.pdcsap_flux_err.value)), np.nan).astype(float)
            ok = np.isfinite(t) & np.isfinite(sap) & np.isfinite(pdc) & np.isfinite(err)
            t, sap, pdc, err = t[ok], sap[ok], pdc[ok], err[ok]
            parts.append((t, np.full(len(t), q), sap, sap/np.median(sap), pdc/np.median(pdc), err/np.median(pdc)))
            info[q] = (lc.meta.get('CROWDSAP'), lc.meta.get('FLFRCSAP'))
    t, q, sap_raw, sap, pdc, err = (np.concatenate(c) for c in zip(*parts))
    order = np.argsort(t)
    return dict(time=t[order], quarter=q[order], sap_raw=sap_raw[order], sap=sap[order], pdc=pdc[order],
                pdc_err=err[order], quarter_info=info)


def bundled_light_curves(target):
    return load_light_curves(glob.glob(os.path.join(DATA, bundled_targets().loc[target, 'folder'], '*.fits')))


def download_light_curves(name):
    """Download all long-cadence Kepler light curves of a star from MAST (needs internet; a few MB per quarter)."""
    res = lk.search_lightcurve(name, mission='Kepler', author='Kepler', cadence='long')
    if len(res) == 0:
        raise ValueError(f'No Kepler long-cadence light curves found for {name}')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lcs = res.download_all()
    return load_light_curves([lc.meta['FILENAME'] for lc in lcs])


def archive_parameters(planet):
    """Transit parameters of a planet from the NASA Exoplanet Archive (its default solution), or None."""
    cols = ('pl_name,pl_refname,pl_orbper,pl_orbpererr1,pl_orbpererr2,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,'
            'pl_ratror,pl_ratrorerr1,pl_ratrorerr2,pl_ratdor,pl_ratdorerr1,pl_ratdorerr2,'
            'pl_orbincl,pl_orbinclerr1,pl_orbinclerr2')
    query = f"select {cols} from ps where pl_name = '{planet}' and default_flag = 1"
    url = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?' + urllib.parse.urlencode({'query': query, 'format': 'csv'})
    df = pd.read_csv(io.BytesIO(urllib.request.urlopen(url, timeout=60).read()))
    if len(df) == 0:
        return None
    r = df.iloc[0]
    sym = lambda c: 0.5*(abs(r[c + 'err1']) + abs(r[c + 'err2']))
    ref = pd.Series([r.pl_refname]).str.extract(r'target=ref>([^<]*)<', expand=False)[0]
    return pd.Series(dict(planet=planet, period=r.pl_orbper, period_err=sym('pl_orbper'),
                          t0_bjd=r.pl_tranmid, t0_err=sym('pl_tranmid'), rp_rs=r.pl_ratror, rp_rs_err=sym('pl_ratror'),
                          a_rs=r.pl_ratdor, a_rs_err=sym('pl_ratdor'), inc=r.pl_orbincl, inc_err=sym('pl_orbincl'),
                          u1=np.nan, u2=np.nan, reference=ref))


# ----------------------------------------------------------------------------------------- detrending
def flatten(time, flux, window_days=2.0, mask=None):
    """Divide out slow trends with a Savitzky-Golay filter (lightkurve's flatten).

    Points where mask is True (e.g. transits) are ignored when the trend is estimated.
    Returns (flat flux, trend).
    """
    cadence = np.median(np.diff(time))
    window = int(window_days/cadence) | 1                       # odd number of points
    # lightkurve treats gaps longer than break_tolerance cadences as breaks and filters each piece
    # separately; gaps left by masked transits (up to ~0.5 day) must not count as breaks
    breaks = max(5, int(np.ceil(0.5/cadence)))
    lc = lk.LightCurve(time=time, flux=flux, flux_err=np.full(len(flux), 1e-4))
    mask = np.zeros(len(time), bool) if mask is None else np.asarray(mask, bool)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        flat, trend = lc.flatten(window_length=window, mask=mask, return_trend=True, break_tolerance=breaks)
    return np.asarray(flat.flux.value, float), np.asarray(trend.flux.value, float)


def robust_sigma(x):
    """Scatter of x estimated from the median absolute deviation (insensitive to outliers and transits)."""
    x = x[np.isfinite(x)]
    return 1.4826*np.median(np.abs(x - np.median(x)))


def quarter_sigma(flux, quarter, use=None):
    """Scatter of each point's quarter (robust_sigma over that quarter's points where use is True).

    Kepler rotated by 90 degrees every quarter, so a star fell on a different CCD with a different
    noise level; each point gets the scatter of its own quarter.
    """
    use = np.ones(len(flux), bool) if use is None else use
    sigma = np.empty(len(flux))
    for q in np.unique(quarter):
        sigma[quarter == q] = robust_sigma(flux[(quarter == q) & use])
    return sigma


# ----------------------------------------------------------------------------------------- transit search
def phase_time(time, t0, period):
    """Time since the nearest transit (days), between -P/2 and +P/2."""
    return (time - t0 + 0.5*period) % period - 0.5*period


def bls_search(time, flux, sigma, pmin, pmax, oversample=2):
    """Box Least Squares periodogram (sigma: scatter of each point, or one value for all).

    The trial periods are spaced so that over the time span of the data a transit drifts by less
    than (shortest trial duration)/oversample between neighbouring trial periods.
    Returns a dict with the periodogram and the best period, transit time, duration, depth, SNR and SDE.
    """
    span = time.max() - time.min()
    periods = np.exp(np.arange(np.log(pmin), np.log(pmax), DURATIONS.min()/(oversample*span)))
    durations = DURATIONS[DURATIONS < 0.5*pmin]
    res = BoxLeastSquares(time, flux, dy=np.broadcast_to(sigma, np.shape(flux))).power(periods, durations, objective='snr')
    k = int(np.argmax(res.power))
    sde = (res.power[k] - np.mean(res.power))/np.std(res.power)       # signal detection efficiency
    return dict(period=float(res.period[k]), t0=float(res.transit_time[k]), duration=float(res.duration[k]),
                depth=float(res.depth[k]), snr=float(res.power[k]), sde=float(sde), span=float(span),
                periods=np.asarray(res.period), power=np.asarray(res.power))


def refine_period(time, flux, sigma, bls, oversample=20):
    """Refine the period found by bls_search, using all the data given.

    A search over a time span T fixes the period to roughly P*duration/T. Search +-5 times that
    (at most +-20%) with all the data: first on a grid fine enough not to miss the peak (a transit
    drifts by at most half its duration over the data between neighbouring periods), then on a grid
    'oversample' times finer around the best period.
    """
    span = time.max() - time.min()
    P0, dur0 = bls['period'], bls['duration']
    durs = DURATIONS[(DURATIONS > 0.5*dur0) & (DURATIONS < 2.01*dur0)]
    model = BoxLeastSquares(time, flux, dy=np.broadcast_to(sigma, np.shape(flux)))
    step = dur0/(2*span)                                  # in ln(period)
    window = min(5*dur0/bls['span'], 0.2)
    res = model.power(P0*np.exp(np.arange(-window, window + step, step)), durs, objective='snr')
    P1 = float(res.period[np.argmax(res.power)])
    res = model.power(P1*np.exp(np.linspace(-step, step, 2*oversample + 1)), durs, objective='snr')
    k = int(np.argmax(res.power))
    return dict(bls, period=float(res.period[k]), t0=float(res.transit_time[k]), duration=float(res.duration[k]),
                depth=float(res.depth[k]))


def transit_mask(time, period, t0, duration, factor=1.5):
    """True for points within factor*duration/2 of a transit centre."""
    return np.abs(phase_time(time, t0, period)) < 0.5*factor*duration


# ----------------------------------------------------------------------------------------- outliers
def find_outliers(flux, sigma, protect, nsigma=7, neighbours=5):
    """Outliers: points more than nsigma*sigma from the running median of their nearest neighbours.

    A transit is a run of low points, so the running median follows it and transit points are not
    flagged; a single bad point (e.g. a cosmic ray) stands out from its neighbours. Points where
    'protect' is True (the transits found) are never flagged.
    """
    resid = flux - median_filter(flux, size=neighbours, mode='nearest')
    return (np.abs(resid) > nsigma*sigma) & ~protect


# ----------------------------------------------------------------------------------------- transit model
def kipping_to_u(q1, q2):
    """Quadratic limb-darkening coefficients from Kipping's (2013) q1, q2 (0 <= q1, q2 <= 1 is physical)."""
    return 2*np.sqrt(q1)*q2, np.sqrt(q1)*(1 - 2*q2)


def u_to_kipping(u1, u2):
    return (u1 + u2)**2, u1/(2*(u1 + u2))


def transit_model(time, t0, period, rp_rs, a_rs, inc, u1, u2):
    """batman model of a transit (circular orbit, quadratic limb darkening), averaged over the
    29.4-minute Kepler exposures.

    Each exposure is represented by SUPERSAMPLE model points, at the middles of SUPERSAMPLE equal parts
    of the exposure (batman spreads its points from the start to the end of exp_time, so exp_time is
    shortened by (n - 1)/n; batman's default would average over n/(n - 1) times the exposure, which for
    n = 7 changes the model of Kepler-7 b and 8 b by 100-150 ppm and biases the fitted a/R* and inclination).
    With 21 points the model is accurate to better than 1 ppm for the bundled planets.
    To save time the model is only calculated within half a transit duration plus one exposure of
    mid-transit; elsewhere it is exactly 1.
    """
    dt = phase_time(np.asarray(time, float), t0, period)
    flux = np.ones(len(dt))
    # upper limit to half the transit duration (the planet is within 1 + Rp/R* of the star's centre)
    half = period/(2*np.pi)*np.arcsin(min(1.0, (1 + rp_rs)/(a_rs*np.sin(np.radians(inc))))) + EXPOSURE
    near = np.abs(dt) < half
    if near.any():
        p = batman.TransitParams()
        p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0., period, rp_rs, a_rs, inc, 0., 90.
        p.limb_dark, p.u = 'quadratic', [u1, u2]
        flux[near] = batman.TransitModel(p, dt[near], supersample_factor=SUPERSAMPLE,
                                        exp_time=EXPOSURE*(SUPERSAMPLE - 1)/SUPERSAMPLE).light_curve(p)
    return flux


def impact_parameter(a_rs, inc):
    return a_rs*np.cos(np.radians(inc))


# ----------------------------------------------------------------------------------------- fitting
PARAMETERS = ['t0', 'period', 'rp_rs', 'a_rs', 'inc', 'q1', 'q2']


def archive_epoch(arch, t_ref):
    """Published mid-transit time moved to the transit nearest t_ref (BKJD), with its uncertainty
    (including that of the period, multiplied by the number of orbits it was moved)."""
    t0 = arch.t0_bjd - BKJD0
    n = np.round((t_ref - t0)/arch.period)
    return t0 + n*arch.period, float(np.hypot(arch.t0_err, n*arch.period_err))


def start_from_archive(arch, t_ref, nsigma=3.0):
    """Starting values: published values shifted by +nsigma uncertainties (epoch moved close to t_ref).

    If that would put the inclination above 90 degrees it is shifted by -nsigma instead, and if the
    impact parameter would be above 0.9 (no or hardly any transit) the inclination is set to give b = 0.9.
    Returns (start dict, list of notes).
    """
    notes = []
    t0, t0_err = archive_epoch(arch, t_ref)
    err = dict(t0=t0_err, period=arch.period_err, rp_rs=arch.rp_rs_err, a_rs=arch.a_rs_err, inc=arch.inc_err)
    for n, e in err.items():
        if not np.isfinite(e):
            err[n] = 0.0
            notes.append(f'{n}: no published uncertainty, so it started at the published value.')
    start = dict(t0=t0 + nsigma*err['t0'], period=arch.period + nsigma*err['period'],
                 rp_rs=arch.rp_rs + nsigma*err['rp_rs'], a_rs=arch.a_rs + nsigma*err['a_rs'])
    inc = arch.inc + nsigma*err['inc']
    if inc > 90:
        inc = arch.inc - nsigma*err['inc']
        notes.append(f'i: shifted by -{nsigma:g} sigma (+{nsigma:g} sigma would exceed 90 deg).')
    if impact_parameter(start['a_rs'], inc) > 0.9:
        inc = np.degrees(np.arccos(min(0.9/start['a_rs'], 1)))
        notes.append(f'i: the shifted values would give hardly any transit (impact parameter above 0.9), '
                     f'so it started at {inc:.2f} deg (b = 0.9).')
    start['inc'] = inc
    return start, notes


def fit_transit(time, flux, sigma, start, free, u1, u2, period_guess):
    """Least-squares fit of the transit model.

    start: dict of starting values (t0, period, rp_rs, a_rs, inc); free: dict name -> bool;
    u1, u2: limb darkening (fitted through Kipping's q1, q2 when free['ld'] is True).
    sigma: scatter of each point (or one value for all).
    Returns a dict with the best values, their uncertainties, the starting values actually used, notes, ...
    Uncertainties come from the covariance matrix, scaled up by sqrt(reduced chi^2) if that is > 1.
    Also returned: parameters at a limit, parameters the data do not constrain (uncertainty larger than
    half the allowed range), and pairs of strongly correlated parameters (|correlation| > 0.95).
    """
    notes = []
    if free['ld']:                     # start from the q1, q2 closest to u1, u2 (q1, q2 must be in 0..1)
        q1, q2 = u_to_kipping(u1, u2) if u1 + u2 > 0 else (0.0, 0.5)
        q1, q2 = float(np.clip(q1, 0, 1)), float(np.clip(q2, 0, 1))
        if not np.allclose(kipping_to_u(q1, q2), (u1, u2), atol=1e-6):
            notes.append('limb darkening: u1 = %g, u2 = %g is not physical; the fit started from u1 = %.3f, u2 = %.3f.'
                         % ((u1, u2) + kipping_to_u(q1, q2)))
    else:                              # fixed: use u1, u2 as they are
        q1 = q2 = np.nan
    values = dict(start, q1=q1, q2=q2)
    names = [n for n in PARAMETERS if (free['ld'] if n in ('q1', 'q2') else free[n])]

    def limb(v):
        return kipping_to_u(v['q1'], v['q2']) if free['ld'] else (u1, u2)

    lower = dict(t0=start['t0'] - 0.1, period=period_guess*(1 - 1e-4), rp_rs=1e-3, a_rs=1.2, inc=50., q1=0., q2=0.)
    upper = dict(t0=start['t0'] + 0.1, period=period_guess*(1 + 1e-4), rp_rs=0.5, a_rs=100., inc=90., q1=1., q2=1.)

    def model_for(p):
        v = dict(values, **dict(zip(names, p)))
        return transit_model(time, v['t0'], v['period'], v['rp_rs'], v['a_rs'], v['inc'], *limb(v))

    p0 = [min(max(values[n], lower[n]), upper[n]) for n in names]
    notes += [f'{n}: the starting value was outside the allowed range and was moved to its edge.'
              for n, v in zip(names, p0) if v != values[n]]
    if not names:
        raise ValueError('tick at least one parameter to fit')
    fit = least_squares(lambda p: (flux - model_for(p))/sigma, p0, bounds=([lower[n] for n in names], [upper[n] for n in names]),
                        x_scale='jac')
    chi2 = float(np.sum(fit.fun**2)); dof = max(len(flux) - len(names), 1)
    best = dict(values, **dict(zip(names, fit.x)))
    at_limit = [n for n, v in zip(names, fit.x) if min(v - lower[n], upper[n] - v) < 5e-3*(upper[n] - lower[n])]
    # Covariance matrix of the parameters that are not at a limit (a parameter at its limit has no
    # meaningful uncertainty). A plain inverse, not a pseudo-inverse: the pseudo-inverse silently
    # drops directions the data cannot constrain and so gives far too small uncertainties.
    inside = [k for k, n in enumerate(names) if n not in at_limit]
    J = fit.jac[:, inside]
    errors, correlated, unconstrained = {}, [], []
    try:
        cov = np.linalg.inv(J.T @ J)*max(chi2/dof, 1)
        err = np.sqrt(np.diag(cov))
        corr = cov/np.outer(err, err)
    except np.linalg.LinAlgError:
        err, corr = np.full(len(inside), np.nan), np.full((len(inside), len(inside)), np.nan)
    for i, k in enumerate(inside):
        n = names[k]
        errors[n] = float(err[i])
        if not np.isfinite(err[i]) or err[i] > 0.5*(upper[n] - lower[n]):
            unconstrained.append(n)
        correlated += [(n, names[inside[j]], float(corr[i, j])) for j in range(i + 1, len(inside)) if abs(corr[i, j]) > 0.95]
    best['u1'], best['u2'] = limb(best)
    start_used = dict(values, **dict(zip(names, p0)))
    start_used['u1'], start_used['u2'] = limb(start_used)
    if not fit.success:
        notes.append(f'the fit did not converge ({fit.message})')
    return dict(best=best, errors=errors, at_limit=at_limit, unconstrained=unconstrained, correlated=correlated,
                chi2_red=chi2/dof, n_points=len(flux), free=names, start=start_used, notes=notes,
                success=bool(fit.success), model=lambda t: transit_model(t, best['t0'], best['period'], best['rp_rs'],
                                                                        best['a_rs'], best['inc'], best['u1'], best['u2']))
