"""Analysis steps for astrometry.ipynb: Gaia epoch astrometry, astrometric orbits and masses.

Gaia measures one coordinate per CCD: the position w of a star along the scan direction, which has
position angle theta (east of north). For a star with offsets (dra*, ddec) from a reference point,
parallax plx, proper motion (pmra*, pmdec) and an orbital wobble of its photocentre (dra*_orb, ddec_orb):

    w = (dra* + pmra* t) sin(theta) + (ddec + pmdec t) cos(theta) + plx f + dra*_orb sin(theta) + ddec_orb cos(theta)

with t in years since J2017.5 and f the along-scan parallax factor (Lindegren et al. 2021; Holl et al. 2023).
The orbit is written with the Thiele-Innes constants A, B, F, G (mas):

    ddec_orb = A X + F Y,   dra*_orb = B X + G Y,   X = cos E - e,   Y = sqrt(1 - e^2) sin E

where E is the eccentric anomaly (Kepler's equation M = E - e sin E, mean anomaly M = 2 pi t/P + M0).
For given P, e and M0 the model is LINEAR in the other 9 parameters, so these are found by linear
least squares; P, e and M0 are found with a grid search.
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'gaia')
YEAR = 365.25                                 # days per Julian year
MJUP = 9.547919e-4                            # Jupiter mass in solar masses
LINEAR = ['dra', 'ddec', 'plx', 'pmra', 'pmdec', 'A', 'F', 'B', 'G']

# Published values, for comparison (see the references in the notebook)
SYSTEMS = {
    'Gaia-4': dict(
        kind='planet', host_mass=0.644, host_mass_err=0.02,
        published=dict(period=571.3, period_err=1.4, e=0.338, a0=0.279, inc=116.9, m2=11.8*MJUP, m2_err=0.7*MJUP),
        source='Stefansson et al. (2025): Gaia DR3 two-body solution combined with radial velocities'),
    'Gaia BH3': dict(
        kind='black hole', host_mass=0.76, host_mass_err=0.05,
        published=dict(period=4194.7, period_err=112.3, e=0.7262, a0=27.07, a0_err=0.56, inc=110.66, m2=32.70, m2_err=0.82),
        source='Gaia Collaboration, Panuzzo et al. (2024): pre-release Gaia astrometry (and, for the mass, radial velocities)'),
    'HD 114762': dict(
        kind='star', host_mass=1.00, host_mass_err=0.10,
        published=dict(period=83.91712, period_err=0.00064, e=0.3442, e_err=0.0012, inc=3.63, inc_err=0.06,
                       m2=0.215, m2_err=0.013),
        source='Winn (2022): Gaia DR3 two-body solution combined with radial velocities'),
}


# ----------------------------------------------------------------------------------------- data
def load(system):
    """Measurements of one system: dict of arrays t (yr since J2017.5), theta (rad), f, w (mas),
    sigma (mas), transit, plus ra0, dec0 (deg) and g_mag."""
    d = pd.read_csv(os.path.join(DATA, 'epoch_astrometry.csv'))
    d = d[d.system == system].sort_values('t')
    s = pd.read_csv(os.path.join(DATA, 'sources.csv')).set_index('system').loc[system]
    return dict(t=d.t.values, theta=np.radians(d.scan_angle.values), f=d.parallax_factor.values, w=d.w.values,
                sigma=d.sigma.values, transit=d.transit_id.values, ra0=float(s.ra0), dec0=float(s.dec0),
                g_mag=float(s.g_mag), system=system)


def subset(d, keep):
    """The measurements in d where keep is True."""
    n = len(d['t'])
    return {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in d.items()}


def transit_means(d, values):
    """Mean of 'values' over the CCDs of each transit, with its standard error (from sigma); also the
    mean time of each transit."""
    _, inverse = np.unique(d['transit'], return_inverse=True)
    wt = 1/d['sigma']**2
    sw = np.bincount(inverse, wt)
    mean = np.bincount(inverse, wt*values)/sw
    t = np.bincount(inverse, d['t'])/np.bincount(inverse)
    return t, mean, 1/np.sqrt(sw)


# ----------------------------------------------------------------------------------------- models
def kepler_E(M, e):
    """Eccentric anomaly E from Kepler's equation M = E - e sin E (Newton's method)."""
    M = np.mod(M, 2*np.pi)
    if np.size(M) == 0:
        return M
    E = M + e*np.sin(M)*(1 + e*np.cos(M))
    for _ in range(30):
        dE = (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
        E -= dE
        if np.max(np.abs(dE)) < 1e-12:
            break
    return E


def orbit_xy(t, period, e, M0):
    """Elliptical rectangular coordinates X, Y (orbit with unit semi-major axis) at times t (yr)."""
    E = kepler_E(2*np.pi*t*YEAR/period + M0, e)
    return np.cos(E) - e, np.sqrt(1 - e**2)*np.sin(E)


def design(d, period=None, e=0.0, M0=0.0):
    """Design matrix: columns dra, ddec, plx, pmra, pmdec (and A, F, B, G if a period is given)."""
    s, c, t = np.sin(d['theta']), np.cos(d['theta']), d['t']
    cols = [s, c, d['f'], t*s, t*c]
    if period is not None:
        X, Y = orbit_xy(t, period, e, M0)
        cols += [X*c, Y*c, X*s, Y*s]
    return np.column_stack(cols)


def linear_fit(D, w, sigma):
    """Weighted linear least squares: parameters, chi^2 and covariance matrix (from sigma)."""
    Dw, ww = D/sigma[:, None], w/sigma
    p, *_ = np.linalg.lstsq(Dw, ww, rcond=None)
    chi2 = float(np.sum((ww - Dw @ p)**2))
    return p, chi2, np.linalg.inv(Dw.T @ Dw)


def single_star_fit(d):
    """Fit position, parallax and proper motion only (5 parameters)."""
    p, chi2, cov = linear_fit(design(d), d['w'], d['sigma'])
    dof = len(d['w']) - 5
    return dict(params=dict(zip(LINEAR[:5], p)), chi2=chi2, dof=dof, chi2_red=chi2/dof,
                errors=dict(zip(LINEAR[:5], np.sqrt(np.diag(cov)*max(chi2/dof, 1)))), resid=d['w'] - design(d) @ p)


def orbit_fit(d, period, e, M0):
    """For given P (days), e and M0 (rad): fit the other 9 parameters (linear)."""
    D = design(d, period, e, M0)
    p, chi2, cov = linear_fit(D, d['w'], d['sigma'])
    dof = len(d['w']) - 12
    return dict(period=period, e=e, M0=M0, params=dict(zip(LINEAR, p)), chi2=chi2, dof=dof, chi2_red=chi2/dof,
                resid=d['w'] - D @ p, single_star_part=D[:, :5] @ p[:5], orbit_part=D[:, 5:] @ p[5:])


# ----------------------------------------------------------------------------------------- searching
def periodogram(d, pmin=10.0, pmax=6000.0, oversample=5):
    """Improvement in chi^2 over the single-star model for circular orbits (e = 0) on a grid of
    periods (days), evenly spaced in frequency. For e = 0 the starting phase is absorbed by the four
    linear Thiele-Innes constants, so each period needs only one linear fit."""
    span = (d['t'].max() - d['t'].min())*YEAR
    freqs = np.arange(1/pmax, 1/pmin, 1/(oversample*span))
    chi2_0 = single_star_fit(d)['chi2']
    delta = np.array([chi2_0 - orbit_fit(d, 1/fr, 0.0, 0.0)['chi2'] for fr in freqs])
    return 1/freqs, delta


def best_orbit(d, pmin=10.0, pmax=6000.0, clip=None):
    """Grid search: periodogram (circular orbits), then a grid in P (around the highest peak), e and M0,
    then a least-squares refinement of P, e, M0 (with the 9 linear parameters solved at every step).

    clip: if given, measurements more than clip times their uncertainty from the best orbit are then
    left out and the orbit is refined once more (Gaia's own solution also gives such outliers almost no
    weight). Returns the fit, with fit['keep'] = the measurements used.
    """
    fit = _best_orbit(d, pmin, pmax)
    keep = np.ones(len(d['t']), bool)
    if clip is not None:
        keep = np.abs(fit['resid']/d['sigma']) <= clip
        dk = subset(d, keep)
        fit = _refine(dk, fit['period'], fit['e'], fit['M0'], pmin, pmax)
    fit['keep'] = keep
    return fit


def _refine(d, P, e, M0, pmin, pmax):
    def resid(x):
        return orbit_fit(d, x[0], x[1], x[2])['resid']/d['sigma']

    sol = least_squares(resid, [P, e, M0], bounds=([pmin, 0.0, -4*np.pi], [pmax, 0.95, 4*np.pi]), x_scale='jac')
    P, e, M0 = sol.x
    return orbit_fit(d, P, e, np.mod(M0, 2*np.pi))


def _best_orbit(d, pmin, pmax):
    periods, delta = periodogram(d, pmin, pmax)
    P0 = periods[np.argmax(delta)]
    span = (d['t'].max() - d['t'].min())*YEAR
    width = P0**2/span                                    # approximate width of a periodogram peak (days)
    best = (np.inf, None)
    for P in np.geomspace(max(P0 - 2*width, pmin), min(P0 + 2*width, pmax), 21):
        for e in np.arange(0.0, 0.95, 0.05):
            for M0 in np.linspace(0, 2*np.pi, 24, endpoint=False):
                chi2 = orbit_fit(d, P, e, M0)['chi2']
                if chi2 < best[0]:
                    best = (chi2, (P, e, M0))

    return _refine(d, *best[1], pmin, pmax)


# ----------------------------------------------------------------------------------------- orbital elements
def thiele_innes(a0, inc, omega, Omega):
    """Thiele-Innes constants A, B, F, G from a0 (mas) and inc, omega, Omega (radians)."""
    ci = np.cos(inc)
    A = a0*(np.cos(omega)*np.cos(Omega) - np.sin(omega)*np.sin(Omega)*ci)
    B = a0*(np.cos(omega)*np.sin(Omega) + np.sin(omega)*np.cos(Omega)*ci)
    F = -a0*(np.sin(omega)*np.cos(Omega) + np.cos(omega)*np.sin(Omega)*ci)
    G = -a0*(np.sin(omega)*np.sin(Omega) - np.cos(omega)*np.cos(Omega)*ci)
    return A, B, F, G


def campbell(A, B, F, G):
    """Campbell elements from the Thiele-Innes constants: a0 (mas), inclination, argument of periastron
    omega and longitude of the ascending node Omega (degrees; Omega in 0-180, as usual for astrometry)."""
    u = (A**2 + B**2 + F**2 + G**2)/2
    v = A*G - B*F
    a0 = np.sqrt(u + np.sqrt((u + v)*(u - v)))
    w_plus_O = np.arctan2(B - F, A + G)
    w_minus_O = np.arctan2(-B - F, A - G)
    omega, Omega = (w_plus_O + w_minus_O)/2, (w_plus_O - w_minus_O)/2
    flip = Omega < 0                                       # put Omega in 0-180 (then omega changes by 180)
    Omega = np.where(flip, Omega + np.pi, Omega)
    omega = np.where(flip, omega + np.pi, omega)
    inc = 2*np.arctan((((A - G)**2 + (B + F)**2)/((A + G)**2 + (B - F)**2))**0.25)   # the ratio is tan^4(i/2)
    return a0, np.degrees(inc), np.degrees(np.mod(omega, 2*np.pi)), np.degrees(Omega)


def mass_function(a0, plx, period):
    """Astrometric mass function (solar masses): (a0/plx)^3/(P/yr)^2 = M2^3/(M1 + M2)^2 for a dark companion."""
    return (a0/plx)**3/(period/YEAR)**2


def companion_mass(fM, M1):
    """Companion mass M2 (solar masses) from the mass function and the host mass M1 (dark companion);
    NaN if the mass function is not positive (e.g. for a negative parallax)."""
    if not (np.isfinite(fM) and fM > 0 and M1 > 0):
        return np.nan
    # m2^3/(M1 + m2)^2 grows monotonically from 0; at m2 = 10 (fM + M1) + 1 it is always larger than fM
    return brentq(lambda m2: m2**3/(M1 + m2)**2 - fM, 0.0, 10*(fM + M1) + 1)


def uncertainties(d, fit, M1=None, M1_err=0.0, full=True, n=4000, seed=1):
    """Uncertainties of the fitted and derived parameters.

    full=True (at the best fit): the covariance matrix of all 12 parameters comes from the Jacobian of
    the model. full=False (P, e and M0 set by hand, so the fit is not at a minimum in these): only the 9
    linear parameters, with P, e, M0 held fixed (their uncertainties are then NaN).
    The covariance is scaled by the reduced chi^2 if that is > 1 (the formal errors of the measurements
    are too small). Derived quantities (a0, i, omega, Omega, time of periastron, mass function, companion
    mass) are found by drawing parameter sets from this covariance matrix. Returns dict name -> (value, error).
    """
    x0 = np.array([fit['period'], fit['e'], fit['M0']] + [fit['params'][k] for k in LINEAR])

    def model(x):
        return design(d, x[0], np.clip(x[1], 0, 0.99), x[2]) @ x[3:]

    cov = np.zeros((12, 12))
    if full:
        steps = np.r_[1e-6*x0[0], 1e-6, 1e-6, np.full(9, 1e-6)]
        J = np.column_stack([(model(x0 + dx) - model(x0 - dx))/(2*dx[k]) for k, dx in enumerate(np.diag(steps))])
        J /= d['sigma'][:, None]
        try:      # plain inverse: a pseudo-inverse would hide directions the data cannot constrain
            cov = np.linalg.inv(J.T @ J)*max(fit['chi2_red'], 1)
        except np.linalg.LinAlgError:
            cov[:] = np.nan
    else:
        _, _, c9 = linear_fit(design(d, fit['period'], fit['e'], fit['M0']), d['w'], d['sigma'])
        cov[3:, 3:] = c9*max(fit['chi2_red'], 1)
    rng = np.random.default_rng(seed)
    if np.all(np.isfinite(cov)):
        vals, vecs = np.linalg.eigh((cov + cov.T)/2)          # draw from the covariance (robust to round-off)
        x = x0 + (rng.standard_normal((n, 12))*np.sqrt(np.clip(vals, 0, None))) @ vecs.T
    else:
        x = np.tile(x0, (n, 1))
    x[:, 1] = np.clip(x[:, 1], 0, 0.99)
    a0, inc, omega, Omega = campbell(x[:, 8], x[:, 10], x[:, 9], x[:, 11])         # columns A, F, B, G -> order A, B, F, G
    T = 2017.5 - x[:, 2]/(2*np.pi)*x[:, 0]/YEAR                                        # a periastron passage (yr)
    T0 = 2017.5 - fit['M0']/(2*np.pi)*fit['period']/YEAR
    tmid = 2017.5 + np.mean(d['t'])
    k = np.round((tmid - T0)/(fit['period']/YEAR))
    T = T + k*x[:, 0]/YEAR                                                             # the one nearest the middle of the data
    fM = mass_function(a0, x[:, 5], x[:, 0])                                          # column 5: parallax
    out = {name: (float(x0[i]), float(np.sqrt(cov[i, i])) if (full or i >= 3) else np.nan)
           for i, name in enumerate(['period', 'e', 'M0'] + LINEAR)}
    A, F, B, G = (fit['params'][k] for k in ['A', 'F', 'B', 'G'])
    best = dict(zip(['a0', 'inc', 'omega', 'Omega'], campbell(A, B, F, G)))
    best['T'] = T0 + k*fit['period']/YEAR
    best['fM'] = mass_function(best['a0'], fit['params']['plx'], fit['period'])
    # Omega is only defined in 0-180 (with omega shifted by 180 when Omega is): put every sample on the
    # same branch as the best fit before measuring the spread
    flip = np.abs((Omega - best['Omega'] + 180) % 360 - 180) > 90
    Omega = np.where(flip, Omega + 180, Omega)
    omega = np.where(flip, omega + 180, omega)
    spread = dict(a0=np.nanstd(a0), inc=np.nanstd(inc), omega=circ_std(omega), Omega=circ_std(Omega), T=np.nanstd(T),
                  fM=np.nanstd(fM))
    for name in spread:
        out[name] = (float(best[name]), float(spread[name]))
    if not full:
        out['T'] = (out['T'][0], np.nan)                     # set by the period and mean anomaly sliders
    if M1 is not None:
        m1 = np.clip(M1 + M1_err*rng.standard_normal(n), 1e-3, None)
        m2 = np.array([companion_mass(f, m) for f, m in zip(fM[:1000], m1[:1000])])
        out['m2'] = (companion_mass(best['fM'], M1), float(np.nanstd(m2)) if np.isfinite(m2).sum() > 10 else np.nan)
    return out


def circ_std(angles_deg):
    """Spread (standard deviation) of angles in degrees, allowing for the wrap at 360."""
    a = np.radians(angles_deg)
    R = np.hypot(np.mean(np.cos(a)), np.mean(np.sin(a)))
    return np.degrees(np.sqrt(-2*np.log(max(R, 1e-12))))


# ----------------------------------------------------------------------------------------- simulation
def earth_position(t):
    """Barycentric position of the Earth (au, equatorial ICRS axes) at times t (yr since J2017.5)."""
    from astropy.time import Time
    from astropy.coordinates import get_body_barycentric
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = get_body_barycentric('earth', Time(2017.5 + np.asarray(t), format='jyear', scale='tcb'))
    return p.x.to('au').value, p.y.to('au').value, p.z.to('au').value


def parallax_factors(t, ra0, dec0, scale=1.01):
    """Parallax factors in R.A. and Dec (the shift of a star with parallax 1 mas, in mas). Gaia orbits
    the Sun about 1% further out than the Earth (around the L2 point), hence scale = 1.01."""
    X, Y, Z = earth_position(t)
    ra, dec = np.radians(ra0), np.radians(dec0)
    fa = X*np.sin(ra) - Y*np.cos(ra)
    fd = X*np.cos(ra)*np.sin(dec) + Y*np.sin(ra)*np.sin(dec) - Z*np.cos(dec)
    return scale*fa, scale*fd


def semi_major_axes(m2, m1, period):
    """Relative orbit a and the host star's orbit around the centre of mass a1 (au); masses in solar
    masses, period in days (Kepler's third law)."""
    a = ((m1 + m2)*(period/YEAR)**2)**(1/3)
    return a, a*m2/(m1 + m2)


def simulate(d, m2, m1, period, e, inc, dist, omega=np.radians(90.0), Omega=np.radians(45.0), M0=1.0,
             pm=(-60.0, 20.0), seed=1):
    """Gaia-like measurements of a simulated star with a dark companion, at the times, scan angles and
    parallax factors of the real measurements in d, with Gaussian noise of the real uncertainties.
    Masses in solar masses, period in days, inc/omega/Omega in radians, distance in pc, pm in mas/yr.
    Returns dict with the simulated w, the true parameters and the photocentre semi-major axis a0 (mas)."""
    plx = 1000.0/dist
    a, a1 = semi_major_axes(m2, m1, period)
    a0 = a1*plx                                           # mas
    A, B, F, G = thiele_innes(a0, inc, omega, Omega)
    true = np.array([0.0, 0.0, plx, pm[0], pm[1], A, F, B, G])
    w_true = design(d, period, e, M0) @ true
    rng = np.random.default_rng(seed)
    return dict(w=w_true + d['sigma']*rng.standard_normal(len(w_true)), w_true=w_true, a0=a0, a=a, a1=a1, plx=plx,
                params=dict(zip(LINEAR, true)), period=period, e=e, M0=M0)


def sky_track(t, ra0, dec0, plx, pm, A=0.0, B=0.0, F=0.0, G=0.0, period=None, e=0.0, M0=0.0):
    """Offsets (dra*, ddec) in mas of the star on the sky at times t: proper motion + parallax (+ orbit)."""
    fa, fd = parallax_factors(t, ra0, dec0)
    x, y = pm[0]*t + plx*fa, pm[1]*t + plx*fd
    if period is not None:
        X, Y = orbit_xy(t, period, e, M0)
        x, y = x + B*X + G*Y, y + A*X + F*Y
    return x, y
