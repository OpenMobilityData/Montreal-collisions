"""Time-aware point rates: the one-day intersection counts turned into a year's crossings with a seasonal profile,
and the collisions of the matching years set against them.

Stage 1, the profile. The Kensington loop counter on rue de Terrebonne (borough release of request 2026-581,
first year July 26, 2025–July 25, 2026, daily bicycles both directions) gives the monthly shape of bicycle traffic on
the street itself: f_m = the month's mean daily count / the year's mean daily count. The City's permanent counters
(open data « Compteurs cyclistes permanents », 2018 daily site totals, 2019 15-minute rows) give the same shape for the
years of the Girouard counts, as a check. The weekday/weekend factor comes from the only full week counted on
Terrebonne (Girouard, September 6–12, 2023), with the Kensington counter's monthly ratios as the check.

Stage 2, the exposure. A count c taken in month m on a weekday or a weekend day is scaled to a year as
c / (f_m × g_t) × 365, where g_t is the day-type factor (g_wd = 7 / (5 + 2 r), g_we = r × g_wd, r = weekend / weekday).
Pedestrians: no seasonal counter exists for the street, the monthly profile is taken as flat (stated as an assumption),
the day-type factor from the same September 2023 week. Several count-days of one visit are averaged.

Stage 3 (after the owner's review of stages 1–2): the collisions within 100 m by year and window, and the rates.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'derived' / 'tables'
OPEN = Path.home() / 'Terrebonne-data/third-party/montreal-opendata'
KENSINGTON = Path.home() / 'GitHub/Terrebonne-access-requests/2026-581/derived/kensington_daily_borough_release_2025-07_2026-08.csv'
YEAR1 = ('2025-07-26', '2026-07-25')
OUTAGE = ['2026-05-29', '2026-05-30', '2026-05-31']  # three zero days, counter outage (2026-581 summary)
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def kensington():
    k = pd.read_csv(KENSINGTON, parse_dates=['date'])
    k = k[(k.date >= YEAR1[0]) & (k.date <= YEAR1[1])].copy()
    k['outage'] = k.date.dt.strftime('%Y-%m-%d').isin(OUTAGE)
    k['month'] = k.date.dt.month; k['weekend'] = k.date.dt.dayofweek >= 5
    v = k[~k.outage]
    annual_mean = v.bike.mean()
    m = v.groupby('month').bike.mean()
    wd = v[~v.weekend].groupby('month').bike.mean(); we = v[v.weekend].groupby('month').bike.mean()
    prof = pd.DataFrame({'kensington_bikes_per_day': m.round(1), 'kensington_factor': (m / annual_mean).round(3),
                         'kensington_weekend_over_weekday': (we / wd).round(2)})
    return prof, annual_mean, k


def eco_2018():
    e = pd.read_csv(OPEN / 'cyclistes_eco_2018_2026-10-27.csv')
    e['date'] = pd.to_datetime(e['Date'], format='%d/%m/%Y'); e['month'] = e.date.dt.month
    sites = [c for c in e.columns if c not in ('Date', e.columns[1], 'date', 'month')]
    rows = {}
    for s in sites:
        x = pd.to_numeric(e[s], errors='coerce')
        if x.notna().sum() < 360 or (x.fillna(0) > 0).sum() < 300:  # a full year, counting in every season
            continue
        mm = x.groupby(e.month).mean()
        rows[s] = mm / x.mean()
    return pd.DataFrame(rows)


def eco_2019():
    e = pd.read_csv(OPEN / 'cyclistes_eco_2019_2026-10-27.csv', usecols=['date', 'id_compteur', 'nb_passages'])
    d = e.groupby(['id_compteur', 'date']).nb_passages.sum().reset_index()
    d['date'] = pd.to_datetime(d.date); d['month'] = d.date.dt.month
    rows = {}
    for cid, g in d.groupby('id_compteur'):
        if g.date.nunique() < 360 or (g.nb_passages > 0).sum() < 300:
            continue
        rows[str(cid)] = g.groupby('month').nb_passages.mean() / g.nb_passages.mean()
    return pd.DataFrame(rows)


def profile_table():
    prof, annual_mean, k = kensington()
    e18, e19 = eco_2018(), eco_2019()
    prof['eco2018_median_factor'] = e18.median(axis=1).round(3); prof['eco2018_sites'] = e18.shape[1]
    prof['eco2019_median_factor'] = e19.median(axis=1).round(3); prof['eco2019_sites'] = e19.shape[1]
    if '100011783' in e19.columns:  # the NDG-area counter (de Maisonneuve near Décarie), the only permanent counter west of downtown in 2019
        prof['eco2019_ndg_100011783'] = e19['100011783'].round(3)
    prof.index = MONTHS
    return prof, annual_mean, e18, e19


def day_type_factors():
    """r = weekend / weekday from the Girouard week of Sept 6–12, 2023 (Wed–Tue: 5 weekdays, Sat, Sun)."""
    d = pd.read_csv(OUT / 'terrebonne_intersection_counts_by_day.csv', parse_dates=['Date'])
    w = d[(d.Nom_Intersection == 'Girouard / Terrebonne') & (d.Date >= '2023-09-06') & (d.Date <= '2023-09-12')]
    we = w.Date.dt.dayofweek >= 5
    out = {}
    for mode in ('bikes', 'peds'):
        r = w[we][mode].mean() / w[~we][mode].mean()
        g_wd = 7 / (5 + 2 * r)
        out[mode] = dict(weekday_mean=round(w[~we][mode].mean(), 1), weekend_mean=round(w[we][mode].mean(), 1), r=round(r, 3), g_weekday=round(g_wd, 3), g_weekend=round(r * g_wd, 3))
    return out, d


def exposure_table(prof, factors, d):
    rows = []
    full = d[d.full_day].copy()
    full['month'] = full.Date.dt.month; full['weekend'] = full.Date.dt.dayofweek >= 5
    for _, r in full.iterrows():
        fm = prof.kensington_factor.iloc[r.month - 1]
        for mode, col in (('bike', 'bikes'), ('ped', 'peds')):
            if r[col] <= 0:
                continue  # the mode was not counted that day
            f = fm if mode == 'bike' else 1.0  # pedestrians: flat profile (assumption)
            g = factors[col]['g_weekend' if r.weekend else 'g_weekday']
            annual = r[col] / (f * g) * 365
            rows.append(dict(street=r.street, intersection=r.Nom_Intersection, date=r.Date.date(), year=r.Date.year, day=r.Date.strftime('%a'),
                             mode=mode, count=int(r[col]), month_factor=round(f, 3), day_factor=round(g, 3), implied_mean_daily=round(r[col] / (f * g), 1), implied_annual_crossings=int(round(annual))))
    E = pd.DataFrame(rows)
    # Per visit (intersection, mode, year-month): the mean of the per-day annual estimates
    E['visit'] = E.date.astype(str).str[:7]
    V = E.groupby(['street', 'intersection', 'mode', 'year', 'visit']).agg(days=('count', 'size'), counts=('count', lambda s: ' '.join(map(str, s))),
                                                                         implied_mean_daily=('implied_mean_daily', 'mean'), implied_annual_crossings=('implied_annual_crossings', 'mean')).reset_index()
    V['implied_mean_daily'] = V.implied_mean_daily.round(1); V['implied_annual_crossings'] = V.implied_annual_crossings.round(0).astype(int)
    return E, V


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', type=int, default=2); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); pd.set_option('display.width', 250)
    prof, annual_mean, e18, e19 = profile_table()
    print(f'Kensington first year, mean bicycles/day excluding the three outage days: {annual_mean:.1f}')
    print(prof.to_string()); prof.to_csv(OUT / 'seasonal_profile.csv', index_label='month')
    e18.to_csv(OUT / 'seasonal_profile_eco2018_by_site.csv', index_label='month'); e19.to_csv(OUT / 'seasonal_profile_eco2019_by_counter.csv', index_label='month')
    print('\nEco 2018 sites kept:', ', '.join(e18.columns)); print('Eco 2019 counters kept:', ', '.join(e19.columns))
    factors, d = day_type_factors()
    print('\nDay-type factors from Girouard, Sept 6–12, 2023:'); print(pd.DataFrame(factors).T.to_string())
    if a.stage >= 2:
        E, V = exposure_table(prof, factors, d)
        E.to_csv(OUT / 'count_day_annualized_exposure.csv', index=False); V.to_csv(OUT / 'count_visit_annualized_exposure.csv', index=False)
        print('\nPer count-day:'); print(E.drop(columns=['visit']).to_string(index=False))
        print('\nPer visit:'); print(V.to_string(index=False))


# ---------------------------------------------------------------- stage 3: windows and rates (rule approved Oct 27, 2026)
WINTER = {12, 1, 2, 3}  # visits in these months are shown but left out of the primary exposure


def ci(k, expo):
    from scipy import stats
    lo = stats.chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0; hi = stats.chi2.ppf(0.975, 2 * (k + 1)) / 2
    return lo / expo, hi / expo


def rate_table(E, d):
    import spvm
    coll = spvm.load(); coll = coll[coll.LOC_X.notna()]
    pts = d[d.full_day].groupby(['street', 'Nom_Intersection']).agg(lat=('lat', 'first'), lon=('lon', 'first')).reset_index()
    E = E.copy(); E['month'] = pd.to_datetime(E.date).dt.month; E['winter'] = E.month.isin(WINTER)
    rows = []
    for _, p in pts.iterrows():
        x, y = spvm.mtm8(float(p.lat), float(p.lon)); near = coll[spvm.dist_m(coll, x, y) <= 100]
        for mode in ('bike', 'ped'):
            e = E[(E.intersection == p.Nom_Intersection) & (E['mode'] == mode)]
            if e.empty:
                continue
            e = e[e.year <= 2021]  # a visit after the collision file's last year (2021) cannot enter a window; the 2023 Girouard week is reported apart
            if e.empty:
                continue
            prim = e[~e.winter] if (~e.winter).any() else e  # primary exposure: non-winter visits; winter-only intersections keep theirs, flagged
            winter_only = not (~e.winter).any()
            years = sorted(prim.year.unique()); y0, y1 = max(2012, min(years) - 1), min(2021, max(years) + 1)
            n_years = y1 - y0 + 1
            daily = prim.implied_mean_daily.mean()  # day-weighted mean over the visits' count-days
            expo = daily * 365.25 * n_years / 1e6
            win = near[(near.year >= y0) & (near.year <= y1)]
            k = int(win[mode].sum()); lo, hi = ci(k, expo)
            visits = e.groupby(e.date.astype(str).str[:7]).agg(days=('count', 'size'), daily=('implied_mean_daily', 'mean'))
            ex_hi = visits.daily.max() * 365.25 * n_years / 1e6; ex_lo = visits.daily.min() * 365.25 * n_years / 1e6
            band_lo = k / ex_hi; band_hi = k / ex_lo; ci_hi_at_hi = ci(k, ex_hi)[1]; ci_hi_at_lo = ci(k, ex_lo)[1]
            rows.append(dict(street=p.street, intersection=p.Nom_Intersection, mode=mode, count_years=' '.join(map(str, years)), window=f'{y0}-{y1}', window_years=n_years,
                             visits=len(visits), count_days=int(prim['count'].size), winter_visits_excluded=int(e.winter.sum() > 0 and not winter_only), winter_only=int(winter_only),
                             implied_mean_daily=round(daily, 1), implied_annual_crossings=int(round(daily * 365.25)), million_crossings_window=round(expo, 3),
                             collisions_in_window=k, injury_collisions_in_window=int((win[mode] & win.severity.isin(['fatal', 'severe', 'light'])).sum()),
                             rate_per_million=round(k / expo, 2), ci_low=round(lo, 2), ci_high=round(hi, 2),
                             rate_at_highest_visit=round(band_lo, 2), rate_at_lowest_visit=round(band_hi, 2), ci_high_at_highest_visit=round(ci_hi_at_hi, 2), ci_high_at_lowest_visit=round(ci_hi_at_lo, 2),
                             visit_daily=' '.join(f'{i}:{v:.1f}' for i, v in visits.daily.items()),
                             collisions_2012_2021=int(near[mode].sum())))
    return pd.DataFrame(rows)


def girouard_by_year(d):
    import spvm
    coll = spvm.load(); coll = coll[coll.LOC_X.notna()]
    x, y = spvm.mtm8(45.478338, -73.624871); near = coll[spvm.dist_m(coll, x, y) <= 100]
    by = near.groupby('year').agg(all=('bike', 'size'), bike=('bike', 'sum'), ped=('ped', 'sum')).reindex(range(2012, 2022), fill_value=0)
    return by


if __name__ == '__main__' and a.stage >= 3:
    R = rate_table(E, d); R.to_csv(OUT / 'terrebonne_point_rates_time_aware.csv', index=False)
    by = girouard_by_year(d); by.to_csv(OUT / 'girouard_terrebonne_collisions_100m_by_year.csv', index_label='year')
    print('\nCollisions within 100 m of Girouard / Terrebonne by year:'); print(by.T.to_string())
    print('\nTime-aware rates (window = count years ± 1, clipped to 2012–2021; primary exposure from non-winter visits):')
    print(R[['street', 'intersection', 'mode', 'count_years', 'window', 'visits', 'count_days', 'winter_visits_excluded', 'winter_only', 'implied_mean_daily', 'implied_annual_crossings', 'million_crossings_window', 'collisions_in_window', 'injury_collisions_in_window', 'rate_per_million', 'ci_low', 'ci_high', 'rate_at_highest_visit', 'rate_at_lowest_visit', 'ci_high_at_highest_visit', 'ci_high_at_lowest_visit', 'collisions_2012_2021']].to_string(index=False))
    t23 = E[(E.intersection == 'Girouard / Terrebonne') & (E.year == 2023)].groupby('mode').implied_annual_crossings.mean()
    print('\nGirouard / Terrebonne 2023 exposure (no collision data for 2022 onwards in the SPVM file; issue #42): ' + ', '.join(f'{m} {int(v):,}/yr' for m, v in t23.items()))
