"""Crude point normalization: collisions near the counted intersections of rue de Terrebonne, and of rue Sherbrooke within the NDG box as the comparison, against the City's
one-day intersection counts (open data « Comptages des véhicules, cyclistes et piétons aux intersections »).

Exposure: for each count-day, the bicycles crossing the intersection (sum of the twelve movement columns of the
« Velos » rows) and the pedestrians (sum of the four approach columns of the « Pietons » rows), with the hours covered.
Only days with 24 hours of rows are used as daily totals. Numerator: police-reported collisions 2012–2021 within 100 m
of the intersection point, involving a bicycle / with a pedestrian victim. Rate: collisions per million crossings,
taking the mean daily count of the full count-days as the daily exposure over the 3,653 days of 2012–2021, with exact
Poisson 95 % intervals. The counts are single days from 2015 to 2023; the limits are stated in the README.
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import spvm

COUNTS = Path.home() / 'Terrebonne-data/third-party/montreal-opendata/comptages-intersections'
OUT = Path(__file__).resolve().parent.parent / 'derived' / 'tables'
MOV = ['NBLT', 'NBT', 'NBRT', 'SBLT', 'SBT', 'SBRT', 'EBLT', 'EBT', 'EBRT', 'WBLT', 'WBT', 'WBRT']
APP = ['Approche_Nord', 'Approche_Sud', 'Approche_Est', 'Approche_Ouest']
DAYS = 3653  # 2012-01-01 to 2021-12-31


STREETS = {'Terrebonne': 'Terrebonne', 'Sherbrooke': 'Sherbrooke'}
NDG = ((45.4390, 45.4875), (-73.7005, -73.5808))


def load_counts():
    frames = []
    for f in sorted(glob.glob(str(COUNTS / 'comptages_vehicules_cyclistes_pietons*.csv'))):
        if f.endswith('comptages_vehicules_cyclistes_pietons.csv'):
            continue  # same content as _latest
        c = pd.read_csv(f, dtype=str, encoding='utf-8')
        c = c[c.Nom_Intersection.str.contains('Terrebonne|Sherbrooke', case=False, na=False, regex=True)]
        frames.append(c)
    c = pd.concat(frames, ignore_index=True)
    c['street'] = np.where(c.Nom_Intersection.str.contains('Terrebonne', case=False), 'Terrebonne', 'Sherbrooke')
    lat = pd.to_numeric(c.Latitude, errors='coerce'); lon = pd.to_numeric(c.Longitude, errors='coerce')
    c = c[(c.street == 'Terrebonne') | (lat.between(*NDG[0]) & lon.between(*NDG[1]))]  # Sherbrooke only within the NDG box
    for col in MOV + APP + ['Longitude', 'Latitude', 'Heure', 'Minute']:
        c[col] = pd.to_numeric(c[col], errors='coerce').fillna(0)
    c = c.drop_duplicates(subset=['Id_Intersection', 'Date', 'Heure', 'Minute', 'Code_Banque'])
    c['bikes'] = np.maximum(c[MOV].sum(axis=1), c[APP].sum(axis=1)).where(c.Code_Banque == '11', 0)  # older files record bicycles by approach, newer by movement
    c['peds'] = c[APP].sum(axis=1).where(c.Code_Banque == '10', 0)
    c['autos'] = c[MOV].sum(axis=1).where(c.Code_Banque == '0', 0)
    return c


def daily(c):
    g = c.groupby(['street', 'Id_Intersection', 'Nom_Intersection', 'Date'])
    d = g.agg(bikes=('bikes', 'sum'), peds=('peds', 'sum'), autos=('autos', 'sum'), lon=('Longitude', 'first'), lat=('Latitude', 'first'),
              hours=('Heure', lambda h: h.nunique())).reset_index()
    d['full_day'] = d.hours >= 24
    return d


def ci(k, expo):
    lo = stats.chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0; hi = stats.chi2.ppf(0.975, 2 * (k + 1)) / 2
    return lo / expo, hi / expo


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    c = load_counts(); d = daily(c)
    d.to_csv(OUT / 'terrebonne_intersection_counts_by_day.csv', index=False)  # Terrebonne and Sherbrooke (NDG box)
    pd.set_option('display.width', 250)
    print(d[['street', 'Nom_Intersection', 'Date', 'hours', 'full_day', 'bikes', 'peds', 'autos']].to_string(index=False))
    coll = spvm.load(); coll = coll[coll.LOC_X.notna()]
    rows = []
    for (street, iid, name), g in d[d.full_day].groupby(['street', 'Id_Intersection', 'Nom_Intersection']):
        x, y = spvm.mtm8(float(g.lat.iloc[0]), float(g.lon.iloc[0]))
        near = coll[spvm.dist_m(coll, x, y) <= 100]
        for mode, col in (('bike', 'bikes'), ('ped', 'peds')):
            gm = g[g[col] > 0]  # a full day with zero for the mode is a day on which that mode was not counted
            if len(gm) == 0:
                continue
            k = int(near[mode].sum()); daily_mean = float(gm[col].mean()); expo = daily_mean * DAYS / 1e6  # million crossings
            lo, hi = ci(k, expo) if expo > 0 else (np.nan, np.nan)
            rows.append(dict(street=street, intersection=name, count_days=len(gm), count_dates=' '.join(sorted(g.Date)), mode=mode, mean_daily_crossings=round(daily_mean, 1),
                             collisions_within_100m_2012_2021=k, injury_collisions=int((near[mode] & near.severity.isin(['fatal', 'severe', 'light'])).sum()),
                             min_daily=int(gm[col].min()), max_daily=int(gm[col].max()),
                             million_crossings_2012_2021=round(expo, 3), rate_per_million=round(k / expo, 2) if expo > 0 else np.nan, ci_low=round(lo, 2), ci_high=round(hi, 2),
                             rate_if_exposure_at_max_daily=round(k / (gm[col].max() * DAYS / 1e6), 2), rate_if_exposure_at_min_daily=round(k / (gm[col].min() * DAYS / 1e6), 2),
                             all_collisions_within_100m=len(near), mean_daily_autos=round(float(g.autos.mean()), 0)))
    R = pd.DataFrame(rows); R.to_csv(OUT / 'terrebonne_point_rates_2012-2021.csv', index=False)
    print(); print(R[['street', 'intersection', 'count_days', 'mode', 'mean_daily_crossings', 'min_daily', 'max_daily', 'collisions_within_100m_2012_2021', 'injury_collisions', 'million_crossings_2012_2021', 'rate_per_million', 'ci_low', 'ci_high', 'mean_daily_autos']].to_string(index=False))
