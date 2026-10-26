"""Issue #40: for each collision with a cyclist killed (SPVM open data, 2012–2021), the earlier record at the
same location under three definitions (same street name; within 100 m; within 250 m), by severity and by mode,
with rue de Terrebonne measured the same way. Writes CSV tables; prints the summary."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import spvm, pandas as pd, numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / 'derived' / 'cyclist-fatality-locations'
OUT.mkdir(parents=True, exist_ok=True)
START = pd.Timestamp('2012-01-01')
d = spvm.load()
d = d[d.date.notna()].copy()
RADII = (100, 250)
MODES = ['ped', 'bike', 'moto', 'motor_only']


def record(prior: pd.DataFrame, prefix: str) -> dict:
    """Counts of earlier collisions by severity and mode, plus victims."""
    r = {f'{prefix}_n': len(prior), f'{prefix}_fatal_any': int(prior.fatal.sum())}
    for sev in spvm.SEVERITY_ORDER:
        r[f'{prefix}_{sev}'] = int((prior.severity == sev).sum())
    for m in MODES:
        r[f'{prefix}_{m}'] = int(prior[m].sum())
    r[f'{prefix}_ped_fatal'] = int((prior.fatal & prior.ped).sum())
    r[f'{prefix}_bike_fatal'] = int((prior.fatal & prior.bike).sum())
    r[f'{prefix}_deaths'] = int(prior.NB_MORTS.fillna(0).sum())
    r[f'{prefix}_serious_inj'] = int(prior.NB_BLESSES_GRAVES.fillna(0).sum())
    return r


def location_record(d, x, y, rue_key, pres_key, before, exclude_id=None) -> dict:
    prior = d[d.date < before]
    if exclude_id is not None:
        prior = prior[prior.NO_SEQ_COLL != exclude_id]
    out = {'years_of_record': round((before - START).days / 365.25, 2)}
    # (a) same street name: the fatal collision's RUE_ACCDN key equals either street key of the earlier collision
    if rue_key:
        same = prior[(prior.rue_key == rue_key) | (prior.pres_key == rue_key)]
    else:
        same = prior.iloc[0:0]
    out['name_n'] = len(same); out['name_fatal_any'] = int(same.fatal.sum())
    # (a') same intersection: both keys
    if rue_key and pres_key:
        keys = {rue_key, pres_key}
        inter = prior[[(a in keys and b in keys and a != b) for a, b in zip(prior.rue_key, prior.pres_key)]]
    else:
        inter = prior.iloc[0:0]
    out['intersection_n'] = len(inter); out['intersection_fatal_any'] = int(inter.fatal.sum())
    # (b, c) radii on the City's metric coordinates
    if not (np.isnan(x) or np.isnan(y)):
        dist = spvm.dist_m(prior, x, y)
        for r in RADII:
            out.update(record(prior[dist <= r], f'd{r}'))
    return out


# --- the 31 collisions with a cyclist killed ---
K = d[d.NB_DECES_VELO.fillna(0) > 0].sort_values('date')
rows = []
for _, k in K.iterrows():
    row = {'id': k.NO_SEQ_COLL, 'date': k.date.date().isoformat(), 'street': k.RUE_ACCDN, 'near': k.ACCDN_PRES_DE if isinstance(k.ACCDN_PRES_DE, str) else '',
           'civic': k.NO_CIVIQ_ACCDN if isinstance(k.NO_CIVIQ_ACCDN, str) else '', 'municipality_code': k.CD_MUNCP,
           'location_imprecise': k.LOC_IMPRECISION == 'O', 'lat': k.LOC_LAT, 'lon': k.LOC_LONG,
           'other_vehicle': 'heavy truck' if k.nb_camionLourd_tractRoutier > 0 else 'bus' if k.nb_tous_autobus_minibus > 0 else 'car/light truck' if k.nb_automobile_camion_leger > 0 else 'none recorded'}
    row.update(location_record(d, k.LOC_X, k.LOC_Y, k.rue_key, k.pres_key, k.date, exclude_id=k.NO_SEQ_COLL))
    rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(OUT / 'cyclist_fatality_locations_2012-2021.csv', index=False)

# --- Terrebonne on the same scale ---
END = pd.Timestamp('2022-01-01')
pts = {'Terrebonne / Girouard': (45.47833, -73.62487), 'Terrebonne / Grand': (45.47054, -73.63175),
       'Terrebonne / Cavendish': (45.46674, -73.63524), 'Terrebonne / Kensington': (45.46990, -73.63231)}
trows = []
# the Sept 3, 2019 pedestrian death at Girouard, measured like the 31
g = d[spvm.on_street(d, 'TERREBONNE') & d.fatal]
assert len(g) == 1, len(g)
g = g.iloc[0]
r = {'location': 'Terrebonne / Girouard, before the Sept 3, 2019 pedestrian death', 'period': f'2012-01-01 to {g.date.date()}'}
r.update(location_record(d, g.LOC_X, g.LOC_Y, g.rue_key, g.pres_key, g.date, exclude_id=g.NO_SEQ_COLL)); trows.append(r)
for name, (lat, lon) in pts.items():
    x, y = spvm.mtm8(lat, lon)
    r = {'location': name + ', whole record', 'period': '2012-01-01 to 2021-12-31'}
    r.update(location_record(d, x, y, 'TERREBONNE', '', END)); trows.append(r)
# whole street, by name and by axis corridor
whole = d[spvm.on_street(d, 'TERREBONNE', 'TERREBONE')]
r = {'location': 'rue de Terrebonne, by street name (RUE_ACCDN or ACCDN_PRES_DE)', 'period': '2012-01-01 to 2021-12-31', 'years_of_record': 10.0}
r.update(record(whole, 'street')); trows.append(r)
A = spvm.mtm8(45.45974, -73.64080); B = spvm.mtm8(45.47833, -73.62487)
axis = d[spvm.dist_to_segment_m(d, A, B) <= 25]
r = {'location': 'rue de Terrebonne, within 25 m of the lane axis (Belmore to Girouard)', 'period': '2012-01-01 to 2021-12-31', 'years_of_record': 10.0}
r.update(record(axis, 'street')); trows.append(r)
TT = pd.DataFrame(trows)
TT.to_csv(OUT / 'terrebonne_same_scale_2012-2021.csv', index=False)

# --- summary ---
n = len(T)
S = []
for label, col in [('same street name', 'name_fatal_any'), ('same intersection (both street names)', 'intersection_fatal_any'),
                   ('within 100 m', 'd100_fatal_any'), ('within 250 m', 'd250_fatal_any')]:
    S.append({'definition': label, 'with an earlier fatal collision': int((T[col] > 0).sum()), 'without': int((T[col] == 0).sum()), 'of': n})
SUM = pd.DataFrame(S); SUM.to_csv(OUT / 'summary_earlier_fatal.csv', index=False)
def record_summary(T, label):
    Q = []
    for r_ in RADII:
        for what, col in [('all collisions', f'd{r_}_n'), ('injury collisions (fatal, severe or light)', None), ('collisions with a bicycle', f'd{r_}_bike'),
                      ('collisions with a pedestrian victim', f'd{r_}_ped'), ('fatal collisions', f'd{r_}_fatal_any'), ('severe-injury collisions', f'd{r_}_severe')]:
            v = (T[f'd{r_}_fatal'] + T[f'd{r_}_severe'] + T[f'd{r_}_light']) if col is None else T[col]
            rate = v / T.years_of_record.replace(0, np.nan)
            Q.append({'cases': label, 'n': len(T), 'radius_m': r_, 'measure': what, 'zero': int((v == 0).sum()), 'median': float(v.median()), 'q1': float(v.quantile(.25)), 'q3': float(v.quantile(.75)),
                      'min': int(v.min()), 'max': int(v.max()), 'median_per_year': round(float(rate.median()), 2)})
    return pd.DataFrame(Q)
QQ = pd.concat([record_summary(T, 'all 31 deaths, 2012-2021'), record_summary(T[T.years_of_record >= 3], 'deaths from 2015 on (at least 3 years of record)')], ignore_index=True)
QQ.to_csv(OUT / 'summary_earlier_record.csv', index=False)
# markdown rendering for the README and the note
def md(df):
    cols = list(df.columns); lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows(): lines.append('| ' + ' | '.join(str(v) for v in r.values) + ' |')
    return '\n'.join(lines)
show31 = pd.DataFrame({'Date': T.date, 'Street': T.street, 'Near': T.near, 'Other vehicle': T.other_vehicle, 'Years of record': T.years_of_record,
    'Earlier fatal: street / intersection / 100 m / 250 m': [f'{a} / {b} / {c} / {e}' for a, b, c, e in zip(T.name_fatal_any, T.intersection_fatal_any, T.d100_fatal_any, T.d250_fatal_any)],
    'Within 100 m: all / injury / bike / ped': [f'{a} / {b} / {c} / {e}' for a, b, c, e in zip(T.d100_n, T.d100_fatal + T.d100_severe + T.d100_light, T.d100_bike, T.d100_ped)],
    'Within 250 m: all / injury / bike / ped': [f'{a} / {b} / {c} / {e}' for a, b, c, e in zip(T.d250_n, T.d250_fatal + T.d250_severe + T.d250_light, T.d250_bike, T.d250_ped)],
    'Location flagged imprecise': ['yes' if v else '' for v in T.location_imprecise]})
with open(OUT / 'tables.md', 'w') as fh:
    fh.write('# Tables (generated by scripts/cyclist_fatality_locations.py)\n\n## The 31 collisions with a cyclist killed, SPVM open data 2012-2021\n\n' + md(show31) + '\n\n## Earlier fatal collision of any kind at the same location\n\n' + md(SUM) + '\n\n## Earlier record at the location\n\n' + md(QQ) + '\n\n## Rue de Terrebonne on the same scale\n\n' + md(TT.fillna('')) + '\n')

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
show = T[['date', 'street', 'near', 'other_vehicle', 'years_of_record', 'name_n', 'name_fatal_any', 'intersection_n', 'intersection_fatal_any',
          'd100_n', 'd100_fatal_any', 'd100_severe', 'd100_light', 'd100_bike', 'd100_ped', 'd250_n', 'd250_fatal_any', 'd250_severe', 'd250_light', 'd250_bike', 'd250_ped', 'location_imprecise']]
print(show.to_string(index=False)); print()
print(SUM.to_string(index=False)); print(); print(QQ.to_string(index=False)); print()
print(TT[['location', 'years_of_record', 'name_n', 'name_fatal_any', 'd100_n', 'd100_fatal_any', 'd100_severe', 'd100_light', 'd100_bike', 'd100_ped', 'd250_n', 'd250_fatal_any', 'd250_severe', 'd250_light', 'd250_bike', 'd250_ped', 'street_n', 'street_fatal_any', 'street_severe', 'street_light', 'street_bike', 'street_ped']].to_string(index=False))
print('deaths per year', K.groupby('year').size().reindex(range(2012, 2022), fill_value=0).tolist(), '| rows without coordinates in the file', int(d.LOC_X.isna().sum()))
