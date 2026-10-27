"""Does the earlier police record at a location predict a later cyclist or pedestrian death?

Units: 100 m (and 200 m) grid cells on the City's MTM-8 coordinates. Windows: before = 2012–2016, after = 2017–2021
(sensitivity: 2012–2015 / 2016–2021). For every cell with at least one collision in the before window, the before
record (all collisions, injury collisions, collisions with a bicycle, with a pedestrian victim, fatal) and the after
outcome (collisions in which a cyclist / a pedestrian was killed). Two questions:
  1. Were the death sites ordinary? The percentile of each death site's before record among all cells with a record.
  2. Does the before record predict a death? Deaths per 100 cells by decile of the before record (Poisson intervals),
     and the share of deaths caught by the top k % of cells.
Writes CSV tables to derived/location-record-and-later-deaths/ and prints the summary. No exposure measure enters.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import spvm

OUT = Path(__file__).resolve().parent.parent / 'derived' / 'location-record-and-later-deaths'
OUT.mkdir(parents=True, exist_ok=True)
SPLITS = {'2012-2016 | 2017-2021': 2017, '2012-2015 | 2016-2021': 2016}
CELLS = (100, 200)
MEASURES = {'all': 'all collisions', 'injury': 'injury collisions', 'bike': 'collisions with a bicycle', 'ped': 'collisions with a pedestrian victim'}


def poisson_ci(k, n, z=1.96):
    lo = stats.chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
    hi = stats.chi2.ppf(0.975, 2 * (k + 1)) / 2
    return 100 * lo / n, 100 * hi / n


def run(d, cell, split_year, label):
    g = d[d.LOC_X.notna()].copy()
    g['cx'] = np.floor(g.LOC_X / cell).astype(int); g['cy'] = np.floor(g.LOC_Y / cell).astype(int)
    g['injury'] = g.severity.isin(['fatal', 'severe', 'light'])
    g['bike_death'] = g.NB_DECES_VELO.fillna(0) > 0; g['ped_death'] = g.NB_DECES_PIETON.fillna(0) > 0
    before = g[g.year < split_year]; after = g[g.year >= split_year]
    B = before.groupby(['cx', 'cy']).agg(all=('bike', 'size'), injury=('injury', 'sum'), bike=('bike', 'sum'), ped=('ped', 'sum'), fatal=('fatal', 'sum'))
    A = after.groupby(['cx', 'cy']).agg(bike_deaths=('bike_death', 'sum'), ped_deaths=('ped_death', 'sum'), any_fatal=('fatal', 'sum'))
    cells = B.join(A, how='outer').fillna(0).astype(int)
    has_record = cells['all'] > 0
    rows_ord = []; rows_dec = []; rows_top = []
    for mode in ('bike', 'ped'):
        deaths_col = f'{mode}_deaths'
        n_deaths = int(cells[deaths_col].sum()); n_no_record = int(cells.loc[~has_record, deaths_col].sum())
        R = cells[has_record]
        # 1. ordinariness: percentile of each death site's before record among cells with a record
        sites = R[R[deaths_col] > 0]
        for m in MEASURES:
            pct = [stats.percentileofscore(R[m].values, v, kind='weak') for v in sites[m].values]
            w = np.repeat(pct, sites[deaths_col].values)  # one entry per death
            rows_ord.append(dict(split=label, cell_m=cell, mode=mode, measure=m, deaths_in_after_window=n_deaths, deaths_at_cells_with_no_earlier_record=n_no_record,
                                 deaths_at_cells_with_a_record=int(w.size), median_percentile=float(np.median(w)) if w.size else np.nan,
                                 share_in_top_decile=float(np.mean(w >= 90)) if w.size else np.nan, share_in_top_quarter=float(np.mean(w >= 75)) if w.size else np.nan,
                                 share_below_median=float(np.mean(w < 50)) if w.size else np.nan))
        # 2. prediction: deciles of the before injury-collision count (ranked; ties broken by all collisions, then at random with a fixed seed)
        rng = np.random.default_rng(0)
        order = np.lexsort((rng.random(len(R)), R['all'].values, R['injury'].values))
        rank = np.empty(len(R), int); rank[order] = np.arange(len(R))
        dec = (10 * rank // len(R)) + 1
        for q in range(1, 11):
            sel = R[dec == q]; k = int(sel[deaths_col].sum()); n = len(sel)
            lo, hi = poisson_ci(k, n)
            rows_dec.append(dict(split=label, cell_m=cell, mode=mode, decile=q, cells=n, min_injury=int(sel.injury.min()), max_injury=int(sel.injury.max()),
                                 deaths=k, deaths_per_100_cells=100 * k / n, ci_low=lo, ci_high=hi))
        # share of deaths caught by the top k % of cells (by the same ranking), counting deaths at cells with no record as uncaught
        total = n_deaths
        for topk in (1, 2, 5, 10, 20, 30, 50):
            cut = rank >= len(R) * (1 - topk / 100)
            caught = int(R[cut][deaths_col].sum())
            rows_top.append(dict(split=label, cell_m=cell, mode=mode, top_percent_of_cells=topk, cells_flagged=int(cut.sum()), deaths_caught=caught, of_deaths=total, share=caught / total if total else np.nan))
    return pd.DataFrame(rows_ord), pd.DataFrame(rows_dec), pd.DataFrame(rows_top), len(cells), int(has_record.sum())


if __name__ == '__main__':
    d = spvm.load()
    O, D, Tk = [], [], []
    for label, y in SPLITS.items():
        for cell in CELLS:
            o, dec, tk, ncell, nrec = run(d, cell, y, label); O.append(o); D.append(dec); Tk.append(tk)
            print(f'{label} cell {cell} m: {ncell} cells with any collision, {nrec} with a before record')
    O = pd.concat(O); D = pd.concat(D); Tk = pd.concat(Tk)
    O.to_csv(OUT / 'death_sites_percentiles.csv', index=False); D.to_csv(OUT / 'deaths_by_decile.csv', index=False); Tk.to_csv(OUT / 'deaths_caught_by_top_cells.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    main = (O.split == '2012-2016 | 2017-2021') & (O.cell_m == 100)
    print(O[main][['mode', 'measure', 'deaths_in_after_window', 'deaths_at_cells_with_no_earlier_record', 'median_percentile', 'share_in_top_decile', 'share_in_top_quarter', 'share_below_median']].round(2).to_string(index=False))
    m2 = (D.split == '2012-2016 | 2017-2021') & (D.cell_m == 100)
    print(D[m2][['mode', 'decile', 'cells', 'min_injury', 'max_injury', 'deaths', 'deaths_per_100_cells', 'ci_low', 'ci_high']].round(2).to_string(index=False))
    m3 = (Tk.split == '2012-2016 | 2017-2021') & (Tk.cell_m == 100)
    print(Tk[m3][['mode', 'top_percent_of_cells', 'cells_flagged', 'deaths_caught', 'of_deaths', 'share']].round(2).to_string(index=False))


def terrebonne_table(d, split_year=2017, cell=100):
    """Where rue de Terrebonne's cells sit in the before-record ranking (percentile among cells with a record)."""
    g = d[d.LOC_X.notna()].copy()
    g['cx'] = np.floor(g.LOC_X / cell).astype(int); g['cy'] = np.floor(g.LOC_Y / cell).astype(int)
    g['injury'] = g.severity.isin(['fatal', 'severe', 'light'])
    before = g[g.year < split_year]
    B = before.groupby(['cx', 'cy']).agg(all=('bike', 'size'), injury=('injury', 'sum'), bike=('bike', 'sum'), ped=('ped', 'sum'))
    rows = []
    pts = {'Terrebonne / Girouard': (45.47833, -73.62487), 'Terrebonne / Draper': (45.47405, -73.62872), 'Terrebonne / Royal': (45.47334, -73.62936),
           'Terrebonne / Grand': (45.47054, -73.63175), 'Terrebonne / Kensington': (45.46990, -73.63231), 'Terrebonne / Cavendish': (45.46674, -73.63524), 'Terrebonne / King Edward': (45.46354, -73.63784)}
    for name, (lat, lon) in pts.items():
        x, y = spvm.mtm8(lat, lon); key = (int(np.floor(x / cell)), int(np.floor(y / cell)))
        rec = B.loc[key] if key in B.index else pd.Series({'all': 0, 'injury': 0, 'bike': 0, 'ped': 0})
        r = dict(location=name, cell_m=cell, before_window=f'2012-{split_year - 1}', **{k: int(rec[k]) for k in ('all', 'injury', 'bike', 'ped')})
        for m in ('all', 'injury', 'bike'):
            r[f'percentile_{m}'] = round(float(stats.percentileofscore(B[m].values, rec[m], kind='weak')), 1) if rec['all'] > 0 else np.nan
        r['in_top_decile_by_injury'] = bool(rec['all'] > 0 and stats.percentileofscore(B['injury'].values, rec['injury'], kind='weak') >= 90)
        rows.append(r)
    # the whole street: cells within 25 m of the axis, how many hold a record and how many are in the top decile
    A = spvm.mtm8(45.45974, -73.64080); Bp = spvm.mtm8(45.47833, -73.62487)
    near = g[spvm.dist_to_segment_m(g, *A) <= 25] if False else g[spvm.dist_to_segment_m(g, A, Bp) <= 25]
    keys = set(zip(near.cx, near.cy)); rec_keys = [k for k in keys if k in B.index]
    top = sum(1 for k in rec_keys if stats.percentileofscore(B['injury'].values, B.loc[k, 'injury'], kind='weak') >= 90)
    rows.append(dict(location=f'rue de Terrebonne, cells within 25 m of the axis: {len(keys)} cells with any collision 2012-2021, {len(rec_keys)} with a record before {split_year}, {top} of them in the top decile by injury count', cell_m=cell, before_window=f'2012-{split_year - 1}'))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    T = terrebonne_table(spvm.load())
    T.to_csv(OUT / 'terrebonne_cells_percentiles.csv', index=False)
    pd.set_option('display.width', 250); print(T.to_string(index=False))
