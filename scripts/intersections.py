"""Collisions by intersection, SPVM open data 2012–2021, and where one intersection stands among the others.

  python3 scripts/intersections.py        ->  derived/tables/intersections_2012-2021.csv
                                              derived/tables/intersection_ranks_<slug>.csv  (one per featured intersection)
                                              derived/tables/intersections_top_cdn_ndg.csv, intersections_top_city.csv

Definition. An intersection is built from the City's géobase (streets of classes 0, 5, 6, 7): a node is a point where
segment ends of at least two street names meet, or at least three segment ends; nodes within MERGE m of each other
(chained) form one intersection, so that the two carriageways of a divided street, or the two service roads of an
expressway crossed by the same street within that distance, are one unit. The name lists the street names met there.
Two counts per intersection, both printed everywhere:
  rec   collisions whose recorded point is within REC m of one of the intersection's nodes, each collision given to
        the nearest intersection only (the police geocoder places a collision reported « at X and Y » on one node)
  r100  collisions within 100 m of any of the intersection's nodes, each counted once per intersection (for a
        single-node intersection, the definition of the Terrebonne / Girouard tables); neighbouring intersections
        closer than 200 m share collisions under this count
Units that carry the same street names and lie within SAME m of each other are then joined: a street crossing the two
service roads of the Décarie expressway is one intersection, as the police record treats it (the City's dictionary
notes that such crossings hold « plusieurs intersections qui portent le même nom » and flags them LOC_IMPRECISION).
Rows located at the centroid of a police district (LOC_COTE_PD = 4, « précision nulle ») are left out of both.
A count at an intersection is a count, not a risk: no traffic, pedestrian or bicycle volume enters.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import spvm, areas

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / 'derived' / 'tables'
MERGE = 30.0; SAME = 120.0; REC = 30.0; R = 100.0
MEASURES = {'all': 'all collisions', 'injury': 'collisions with an injury or a death', 'severe_fatal': 'collisions with a severe injury or a death',
            'ped': 'collisions with a pedestrian victim', 'ped_severe_fatal': 'pedestrian-victim collisions with a severe injury or a death',
            'bike': 'collisions involving a bicycle', 'bike_severe_fatal': 'bicycle collisions with a severe injury or a death'}
FEATURED = {'queen_mary_decarie': ('Queen-Mary', 'Décarie'), 'terrebonne_girouard': ('Terrebonne', 'Girouard')}


def flags(d):
    f = pd.DataFrame(index=d.index)
    f['all'] = True; f['injury'] = d.severity.isin(['fatal', 'severe', 'light']); f['severe_fatal'] = d.severity.isin(['fatal', 'severe'])
    f['ped'] = d.ped; f['ped_severe_fatal'] = d.ped & f.severe_fatal; f['bike'] = d.bike; f['bike_severe_fatal'] = d.bike & f.severe_fatal
    return f


def build_intersections():
    s = areas.streets()
    rows = []
    for geom, name in zip(s.geometry, s.NOM_VOIE):
        for part in (geom.geoms if geom.geom_type == 'MultiLineString' else [geom]):
            c = part.coords
            rows.append((round(c[0][0] * 2) / 2, round(c[0][1] * 2) / 2, name)); rows.append((round(c[-1][0] * 2) / 2, round(c[-1][1] * 2) / 2, name))
    e = pd.DataFrame(rows, columns=['x', 'y', 'name'])
    g = e.groupby(['x', 'y']).agg(ends=('name', 'size'), names=('name', lambda v: tuple(sorted(set(v))))).reset_index()
    nodes = g[(g.ends >= 3) | (g.names.map(len) >= 2)].reset_index(drop=True)
    tree = cKDTree(nodes[['x', 'y']].values)
    pairs = np.array(sorted(tree.query_pairs(MERGE)))
    n = len(nodes)
    _, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)), directed=False)
    nodes['iid'] = lab
    def nm(v):
        c = {}
        for t in v:
            for k in t: c[k] = c.get(k, 0) + 1
        return ' / '.join(sorted(c, key=lambda k: (-c[k], k))[:4])
    U = nodes.groupby('iid').agg(x=('x', 'mean'), y=('y', 'mean'), name=('names', nm)).reset_index()
    # second step: units with the same street names within SAME m of each other are one intersection (the two
    # service-road crossings of an expressway by one street; the City's dictionary: « plusieurs intersections qui portent le même nom »)
    pairs = np.array([p for p in sorted(cKDTree(U[['x', 'y']].values).query_pairs(SAME)) if U.name[p[0]] == U.name[p[1]] and ' / ' in U.name[p[0]]])
    _, lab2 = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(U), len(U))), directed=False)
    nodes['iid'] = lab2[nodes.iid.values]
    I = nodes.groupby('iid').agg(x=('x', 'mean'), y=('y', 'mean'), n_nodes=('x', 'size'), name=('names', nm)).reset_index()
    return nodes, I


def run():
    d = spvm.load(); d = d[d.LOC_X.notna() & (d.LOC_COTE_PD != '4')].copy()
    nodes, I = build_intersections()
    f = flags(d)
    dist, idx = cKDTree(nodes[['x', 'y']].values).query(d[['LOC_X', 'LOC_Y']].values)
    d['iid_rec'] = np.where(dist <= REC, nodes.iid.values[idx], -1)
    rec = f.groupby(d.iid_rec).sum().reindex(I.iid).fillna(0).astype(int)
    ctree = cKDTree(d[['LOC_X', 'LOC_Y']].values); F = f.values
    near = ctree.query_ball_point(nodes[['x', 'y']].values, R); by = {}
    for i, ix in zip(nodes.iid.values, near): by.setdefault(i, set()).update(ix)
    r100 = np.array([F[sorted(by[i])].sum(axis=0) if by[i] else np.zeros(F.shape[1], int) for i in I.iid.values])
    for j, m in enumerate(f.columns):
        I[f'rec_{m}'] = rec[m].values; I[f'r100_{m}'] = r100[:, j]
    pts = gpd.GeoDataFrame(I[['iid']], geometry=gpd.points_from_xy(I.x, I.y), crs=areas.MTM8)
    t = gpd.sjoin(pts, areas.territories().to_crs(areas.MTM8), how='left', predicate='within'); t = t[~t.index.duplicated()]
    q = gpd.sjoin(pts, areas.districts().to_crs(areas.MTM8), how='left', predicate='within'); q = q[~q.index.duplicated()]
    I['territory'] = t.territory; I['territory_type'] = t.territory_type; I['district'] = q.district
    ll = pts.to_crs('EPSG:4326'); I['lat'] = ll.geometry.y.round(6); I['lon'] = ll.geometry.x.round(6)
    I = I[I.territory.notna()].reset_index(drop=True)
    cols = ['iid', 'name', 'territory', 'territory_type', 'district', 'lat', 'lon', 'x', 'y', 'n_nodes'] + [f'{k}_{m}' for k in ('rec', 'r100') for m in f.columns]
    I['x'] = I.x.round(1); I['y'] = I.y.round(1)
    I[cols].to_csv(TABLES / 'intersections_2012-2021.csv', index=False)
    sets = {'cdn_ndg': I.territory == areas.CDN_NDG, 'city': I.territory_type == 'borough', 'agglomeration': pd.Series(True, index=I.index)}
    log = [f'collisions used: {len(d):,} (located, LOC_COTE_PD != 4); recorded within {REC:g} m of an intersection node: {(d.iid_rec >= 0).sum():,}',
           f'géobase nodes: {len(nodes):,}; intersections after merging nodes within {MERGE:g} m and same-name units within {SAME:g} m: {len(I):,} (largest: {I.n_nodes.max()} nodes); in CDN-NDG {sets["cdn_ndg"].sum():,}, City {sets["city"].sum():,}']
    for slug, (a, b) in FEATURED.items():
        hit = I[I.name.str.contains(a) & I.name.str.contains(b)]
        hit = hit.sort_values('rec_all', ascending=False)
        log.append(f'{slug}: {len(hit)} intersection(s) named with both: ' + '; '.join(f'{r.name_} [{r.n_nodes} nodes, rec {r.rec_all}, r100 {r.r100_all}]' for r in hit.rename(columns={'name': 'name_'}).itertuples()))
        me = hit.iloc[0]; out = []
        for k in ('rec', 'r100'):
            for m, lab in MEASURES.items():
                col = f'{k}_{m}'
                for sname, mask in sets.items():
                    S = I[mask]; v = me[col]
                    top = S.sort_values(col, ascending=False).iloc[0]
                    out.append(dict(intersection=me['name'], count=k, measure=m, measure_label=lab, comparison_set=sname, n_in_set=len(S), value=int(v),
                                    rank=int((S[col] > v).sum() + 1), tied_at_rank=int((S[col] == v).sum()), set_median_nonzero=float(S[col][S[col] > 0].median()) if (S[col] > 0).any() else 0.0,
                                    first_name=top['name'], first_territory=top.territory, first_value=int(top[col])))
        pd.DataFrame(out).to_csv(TABLES / f'intersection_ranks_{slug}.csv', index=False)
    for sname, fn in (('cdn_ndg', 'intersections_top_cdn_ndg.csv'), ('city', 'intersections_top_city.csv')):
        S = I[sets[sname]]; out = []
        for k in ('rec', 'r100'):
            for m in MEASURES:
                T = S.sort_values([f'{k}_{m}', f'{k}_all'], ascending=False).head(15)
                for r, row in enumerate(T.itertuples(), 1):
                    out.append(dict(count=k, measure=m, rank=r, name=row.name, territory=row.territory, district=row.district, lat=row.lat, lon=row.lon, value=getattr(row, f'{k}_{m}'), **{c: getattr(row, c) for c in cols[10:]}))
        pd.DataFrame(out).to_csv(TABLES / fn, index=False)
    (TABLES / 'intersections_run.txt').write_text('\n'.join(log) + '\n'); print('\n'.join(log))
    return I


if __name__ == '__main__':
    I = run()
    pd.set_option('display.width', 250)
    r = pd.read_csv(TABLES / 'intersection_ranks_queen_mary_decarie.csv'); print(r.drop(columns=['intersection', 'measure_label']).to_string())
    t = pd.read_csv(TABLES / 'intersections_top_cdn_ndg.csv'); print(t[(t['count'] == 'rec') & t.measure.isin(['all', 'injury', 'ped', 'bike'])][['measure', 'rank', 'name', 'value', 'rec_all', 'rec_injury', 'rec_ped', 'rec_bike']].to_string())
    print(I.n_nodes.value_counts().head(12))
