"""Tables for the road-safety deck on Côte-des-Neiges–Notre-Dame-de-Grâce (SPVM open data 2012–2021).

  python3 scripts/cdn_ndg_tables.py   ->  derived/tables/
      boroughs_2012-2021.csv                 the 19 boroughs and the 15 other municipalities: counts by severity and mode, the three
                                             denominators (areas.py), counts per 100 km of street and per 10,000 residents, ranks among boroughs
      cdn_ndg_districts_2012-2021.csv        the borough's five electoral districts (2025 boundaries)
      cdn_ndg_yearly_2012-2021.csv           the borough and the City (19 boroughs) by year
      cdn_ndg_other_vehicle_2012-2021.csv    bicycle and pedestrian collisions in the borough by the other vehicle and severity
      queen_mary_intersections_2012-2021.csv the intersections of chemin Queen-Mary inside the borough, east to west (from intersections.py)
      queen_mary_corridor_2012-2021.csv      collisions within 30 m of chemin Queen-Mary inside the borough: summary rows
      queen_mary_decarie_collisions_2012-2021.csv   one row per collision within 100 m of the Queen-Mary / Décarie nodes
      queen_mary_decarie_<by>.csv            the same by year, hour, weekday, kind of collision, road category, mode and severity
Counts are collisions, not victims, unless a column says deaths. A collision is in a territory when its recorded point falls
inside the polygon; rows without coordinates (11) and rows placed at a police district's centroid (LOC_COTE_PD = 4) are counted
separately. Rates use residents and street length because no traffic, pedestrian or bicycle volume exists by territory: they are
not risks per trip or per kilometre travelled.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree
import spvm, areas, intersections
from maps import other_vehicle

TABLES = Path(__file__).resolve().parent.parent / 'derived' / 'tables'
B = areas.CDN_NDG
GENRE = {'31': 'another road vehicle', '32': 'a pedestrian', '33': 'a cyclist', '59': 'a fixed object', '40': 'a fixed object', '41': 'a fixed object', '42': 'a fixed object', '43': 'a fixed object',
         '44': 'a fixed object', '45': 'a fixed object', '46': 'a fixed object', '47': 'a fixed object', '48': 'a fixed object', '49': 'a fixed object', '50': 'a fixed object', '51': 'a fixed object', '52': 'a fixed object',
         '53': 'a fixed object', '54': 'a fixed object', '55': 'a fixed object', '56': 'a fixed object', '57': 'a fixed object', '58': 'a fixed object'}
CATEG = {'11': 'ramp, collector or service road of an expressway', '12': 'numbered route', '13': 'main artery', '14': 'residential street', '15': 'local road', '16': 'lane (ruelle)', '19': 'other public road',
         '21': 'parking lot', '22': 'private land', '23': 'private road', '24': 'forest road', '25': 'trail', '29': 'other off-road'}
DAY = {'LU': 'Mon', 'MA': 'Tue', 'ME': 'Wed', 'JE': 'Thu', 'VE': 'Fri', 'SA': 'Sat', 'DI': 'Sun'}


def counts(s: pd.DataFrame) -> dict:
    sev = s.severity
    inj = sev.isin(['fatal', 'severe', 'light']); sf = sev.isin(['fatal', 'severe'])
    z = lambda c: int(s[c].fillna(0).sum())
    return dict(collisions=len(s), fatal=int((sev == 'fatal').sum()), severe=int((sev == 'severe').sum()), light=int((sev == 'light').sum()), damage=int((sev == 'damage').sum()),
                injury=int(inj.sum()), severe_fatal=int(sf.sum()), ped=int(s.ped.sum()), ped_severe_fatal=int((s.ped & sf).sum()), deaths_ped=z('NB_DECES_PIETON'),
                bike=int(s.bike.sum()), bike_severe_fatal=int((s.bike & sf).sum()), deaths_bike=z('NB_DECES_VELO'), moto=int(s.moto.sum()), deaths_all=z('NB_MORTS'))


def main():
    d = areas.assign(spvm.load())
    d['centroid'] = d.LOC_COTE_PD == '4'
    log = [f'rows {len(d):,}; without coordinates {d.LOC_LAT.isna().sum()}; at a police-district centroid (LOC_COTE_PD = 4) {int(d.centroid.sum()):,}, of which in {B}: {int((d.centroid & (d.territory == B)).sum())}',
           f'outside every polygon (with coordinates): {int((d.territory.isna() & d.LOC_LAT.notna()).sum())}']
    use = d[d.territory.notna() & ~d.centroid]
    # 1. territories
    den = areas.denominators().set_index('territory')
    T = pd.DataFrame({t: counts(s) for t, s in use.groupby('territory')}).T.join(den)
    for c in ('collisions', 'injury', 'severe_fatal'):
        T[f'{c}_per_100km'] = 100 * T[c] / T.street_km
    for c in ('injury', 'severe_fatal', 'ped', 'ped_severe_fatal', 'bike', 'bike_severe_fatal'):
        T[f'{c}_per_10k_res'] = 1e4 * T[c] / T.residents_2021.astype(float)
    bo = T.territory_type == 'borough'
    rate_cols = [c for c in T.columns if c.endswith('_per_100km') or c.endswith('_per_10k_res')]
    for c in ['collisions', 'injury', 'severe_fatal', 'ped', 'ped_severe_fatal', 'bike', 'bike_severe_fatal'] + rate_cols:
        T.loc[bo, f'rank_{c}'] = T.loc[bo, c].astype(float).rank(ascending=False, method='min').astype(int)
    for c in [c for c in T.columns if c.startswith('rank_')]: T[c] = T[c].astype('Int64')
    T[rate_cols] = T[rate_cols].astype(float).round(2); T['area_km2'] = T.area_km2.round(2); T['street_km'] = T.street_km.round(1)
    T.index.name = 'territory'
    city = pd.Series(counts(use[use.territory_type == 'borough'])); cden = den[den.territory_type == 'borough']
    T.sort_values(['territory_type', 'collisions'], ascending=[True, False]).to_csv(TABLES / 'boroughs_2012-2021.csv')
    log.append(f'City (19 boroughs): {city.collisions:,} collisions, street {cden.street_km.sum():,.0f} km, residents {int(cden.residents_2021.sum()):,}')
    s = use[use.territory == B]
    # 2. districts
    D = pd.DataFrame({k: counts(v) for k, v in s.groupby('district')}).T; D.index.name = 'district'
    dg = areas.districts(); dg = dg[dg.district.isin(D.index) & dg.district_borough.str.contains('Neiges')].to_crs(areas.MTM8)
    st = areas.streets(); mid = gpd.GeoDataFrame(st[['km']], geometry=st.interpolate(0.5, normalized=True), crs=areas.MTM8)
    D['street_km'] = gpd.sjoin(mid, dg[['district', 'geometry']], predicate='within').groupby('district').km.sum().round(1)
    D['area_km2'] = (dg.set_index('district').area / 1e6).round(2)
    D.sort_values('collisions', ascending=False).to_csv(TABLES / 'cdn_ndg_districts_2012-2021.csv')
    log.append(f'{B}: {len(s):,} collisions inside the polygon; districts sum {int(D.collisions.sum()):,}')
    # 3. yearly
    Y = pd.concat({'borough': pd.DataFrame({y: counts(v) for y, v in s.groupby('year')}).T, 'city': pd.DataFrame({y: counts(v) for y, v in use[use.territory_type == 'borough'].groupby('year')}).T}, names=['scope', 'year'])
    Y.to_csv(TABLES / 'cdn_ndg_yearly_2012-2021.csv')
    # 4. other vehicle
    rows = []
    for mode in ('bike', 'ped'):
        m = s[s[mode]]; ov = other_vehicle(m)
        for k in ('car', 'heavy', 'bus', 'other'):
            mm = m[ov == k]
            rows.append(dict(mode=mode, other_vehicle=k, collisions=len(mm), fatal=int((mm.severity == 'fatal').sum()), severe=int((mm.severity == 'severe').sum()), light=int((mm.severity == 'light').sum()), damage=int((mm.severity == 'damage').sum())))
    pd.DataFrame(rows).to_csv(TABLES / 'cdn_ndg_other_vehicle_2012-2021.csv', index=False)
    # 5. chemin Queen-Mary inside the borough
    I = pd.read_csv(TABLES / 'intersections_2012-2021.csv')
    Q = I[(I.territory == B) & I.name.str.contains('Queen-Mary')].sort_values('lon', ascending=False)
    Q.drop(columns=['iid', 'x', 'y', 'territory_type']).to_csv(TABLES / 'queen_mary_intersections_2012-2021.csv', index=False)
    qm = st[st.NOM_VOIE == 'Queen-Mary']
    P = gpd.GeoSeries(gpd.points_from_xy(s.LOC_X, s.LOC_Y), index=s.index, crs=areas.MTM8)
    near = P.distance(qm.union_all()) <= 30
    c = s[near]
    qm_in = gpd.clip(qm, areas.territories().to_crs(areas.MTM8).query('territory == @B'))
    km_axis = qm_in.length.sum() / 1000
    pd.DataFrame([dict(selection='within 30 m of chemin Queen-Mary, inside the borough', geobase_km_carriageways=round(km_axis, 2), **counts(c)),
                  dict(selection='Queen-Mary in a street field (RUE_ACCDN or ACCDN_PRES_DE), inside the borough', geobase_km_carriageways=round(km_axis, 2), **counts(s[spvm.on_street(s, 'QUEEN MARY', 'QUEEN-MARY')]))]).to_csv(TABLES / 'queen_mary_corridor_2012-2021.csv', index=False)
    c.assign(other_vehicle=other_vehicle(c))[['NO_SEQ_COLL', 'date', 'severity', 'ped', 'bike', 'moto', 'other_vehicle', 'RUE_ACCDN', 'ACCDN_PRES_DE', 'LOC_LAT', 'LOC_LONG']].to_csv(TABLES / 'queen_mary_corridor_collisions_2012-2021.csv', index=False)
    # 6. Queen-Mary / Décarie
    nodes, I2 = intersections.build_intersections()
    iid = I2[I2.name.str.contains('Queen-Mary') & I2.name.str.contains('Décarie')].iid.values
    nn = nodes[nodes.iid.isin(iid)][['x', 'y']].values
    dist, _ = cKDTree(nn).query(use[['LOC_X', 'LOC_Y']].fillna(0).values)
    h = use[dist <= 100].copy(); h['dist_m'] = dist[dist <= 100].round(1)
    h['hour'] = pd.to_numeric(h.HEURE_ACCDN.str[:2], errors='coerce')
    h['kind'] = h.CD_GENRE_ACCDN.map(GENRE).fillna('other or not stated'); h['road_category'] = h.CD_CATEG_ROUTE.map(CATEG).fillna('not stated')
    h['weekday'] = h.JR_SEMN_ACCDN.map(DAY); h['other_vehicle'] = other_vehicle(h)
    h['mode'] = np.where(h.ped, 'pedestrian victim', np.where(h.bike, 'bicycle involved', np.where(h.moto, 'motorcycle involved', 'motor vehicles only')))
    h['at_recorded_point'] = h.dist_m <= 30
    h[['NO_SEQ_COLL', 'date', 'weekday', 'hour', 'severity', 'mode', 'ped', 'bike', 'moto', 'other_vehicle', 'kind', 'road_category', 'CD_LOCLN_ACCDN', 'RUE_ACCDN', 'ACCDN_PRES_DE', 'NO_CIVIQ_ACCDN',
       'NB_MORTS', 'NB_BLESSES_GRAVES', 'NB_BLESSES_LEGERS', 'LOC_LAT', 'LOC_LONG', 'LOC_IMPRECISION', 'dist_m', 'at_recorded_point']].sort_values('date').to_csv(TABLES / 'queen_mary_decarie_collisions_2012-2021.csv', index=False)
    pd.DataFrame(nn, columns=['x', 'y']).to_csv(TABLES / 'queen_mary_decarie_nodes.csv', index=False)
    def by(col, order=None):
        t = pd.DataFrame({k: counts(v) for k, v in h.groupby(col)}).T[['collisions', 'injury', 'severe_fatal', 'fatal', 'ped', 'bike', 'moto']]; t.index.name = col
        if order is not None: t = t.reindex(order).fillna(0).astype(int)
        t.to_csv(TABLES / f'queen_mary_decarie_by_{col}.csv')
    by('year', range(2012, 2022)); by('hour', range(24)); by('weekday', list(DAY.values())); by('kind'); by('road_category'); by('mode'); by('severity', spvm.SEVERITY_ORDER)
    tr = h[h.rue.str.contains('AUT|A-15|A 15') | h.pres.str.contains('AUT ')]
    log.append(f'Queen-Mary / Décarie, within 100 m of its {len(nn)} nodes: {len(h)} collisions, {int(h.at_recorded_point.sum())} within 30 m; hour not stated {int(h.hour.isna().sum())}; '
               f'flagged LOC_IMPRECISION {int((h.LOC_IMPRECISION == "O").sum())}; with an autoroute named in a street field {len(tr)}; road category 11 (ramp, collector or service road) {int((h.CD_CATEG_ROUTE == "11").sum())}, 12 (numbered route) {int((h.CD_CATEG_ROUTE == "12").sum())}')
    log.append(f'Queen-Mary corridor in the borough: {len(c)} collisions within 30 m of {km_axis:.2f} km of géobase carriageway')
    (TABLES / 'cdn_ndg_tables_run.txt').write_text('\n'.join(log) + '\n'); print('\n'.join(log))


if __name__ == '__main__':
    main()
