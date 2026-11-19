"""Territories for the SPVM collisions: the City's 19 boroughs and the 15 other municipalities of the agglomeration
(City open data « Limites administratives de l'agglomération de Montréal », CC BY 4.0), the City's 58 electoral
districts of 2025 (« Districts électoraux 2025 », CC BY 4.0), and three denominators per territory: area, kilometres
of street (City géobase, classes 0, 5, 6 and 7: local, collector and arterial streets; autoroutes, private ways,
pedestrian ways and projected streets left out; a divided street counts once per carriageway, as the géobase draws
it) and residents (Statistics Canada, 2021 Census, dissemination areas placed by a point inside each area).
The source files are read from ~/Terrebonne-data/third-party/ and are not redistributed here.

  assign(d)        adds 'territory', 'territory_type' and 'district' to the collisions (point in polygon on LOC_LAT/LOC_LONG)
  denominators()   one row per territory: area_km2, street_km, residents_2021
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd

TP = Path.home() / 'Terrebonne-data' / 'third-party'
LIMADMIN = TP / 'montreal-opendata/limadmin/limadmin.json'
DISTRICTS = TP / 'montreal-opendata/districts-electoraux/districts-electoraux-2025.json'
GEOBASE = TP / 'montreal-opendata/geobase_2026-10-14.json'
DA_SHP = TP / 'statcan-census-2021/boundaries/lda_000a21a_e/lda_000a21a_e.shp'
DA_CSV = TP / 'statcan-census-2021/98-401-X2021006_Quebec_DA/98-401-X2021006_English_CSV_data_Quebec.csv'
MTM8 = 'EPSG:32188'  # NAD83 / MTM zone 8, the system of the City's LOC_X / LOC_Y
STREET_CLASSES = (0, 5, 6, 7)
CDN_NDG = 'Côte-des-Neiges–Notre-Dame-de-Grâce'
DASH = {'Côte-des-Neiges-Notre-Dame-de-Grâce': CDN_NDG, 'Rosemont-La Petite-Patrie': 'Rosemont–La Petite-Patrie',
        'Villeray-Saint-Michel-Parc-Extension': 'Villeray–Saint-Michel–Parc-Extension', 'Mercier-Hochelaga-Maisonneuve': 'Mercier–Hochelaga-Maisonneuve',
        'Rivière-des-Prairies-Pointe-aux-Trembles': 'Rivière-des-Prairies–Pointe-aux-Trembles', "L'Île-Bizard-Sainte-Geneviève": 'L’Île-Bizard–Sainte-Geneviève',
        "L'Île-Dorval": 'L’Île-Dorval', "Baie-d'Urfé": 'Baie-D’Urfé'}


def territories() -> gpd.GeoDataFrame:
    g = gpd.read_file(LIMADMIN)[['NOM', 'TYPE', 'geometry']]
    g['territory'] = g.NOM.map(lambda s: DASH.get(s, s)); g['territory_type'] = g.TYPE.map({'Arrondissement': 'borough', 'Ville liée': 'other municipality'})
    return g[['territory', 'territory_type', 'geometry']]


def districts() -> gpd.GeoDataFrame:
    g = gpd.read_file(DISTRICTS)
    return g.rename(columns={'NOM_DISTRICT': 'district', 'NOM_ARR': 'district_borough'})[['district', 'district_borough', 'geometry']]


def assign(d: pd.DataFrame) -> pd.DataFrame:
    pts = gpd.GeoDataFrame(d[[]].copy(), geometry=gpd.points_from_xy(d.LOC_LONG, d.LOC_LAT), crs='EPSG:4326')
    t = gpd.sjoin(pts, territories().to_crs('EPSG:4326'), how='left', predicate='within')
    t = t[~t.index.duplicated()]
    q = gpd.sjoin(pts, districts().to_crs('EPSG:4326'), how='left', predicate='within')
    q = q[~q.index.duplicated()]
    d = d.copy(); d['territory'] = t.territory; d['territory_type'] = t.territory_type; d['district'] = q.district
    return d


def streets() -> gpd.GeoDataFrame:
    g = gpd.read_file(GEOBASE)
    g = g[g.CLASSE.isin(STREET_CLASSES)].to_crs(MTM8)
    g['km'] = g.length / 1000
    return g


def denominators() -> pd.DataFrame:
    T = territories().to_crs(MTM8)
    out = T[['territory', 'territory_type']].copy(); out['area_km2'] = (T.area / 1e6).values
    s = streets(); mid = gpd.GeoDataFrame(s[['km']], geometry=s.interpolate(0.5, normalized=True), crs=MTM8)
    j = gpd.sjoin(mid, T[['territory', 'geometry']], how='left', predicate='within')
    out['street_km'] = out.territory.map(j.groupby('territory').km.sum())
    da = gpd.read_file(DA_SHP, where="PRUID = '24'")[['DAUID', 'geometry']].to_crs(MTM8)
    da = da[da.intersects(T.union_all().envelope)]
    c = pd.read_csv(DA_CSV, encoding='latin-1', usecols=['ALT_GEO_CODE', 'GEO_LEVEL', 'CHARACTERISTIC_ID', 'C1_COUNT_TOTAL'], dtype=str)
    c = c[(c.GEO_LEVEL == 'Dissemination area') & (c.CHARACTERISTIC_ID == '1')]
    da['pop'] = da.DAUID.map(pd.to_numeric(c.set_index('ALT_GEO_CODE').C1_COUNT_TOTAL, errors='coerce')).fillna(0)
    rp = gpd.GeoDataFrame(da[['pop']], geometry=da.representative_point(), crs=MTM8)
    j = gpd.sjoin(rp, T[['territory', 'geometry']], how='inner', predicate='within')
    out['residents_2021'] = out.territory.map(j.groupby('territory')['pop'].sum()).round().astype('Int64')
    return out.reset_index(drop=True)


if __name__ == '__main__':
    print(denominators().sort_values('residents_2021', ascending=False).to_string())
