"""Load the SPVM collisions open data (Ville de Montréal, « Collisions routières », 2012–2021).

One module for every script in the repository: loading, typed columns, the street-name
normalisation, the mode and severity classes, the map extents and the distance helpers.
Nothing here evaluates; the definitions are stated so that every table can be re-run.
"""
from __future__ import annotations
import math, re, unicodedata
from pathlib import Path
import numpy as np, pandas as pd

DEFAULT_CSV = Path.home() / 'Terrebonne-data/third-party/montreal-opendata/collisions/collisions_routieres_MTL.csv'

NUMERIC = ['LOC_LAT', 'LOC_LONG', 'LOC_X', 'LOC_Y', 'AN', 'NB_VEH_IMPLIQUES_ACCDN', 'NB_MORTS', 'NB_BLESSES_GRAVES',
           'NB_BLESSES_LEGERS', 'NB_VICTIMES_TOTAL', 'nb_automobile_camion_leger', 'nb_camionLourd_tractRoutier',
           'nb_outil_equipement', 'nb_tous_autobus_minibus', 'nb_bicyclette', 'nb_cyclomoteur', 'nb_motocyclette',
           'nb_taxi', 'nb_urgence', 'nb_motoneige', 'nb_VHR', 'nb_autres_types', 'nb_veh_non_precise',
           'NB_DECES_PIETON', 'NB_BLESSES_PIETON', 'NB_VICTIMES_PIETON', 'NB_DECES_MOTO', 'NB_BLESSES_MOTO',
           'NB_VICTIMES_MOTO', 'NB_DECES_VELO', 'NB_BLESSES_VELO', 'NB_VICTIMES_VELO', 'VITESSE_AUTOR']

# Severity: the collision's own GRAVITE field (the most severe outcome of the collision, SAAQ coding).
SEVERITY = {'Mortel': 'fatal', 'Grave': 'severe', 'Léger': 'light',
            'Dommages matériels seulement': 'damage',
            'Dommages matériels inférieurs au seuil de rapportage': 'damage'}
SEVERITY_ORDER = ['fatal', 'severe', 'light', 'damage']

# Map extents of the 2024 figures (lat/long boxes as written in the retired MATLAB options; strict bounds).
EXTENTS = {
    'montreal': dict(name='Montréal', box=None),
    'ndg': dict(name='NDG box', box=((45.4390, 45.4875), (-73.7005, -73.5808))),
    'h4c': dict(name='H4C box', box=((45.4645, 45.4850), (-73.5870, -73.5640))),
    'terrebonne_narrow': dict(name='Terrebonne narrow box', box=((45.45740, 45.47992), (-73.64397, -73.62163))),
}

GENERIC = {'RUE', 'AV', 'AVE', 'AVEN', 'AVENUE', 'BD', 'BOUL', 'BOULEVARD', 'CH', 'CHEMIN', 'PL', 'PLACE', 'VOIE',
           'CROIS', 'CROISSANT', 'CRESCENT', 'CRES', 'ROUTE', 'RTE', 'RD', 'ROAD', 'ST', 'STREET', 'MONTEE', 'MTEE',
           'TERRASSE', 'TSSE', 'CARRE', 'SQUARE', 'COURS', 'ALLEE', 'PROM', 'PROMENADE', 'DR', 'DRIVE', 'DE', 'DU', 'DES',
           'DE LA', 'LA', 'LE', 'LES', 'L', 'D', 'O', 'E', 'OUEST', 'EST', 'N', 'S', 'NORD', 'SUD', 'FACE', 'OPP', 'STAT',
           'PRES', 'ET', 'A', 'AU'}
SAINT = {'ST': 'SAINT', 'STE': 'SAINTE', 'S': 'SAINT', 'RIV': 'RIVIERE'}
ABBREV = {'N-D': 'NOTRE DAME', 'N D': 'NOTRE DAME'}


def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))


def street_key(s) -> str:
    """Crude key for « same street »: upper case, no accents, hyphens as spaces, generic words removed,
    ST/STE expanded, tokens sorted. Empty when the field is blank."""
    if not isinstance(s, str) or not s.strip():
        return ''
    s = strip_accents(s.upper())
    for k, v in ABBREV.items():
        s = re.sub(r'(^|\s)' + re.escape(k) + r'(?=\s|$)', r'\1' + v, s)
    s = s.replace('-', ' ').replace("'", ' ').replace('.', ' ')
    toks = [t for t in re.split(r'\s+', s) if t]
    toks = [SAINT.get(t, t) for t in toks]
    toks = [t for t in toks if t not in GENERIC and not t.isdigit()]
    return ' '.join(sorted(toks))


def load(path: Path | str = DEFAULT_CSV) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False, dtype=str)
    for c in NUMERIC:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d['date'] = pd.to_datetime(d['DT_ACCDN'], format='%Y/%m/%d', errors='coerce')
    d['year'] = d['date'].dt.year
    d['month'] = d['date'].dt.month
    d['severity'] = d['GRAVITE'].map(SEVERITY)
    d['rue'] = d['RUE_ACCDN'].fillna('').str.upper().str.strip()
    d['pres'] = d['ACCDN_PRES_DE'].fillna('').str.upper().str.strip()
    d['rue_key'] = d['rue'].map(street_key)
    d['pres_key'] = d['pres'].map(street_key)
    z = lambda c: d[c].fillna(0)
    d['bike'] = (z('nb_bicyclette') > 0) | (z('NB_VICTIMES_VELO') > 0)
    d['ped'] = z('NB_VICTIMES_PIETON') > 0
    d['moto'] = (z('nb_motocyclette') > 0) | (z('NB_VICTIMES_MOTO') > 0)
    d['motor_only'] = ~(d['bike'] | d['ped'] | d['moto'])
    d['vru'] = d['bike'] | d['ped']  # vulnerable road users: a bicycle involved or a pedestrian victim
    d['fatal'] = z('NB_MORTS') > 0
    return d


def in_box(d: pd.DataFrame, box) -> pd.Series:
    (lat0, lat1), (lon0, lon1) = box
    return (d.LOC_LAT > lat0) & (d.LOC_LAT < lat1) & (d.LOC_LONG > lon0) & (d.LOC_LONG < lon1)


def on_street(d: pd.DataFrame, *needles: str) -> pd.Series:
    """Substring match on RUE_ACCDN or ACCDN_PRES_DE (upper case), as the 2024 figures did."""
    m = pd.Series(False, index=d.index)
    for n in needles:
        n = n.upper()
        m |= d.rue.str.contains(n, regex=False) | d.pres.str.contains(n, regex=False)
    return m


def box_area_km2(lat0, lat1, lon0, lon1) -> float:
    """Area of a latitude/longitude rectangle on a sphere of radius 6,371 km."""
    R = 6371.0
    return R * R * abs(math.radians(lon1 - lon0)) * abs(math.sin(math.radians(lat1)) - math.sin(math.radians(lat0)))


def box_area_km2_2024(lat0, lat1, lon0, lon1) -> float:
    """The figure printed on the 2024 maps: the box's diagonal (haversine) times its width at the southern edge.
    Kept only to reconcile those figures; not an area."""
    R = 6371.0
    la0, la1, lo0, lo1 = map(math.radians, (lat0, lat1, lon0, lon1))
    a = math.sin((la1 - la0) / 2) ** 2 + math.cos(la0) * math.cos(la1) * math.sin((lo1 - lo0) / 2) ** 2
    diag = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return diag * R * (lo1 - lo0) * math.cos(la0)


def extent_bounds(d: pd.DataFrame):
    return float(d.LOC_LAT.min()), float(d.LOC_LAT.max()), float(d.LOC_LONG.min()), float(d.LOC_LONG.max())


def dist_m(d: pd.DataFrame, x: float, y: float) -> pd.Series:
    """Planar distance in metres from (x, y) using the City's projected LOC_X/LOC_Y (MTM zone 8, NAD83)."""
    return np.hypot(d.LOC_X - x, d.LOC_Y - y)


def dist_to_segment_m(d: pd.DataFrame, p0, p1) -> pd.Series:
    """Perpendicular distance (m) to the segment p0-p1 given in LOC_X/LOC_Y metres."""
    P = np.c_[d.LOC_X.values - p0[0], d.LOC_Y.values - p0[1]]
    b = np.array([p1[0] - p0[0], p1[1] - p0[1]])
    t = np.clip((P @ b) / (b @ b), 0, 1)
    return pd.Series(np.linalg.norm(P - np.outer(t, b), axis=1), index=d.index)


def mtm8(lat, lon):
    """Latitude/longitude (degrees, NAD83) to MTM zone 8 easting/northing in metres (central meridian -73.5,
    scale 0.9999, false easting 304,800 m), the system of the City's LOC_X/LOC_Y."""
    a, f = 6378137.0, 1 / 298.257222101
    e2 = f * (2 - f); ep2 = e2 / (1 - e2); k0 = 0.9999; lon0 = math.radians(-73.5)
    phi, lam = math.radians(lat), math.radians(lon)
    N = a / math.sqrt(1 - e2 * math.sin(phi) ** 2); T = math.tan(phi) ** 2; C = ep2 * math.cos(phi) ** 2
    A = (lam - lon0) * math.cos(phi)
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * phi)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * phi) - (35 * e2 ** 3 / 3072) * math.sin(6 * phi))
    x = 304800 + k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120)
    y = k0 * (M + N * math.tan(phi) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                                       + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720))
    return x, y


LABELS = {'CHRISTOPHE COLOMB': 'Christophe-Colomb', 'ST ZOTIQUE': 'Saint-Zotique', 'ST ZOTIQUE E': 'Saint-Zotique E.', 'ST PIERRE': 'Saint-Pierre', 'BERGE DU CANAL': 'berge du canal',
          'MAURICE DUPLESSIS': 'Maurice-Duplessis', 'MARC AURELE FORTIN': 'Marc-Aurèle-Fortin', 'DE LA MONTAGNE': 'de la Montagne', 'N-D': 'Notre-Dame', 'COTE DES NEIGES': 'Côte-des-Neiges',
          'AV PARC': 'av. du Parc', 'RENE LEVESQUE': 'René-Lévesque', 'ST DENIS': 'Saint-Denis', 'DES CARRIERES': 'des Carrières', 'ST DENIS FACE': 'Saint-Denis', 'SHERBROOKE O': 'Sherbrooke O.',
          'RIV DES PRAIRIES': 'Rivière-des-Prairies', 'DE LA MARTINIERE': 'de la Martinière', 'BD ROSEMONT': 'boul. Rosemont', 'BELANGER': 'Bélanger', '6E AV': '6e Av.', 'AV DU PARC': 'av. du Parc',
          'AV DES PINS': 'av. des Pins', 'VOIE CAMILIEN HOUDE': 'voie Camillien-Houde', 'CREMAZIE': 'Crémazie', '23E AV': '23e Av.', '19E AV': '19e Av.', 'FORDEN CRESCENT': 'Forden Cres. (Westmount)',
          'FORDEN': 'Forden', 'AV PAPINEAU': 'av. Papineau', 'CH DE L ANSE A L ORME': "ch. de l'Anse-à-l'Orme", 'BD GOUIN O': 'boul. Gouin O.', 'PARC DU AVEN': 'av. du Parc', 'MONT-ROYAL DU O AVEN': 'av. du Mont-Royal O.',
          'NOTRE-DAME RUE': 'Notre-Dame', 'ST-PIERRE AVEN': 'Saint-Pierre', 'ST-LAURENT BOUL': 'boul. Saint-Laurent', 'LIEGE DE O RUE': 'de Liège O.'}


def street_label(s) -> str:
    """A readable form of a street field for labels (known spellings mapped, the rest title-cased)."""
    if not isinstance(s, str):
        return ''
    s = s.strip()
    return LABELS.get(s, s.title())
