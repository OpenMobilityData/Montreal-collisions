"""Load the SAAQ accident reports open data (Données Québec, « Rapports d'accident », one CSV per year, 2011–2022).

Province-wide, no coordinates; the administrative region (REG_ADM) selects Montréal (06). Kept apart from the
SPVM series: different years, territory and reporting; never merged.
"""
from __future__ import annotations
import glob
from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / 'raw' / 'saaq'
MONTREAL = 'Montréal (06)'
SEVERITY = {'Mortel ou grave': 'fatal_or_severe', 'Léger': 'light',
            'Dommages matériels seulement': 'damage',
            'Dommages matériels inférieurs au seuil de rapportage': 'damage_under_threshold'}
SEVERITY_ORDER = ['fatal_or_severe', 'light', 'damage', 'damage_under_threshold']
GENRE = {'cycliste': 'bicycle', 'piéton': 'pedestrian', 'véhicule': 'motor_vehicle', 'objet fixe': 'fixed_object',
         'animal': 'animal', 'autre': 'other', 'sans collision': 'no_collision'}


def load_year(path) -> pd.DataFrame:
    # The header carries stray tabs between some fields; the python engine with skipinitialspace copes.
    s = pd.read_csv(path, dtype=str, encoding='utf-8-sig', sep=',', skipinitialspace=True, quotechar='"', engine='python')
    s.columns = [c.strip().strip('"') for c in s.columns]
    for c in s.columns:
        s[c] = s[c].str.strip()
    s['year'] = pd.to_numeric(s['AN'], errors='coerce')
    s['severity'] = s['GRAVITE'].map(SEVERITY)
    s['genre'] = s['CD_GENRE_ACCDN'].map(GENRE)
    for flag, col in [('bike', 'IND_VELO'), ('ped', 'IND_PIETON'), ('moto', 'IND_MOTO_CYCLO'), ('heavy', 'IND_VEH_LOURD'), ('car', 'IND_AUTO_CAMION_LEGER')]:
        s[flag] = s[col] == 'O'
    s['montreal'] = s['REG_ADM'] == MONTREAL
    return s


def load(years=range(2011, 2023), montreal_only=False) -> pd.DataFrame:
    frames = []
    for y in years:
        files = sorted(glob.glob(str(RAW / f'Rapport_Accident_{y}.csv*')))
        if not files:
            raise FileNotFoundError(f'no SAAQ file for {y} in {RAW}')
        f = load_year(files[0])
        frames.append(f[f.montreal] if montreal_only else f)
    return pd.concat(frames, ignore_index=True)


def yearly(d: pd.DataFrame) -> pd.DataFrame:
    """Collisions per year by severity, by genre and by mode flag (one row per year)."""
    g = d.groupby('year')
    out = pd.DataFrame({'total': g.size()})
    for sev in SEVERITY_ORDER:
        out[sev] = g.apply(lambda x: int((x.severity == sev).sum()))
    for k in GENRE.values():
        out['genre_' + k] = g.apply(lambda x: int((x.genre == k).sum()))
    for flag in ['bike', 'ped', 'moto', 'heavy', 'car']:
        out['with_' + flag] = g[flag].sum().astype(int)
    return out


if __name__ == '__main__':
    m = load(montreal_only=True)
    t = yearly(m)
    out = Path(__file__).resolve().parent.parent / 'derived' / 'tables'
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / 'saaq_montreal_yearly_2011-2022.csv')
    q = yearly(load())
    q.to_csv(out / 'saaq_quebec_yearly_2011-2022.csv')
    pd.set_option('display.width', 250)
    print(t[['total'] + SEVERITY_ORDER + ['with_bike', 'with_ped', 'with_moto']].to_string())
