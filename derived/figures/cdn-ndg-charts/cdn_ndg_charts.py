"""Charts for Côte-des-Neiges–Notre-Dame-de-Grâce, SPVM collisions 2012–2021, English and French, PNG and SVG.

  python3 derived/figures/cdn-ndg-charts/cdn_ndg_charts.py

  cdn_ndg_districts_<lang>       the five electoral districts (2025 boundaries): four panels, each from zero
  cdn_ndg_yearly_<lang>          the borough by year: four panels, each from zero
  cdn_ndg_intersections_<lang>   the ten intersections with the longest record in the borough under four measures,
                                 chemin Queen-Mary / boulevard Décarie marked wherever it appears
Every value is read from derived/tables/ (scripts/cdn_ndg_tables.py, scripts/intersections.py)."""
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]; TB = ROOT / 'derived/tables'
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURF = '#fcfcfb'; INK = '#0b0b0b'; INK2 = '#52514e'; GRID = '#e6e5e1'; HI = '#2a78d6'; LO = '#c9c8c1'
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', red='#e34948')
M4 = [('collisions', 'blue'), ('injury', 'red'), ('ped', 'orange'), ('bike', 'aqua')]
I4 = ['all', 'injury', 'ped', 'bike']
QMD = 'Décarie / Queen-Mary'
T = {'en': dict(m={'collisions': 'All collisions', 'injury': 'With an injury or a death', 'ped': 'With a pedestrian victim', 'bike': 'Involving a bicycle', 'all': 'All collisions'},
                dt='Côte-des-Neiges–Notre-Dame-de-Grâce by electoral district', ds='Police-reported collisions, 2012–2021 · districts of 2025 · each panel from zero on its own scale',
                df='A district’s count follows its streets: Snowdon holds the Décarie service roads from Queen-Mary to Jean-Talon. Counts, not risks: no traffic, pedestrian or bicycle volume by district.',
                yt='Côte-des-Neiges–Notre-Dame-de-Grâce by year', ys='Police-reported collisions inside the borough · each panel from zero on its own scale',
                yf='2020 and 2021 are the pandemic years. The record ends in 2021. First and last values labelled.',
                it='The ten intersections with the longest record in the borough, by four measures', is_='Collisions recorded at the intersection (within 30 m of its nodes), 2012–2021 · {n} intersections in the borough · Queen-Mary / Décarie in blue',
                if_='An intersection crossing both service roads of the Décarie expressway is one unit. A long record is a count, not a risk: the busiest intersections have the most collisions, and no traffic, pedestrian or bicycle volume enters.',
                rank='{v} · rank {r} of {n}', src='Source: Ville de Montréal open data « Collisions routières » (SPVM reports, CC BY 4.0); intersections from the City géobase.'),
     'fr': dict(m={'collisions': 'Toutes les collisions', 'injury': 'Avec blessés ou décès', 'ped': 'Avec victime piétonne', 'bike': 'Impliquant une bicyclette', 'all': 'Toutes les collisions'},
                dt='Côte-des-Neiges–Notre-Dame-de-Grâce par district électoral', ds='Collisions déclarées à la police, 2012–2021 · districts de 2025 · chaque panneau part de zéro, à sa propre échelle',
                df='Le compte d’un district suit ses rues : Snowdon contient les voies de service de Décarie de Queen-Mary à Jean-Talon. Des comptes, pas des risques : aucun débit de véhicules, de piétons ou de vélos par district.',
                yt='Côte-des-Neiges–Notre-Dame-de-Grâce par année', ys='Collisions déclarées à la police dans l’arrondissement · chaque panneau part de zéro, à sa propre échelle',
                yf='2020 et 2021 sont les années de la pandémie. Le fichier s’arrête en 2021. Première et dernière valeurs indiquées.',
                it='Les dix intersections au dossier le plus long de l’arrondissement, selon quatre mesures', is_='Collisions enregistrées à l’intersection (à 30 m ou moins de ses nœuds), 2012–2021 · {n} intersections dans l’arrondissement · Queen-Mary / Décarie en bleu',
                if_='Une intersection qui traverse les deux voies de service de l’autoroute Décarie compte pour une seule. Un long dossier est un compte, pas un risque : les intersections les plus achalandées comptent le plus de collisions, et aucun débit n’entre dans le calcul.',
                rank='{v} · rang {r} sur {n}', src='Source : données ouvertes de la Ville de Montréal « Collisions routières » (rapports du SPVM, CC BY 4.0) ; intersections tirées de la géobase de la Ville.')}


def fmt(v, lang):
    s = f'{int(round(v)):,}'
    return s.replace(',', ' ') if lang == 'fr' else s


def style(ax, axis='y'):
    ax.set_facecolor(SURF)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0); ax.grid(axis=axis, color=GRID, lw=0.8); ax.set_axisbelow(True)


def finish(fig, tx, title, sub, foot, stem, top=0.86, bottom=0.13, **kw):
    H = fig.get_size_inches()[1]
    fig.text(0.04, 1 - 0.38 / H, title, ha='left', fontsize=14, color=INK, fontweight='bold'); fig.text(0.04, 1 - 0.66 / H, sub, fontsize=9.5, color=INK2)
    fig.text(0.04, 0.36 / H, foot, fontsize=8.3, color=INK2); fig.text(0.04, 0.12 / H, tx['src'], fontsize=8, color=INK2); fig.text(0.96, 0.12 / H, 'github.com/OpenMobilityData/Montreal-collisions', fontsize=8, color=INK2, ha='right')
    fig.subplots_adjust(top=top, bottom=bottom, **kw)
    fig.savefig(HERE / f'{stem}.png', dpi=150, facecolor=SURF); fig.savefig(HERE / f'{stem}.svg', facecolor=SURF); plt.close(fig)


def districts(lang):
    tx = T[lang]; d = pd.read_csv(TB / 'cdn_ndg_districts_2012-2021.csv'); d = d[d.street_km.notna()].sort_values('collisions')
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.6), dpi=150, facecolor=SURF)
    for i, (ax, (col, hue)) in enumerate(zip(axes, M4)):
        style(ax, 'x'); ax.spines['left'].set_visible(False)
        ax.barh(d.district, d[col], color=C[hue], height=0.62)
        for y, v in enumerate(d[col]): ax.text(v + d[col].max() * 0.02, y, fmt(v, lang), va='center', fontsize=8.5, color=INK2)
        ax.set_xlim(0, d[col].max() * 1.18); ax.set_title(tx['m'][col], loc='left', fontsize=10.5, color=INK, fontweight='bold', pad=8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: fmt(v, lang)))
        if i: ax.set_yticklabels([])
        else: ax.tick_params(axis='y', labelsize=9.5, labelcolor=INK)
    finish(fig, tx, tx['dt'], tx['ds'], tx['df'], f'cdn_ndg_districts_{lang}', top=0.80, bottom=0.2, left=0.14, right=0.975, wspace=0.12)


def yearly(lang):
    tx = T[lang]; y = pd.read_csv(TB / 'cdn_ndg_yearly_2012-2021.csv'); y = y[y.scope == 'borough'].set_index('year')
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.4), dpi=150, facecolor=SURF)
    for ax, (col, hue) in zip(axes, M4):
        style(ax); v = y[col].values; yr = y.index.values
        ax.plot(yr, v, color=C[hue], lw=2, marker='o', ms=4.5, markerfacecolor=SURF, markeredgewidth=1.5)
        ax.set_ylim(0, v.max() * 1.2); ax.set_xlim(yr[0] - 0.5, yr[-1] + 0.5); ax.set_xticks(yr[::3])
        ax.text(yr[0], v[0] + v.max() * 0.06, fmt(v[0], lang), fontsize=8.5, color=INK2, ha='left'); ax.text(yr[-1], v[-1] + v.max() * 0.06, fmt(v[-1], lang), fontsize=8.5, color=INK2, ha='right')
        ax.set_title(f"{tx['m'][col]}: {fmt(v.sum(), lang)}", loc='left', fontsize=10.5, color=INK, fontweight='bold', pad=8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: fmt(v, lang)))
    finish(fig, tx, tx['yt'], tx['ys'], tx['yf'], f'cdn_ndg_yearly_{lang}', top=0.79, bottom=0.2, left=0.05, right=0.98, wspace=0.28)


def top_intersections(lang):
    tx = T[lang]; t = pd.read_csv(TB / 'intersections_top_cdn_ndg.csv'); t = t[t['count'] == 'rec']
    r = pd.read_csv(TB / 'intersection_ranks_queen_mary_decarie.csv'); r = r[(r['count'] == 'rec') & (r.comparison_set == 'cdn_ndg')].set_index('measure')
    fig, axes = plt.subplots(1, 4, figsize=(14, 5.4), dpi=150, facecolor=SURF)
    for ax, m in zip(axes, I4):
        s = t[(t.measure == m) & (t['rank'] <= 10)].iloc[::-1]; style(ax, 'x'); ax.spines['left'].set_visible(False)
        hi = (s.name == QMD).values
        ax.barh(range(10), s.value, color=[HI if h else LO for h in hi], height=0.66)
        ax.set_yticks(range(10)); ax.set_yticklabels([n if len(n) < 30 else n[:28] + '…' for n in s.name], fontsize=8)
        for lab, h in zip(ax.get_yticklabels(), hi): lab.set_color(INK if h else INK2); lab.set_fontweight('bold' if h else 'normal')
        for yv, v in enumerate(s.value): ax.text(v + s.value.max() * 0.02, yv, fmt(v, lang), va='center', fontsize=8, color=INK2)
        ax.set_xlim(0, s.value.max() * 1.15); ax.set_title(tx['m'][m], loc='left', fontsize=10.5, color=INK, fontweight='bold', pad=20)
        ax.text(0, 1.025, tx['rank'].format(v=int(r.value[m]), r=int(r['rank'][m]), n=fmt(r.n_in_set[m], lang)), transform=ax.transAxes, fontsize=8.6, color=HI, fontweight='bold')
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: fmt(v, lang)))
    finish(fig, tx, tx['it'], tx['is_'].format(n=fmt(r.n_in_set['all'], lang)), tx['if_'], f'cdn_ndg_intersections_{lang}', top=0.80, bottom=0.17, left=0.125, right=0.985, wspace=0.95)


if __name__ == '__main__':
    for lang in ('en', 'fr'): districts(lang); yearly(lang); top_intersections(lang)
