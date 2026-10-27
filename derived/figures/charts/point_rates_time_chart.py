"""Dot plot of the time-aware point rates: collisions within 100 m per million crossings in the window of the count years
(count years ± 1 within 2012–2021), with the exposure from the City's one-day counts scaled to a year by the seasonal
profile (Kensington counter, first year) and the weekday/weekend factor (Girouard week of September 2023). Exact
Poisson 95 % interval; where there is no collision in the window, the upper bound is drawn as an open bar from the axis.
The lighter band is the interval's upper end at the lowest and the highest single visit's exposure. Log axis. EN/FR.
Reads ../../tables/terrebonne_point_rates_time_aware.csv (scripts/point_rates_time.py)."""
from pathlib import Path
import textwrap
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = Path(__file__).resolve().parent; TABLES = HERE.parent.parent / 'tables'
SURF = '#fcfcfb'; INK = '#0b0b0b'; INK2 = '#52514e'; GRID = '#e6e5e1'; C = {'bike': '#1baf7a', 'ped': '#2a78d6'}
XMIN = 0.004


def short(name, street):
    parts = [x.strip() for x in name.split('/')]
    other = [x for x in parts if street.lower() not in x.lower()]
    o = other[0] if other else name
    for pre in ('chemin de la ', 'avenue ', 'rue ', 'boulevard '): o = o.replace(pre, '')
    return f"{street} / {o}"


T = {'en': dict(title='Collisions per million crossings in the years of the counts, Terrebonne and Sherbrooke (NDG) intersections',
                sub='Police-reported collisions within 100 m in the window of the count years (± 1 year, within 2012–2021), against the one-day counts scaled to a year by a seasonal profile',
                x='collisions per million crossings (log scale)', bike='bicycle', ped='pedestrian',
                leg_pt='rate in the window, with its exact Poisson 95 % interval', leg_ub='no collision in the window: the interval’s upper bound only',
                leg_band='upper bound if the exposure were the lowest, or the highest, single visit',
                lab='{w}: {k} coll. / {a} crossings a year ({v} visit{s}, {n} count-day{t}{x})',
                winter=', winter only', dec=', December days left out',
                note='Exposure: each count-day is divided by its month’s factor (the Kensington counter’s first year, July 2025–July 2026, bicycles; pedestrians taken as flat) and its day-type factor (the Girouard week of September 2023), then multiplied by 365 and by the window’s years. The two Girouard visits of 2018–19 imply exposures 22 times apart (single digits in late November 2018 vs 101–163 in May 2019); their pooled value is used and the band shows either alone. The 2023 Girouard week (159,800 bicycle and 720,400 pedestrian crossings a year implied) has no collision data to set against it (the file ends in 2021). Winter visits (December–March) are left out of the exposure where another visit exists, since a winter count scaled by a factor near 0.1 is unstable; Draper has only a February count. An exposure-aware comparison with wide intervals, not a measure of risk.',
                src='Source: Ville de Montréal open data (« Collisions routières », « Comptages aux intersections », « Compteurs cyclistes permanents », CC BY 4.0); Kensington counter, borough release of request 2026-581; scripts/point_rates_time.py.'),
     'fr': dict(title='Collisions par million de passages dans les années des comptages, Terrebonne et Sherbrooke (NDG)',
                sub='Collisions déclarées à la police à 100 m ou moins, dans la fenêtre des années de comptage (± 1 an, entre 2012 et 2021), rapportées aux comptages d’une journée étendus à l’année par un profil saisonnier',
                x='collisions par million de passages (échelle log)', bike='bicyclette', ped='piéton',
                leg_pt='taux dans la fenêtre, avec son intervalle de Poisson exact à 95 %', leg_ub='aucune collision dans la fenêtre : borne supérieure de l’intervalle seulement',
                leg_band='borne supérieure si l’exposition était la plus faible, ou la plus forte, visite seule',
                lab='{w} : {k} coll. / {a} passages par an ({v} visite{s}, {n} journée{t}{x})',
                winter=', hiver seulement', dec=', journées de décembre écartées',
                note='Exposition : chaque journée de comptage est divisée par le facteur de son mois (première année du compteur Kensington, juillet 2025–juillet 2026, bicyclettes ; piétons supposés constants) et par son facteur de type de jour (semaine de Girouard, septembre 2023), puis multipliée par 365 et par le nombre d’années de la fenêtre. Les deux visites de Girouard en 2018–19 impliquent des expositions 22 fois différentes (quelques unités fin novembre 2018, 101 à 163 en mai 2019) ; leur valeur combinée est retenue et la bande montre chacune seule. La semaine de septembre 2023 à Girouard (159 800 passages de bicyclettes et 720 400 de piétons par an impliqués) n’a pas de collisions à lui opposer (le fichier s’arrête en 2021). Les visites d’hiver (décembre à mars) sont écartées de l’exposition lorsqu’une autre visite existe, un comptage d’hiver divisé par un facteur proche de 0,1 étant instable ; Draper n’a qu’un comptage de février. Comparaison tenant compte de l’exposition, aux intervalles larges, non une mesure du risque.',
                src='Source : données ouvertes de la Ville de Montréal (« Collisions routières », « Comptages aux intersections », « Compteurs cyclistes permanents », CC BY 4.0) ; compteur Kensington, communication de l’arrondissement (demande 2026-581) ; scripts/point_rates_time.py.')}
R = pd.read_csv(TABLES / 'terrebonne_point_rates_time_aware.csv')
R['short'] = [short(n, st) for n, st in zip(R.intersection, R.street)]
order = [(st, n) for st in ('Terrebonne', 'Sherbrooke') for n in R[R.street == st].sort_values(['count_days', 'intersection'], ascending=[False, True]).intersection.unique()]
for lang in ('en', 'fr'):
    t = T[lang]; fig, ax = plt.subplots(figsize=(11, 9.6), dpi=150, facecolor=SURF); ax.set_facecolor(SURF)
    y = 0
    for st, name in order:
        for mode in ('bike', 'ped'):
            rr = R[(R.intersection == name) & (R['mode'] == mode)]
            if rr.empty: continue
            r = rr.iloc[0]; col = C[mode]; k = int(r.collisions_in_window)
            if r.ci_high_at_lowest_visit > r.ci_high_at_highest_visit:
                ax.plot([r.ci_high_at_highest_visit, r.ci_high_at_lowest_visit], [y, y], color=col, lw=7, alpha=0.18, solid_capstyle='butt', zorder=1)
            if k > 0:
                ax.plot([r.ci_low, r.ci_high], [y, y], color=col, lw=1.6, zorder=2)
                ax.plot(r.rate_per_million, y, 'o', color=col, ms=8, markeredgecolor=SURF, markeredgewidth=1.2, zorder=3)
            else:
                ax.plot([XMIN, r.ci_high], [y, y], color=col, lw=1.6, ls=(0, (3, 2)), zorder=2)
                ax.plot(r.ci_high, y, marker='|', color=col, ms=11, markeredgewidth=1.8, zorder=3)
            extra = (t['winter'] if r.winter_only else '') + (t['dec'] if r.winter_visits_excluded else '')
            a_str = f'{r.implied_annual_crossings:,.0f}'.replace(',', '\u202f' if lang == 'fr' else ',')
            lab = t['lab'].format(w=r.window.replace('-', '–'), k=k, a=a_str, v=int(r.visits), s='' if r.visits == 1 else 's', n=int(r.count_days), t='' if r.count_days == 1 else 's', x=extra)
            ax.text(0.0045, y + 0.3, f"{r.short}, {t[mode]}, {lab}", fontsize=8.5, color=INK2, va='bottom')
            y -= 1
        y -= 0.6
    ax.set_xscale('log'); ax.set_xlim(XMIN, 1500); ax.set_yticks([]); ax.set_ylim(y + 0.3, 0.9)
    ax.set_xlabel(t['x'], fontsize=9, color=INK2)
    for sp in ['top', 'right', 'left']: ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color(GRID); ax.tick_params(colors=INK2, labelsize=9, length=0); ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f'{v:g}'.replace('.', ',' if lang == 'fr' else '.')))
    h = [Line2D([], [], marker='o', color=INK2, lw=1.6, markersize=7, label=t['leg_pt']),
         Line2D([], [], marker='|', color=INK2, lw=1.6, ls=(0, (3, 2)), markersize=9, markeredgewidth=1.6, label=t['leg_ub']),
         Line2D([], [], color=INK2, lw=7, alpha=0.18, label=t['leg_band']),
         Line2D([], [], marker='o', color=C['bike'], lw=0, markersize=7, label=t['bike']), Line2D([], [], marker='o', color=C['ped'], lw=0, markersize=7, label=t['ped'])]
    fig.legend(handles=h, loc='upper left', bbox_to_anchor=(0.04, 0.228), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2, columnspacing=2.5, handlelength=2.4)
    fig.suptitle(t['title'], x=0.045, y=0.978, ha='left', fontsize=12, color=INK, fontweight='bold')
    fig.text(0.045, 0.945, textwrap.fill(t['sub'], 190), fontsize=8.5, color=INK2, va='top', linespacing=1.3)
    fig.text(0.045, 0.145, textwrap.fill(t['note'], 180), fontsize=8, color=INK2, va='top', linespacing=1.4); fig.text(0.045, 0.008, textwrap.fill(t['src'], 165), fontsize=7.5, color=INK2, va='bottom')
    fig.subplots_adjust(left=0.045, right=0.97, top=0.905, bottom=0.26)
    fig.savefig(HERE / f'point_rates_time_{lang}.png', dpi=150, facecolor=SURF); fig.savefig(HERE / f'point_rates_time_{lang}.svg', facecolor=SURF); plt.close(fig)
print('written')
