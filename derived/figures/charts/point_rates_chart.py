"""Dot plot of the crude point rates: collisions within 100 m (2012–2021) per million crossings, by intersection and mode,
with the exact Poisson 95 % interval at the mean daily count and the exposure band (rate at the highest and the lowest
daily count observed). Log axis. EN/FR."""
from pathlib import Path
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; TABLES = HERE.parent.parent / 'tables'
SURF = '#fcfcfb'; INK = '#0b0b0b'; INK2 = '#52514e'; GRID = '#e6e5e1'; C = {'bike': '#1baf7a', 'ped': '#2a78d6'}
NAME = {'Girouard / Terrebonne': 'Girouard', 'Draper / Terrebonne': 'Draper', 'chemin de la Côte-Saint-Luc / rue de Terrebonne': 'Côte-Saint-Luc'}
T = {'en': dict(title='Collisions per million crossings at Terrebonne’s counted intersections, 2012–2021', sub='Police-reported collisions within 100 m of the intersection, 2012–2021, against the City’s one-day counts scaled to the ten years',
                x='collisions per million crossings (log scale)', bike='involving a bicycle', ped='with a pedestrian victim',
                leg_pt='rate at the mean daily count, with its exact Poisson 95 % interval', leg_band='rate if the exposure were the lowest, or the highest, daily count observed',
                lab='{k} coll. / {m:,.0f} crossings a day ({n} count-day{s})',
                note='The exposure is the weak link: a one-day count is not ten years of use, and at Girouard the daily bicycle count rose from 1 (November 2018) to 1,128 (September 2023). Draper and Côte-Saint-Luc rest on one count-day each. The figure is the first exposure-aware number for the street, not a measure of risk.',
                src='Source: Ville de Montréal open data (« Collisions routières », « Comptages aux intersections », CC BY 4.0); scripts/point_rates.py.'),
     'fr': dict(title='Collisions par million de passages aux intersections comptées de Terrebonne, 2012–2021', sub='Collisions déclarées à la police à 100 m ou moins de l’intersection, 2012–2021, rapportées aux comptages d’une journée de la Ville étendus aux dix ans',
                x='collisions par million de passages (échelle log)', bike='impliquant une bicyclette', ped='avec victime piétonne',
                leg_pt='taux au comptage journalier moyen, avec son intervalle de Poisson exact à 95 %', leg_band='taux si l’exposition était le plus faible, ou le plus fort, comptage journalier observé',
                lab='{k} coll. / {m:,.0f} passages par jour ({n} journée{s})',
                note='L’exposition est le maillon faible : un comptage d’une journée n’est pas dix ans d’usage, et à Girouard le nombre de bicyclettes par jour est passé de 1 (novembre 2018) à 1 128 (septembre 2023). Draper et Côte-Saint-Luc reposent sur une seule journée. La figure est le premier chiffre tenant compte de l’exposition pour la rue, non une mesure du risque.',
                src='Source : données ouvertes de la Ville de Montréal (« Collisions routières », « Comptages aux intersections », CC BY 4.0) ; scripts/point_rates.py.')}
R = pd.read_csv(TABLES / 'terrebonne_point_rates_2012-2021.csv')
order = ['Girouard / Terrebonne', 'chemin de la Côte-Saint-Luc / rue de Terrebonne', 'Draper / Terrebonne']
for lang in ('en', 'fr'):
    t = T[lang]; fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150, facecolor=SURF); ax.set_facecolor(SURF)
    ys = []; labels = []; y = 0
    for name in order:
        for mode in ('bike', 'ped'):
            r = R[(R.intersection == name) & (R['mode'] == mode)].iloc[0]; col = C[mode]
            lo_b, hi_b = r.rate_if_exposure_at_max_daily, r.rate_if_exposure_at_min_daily
            if pd.notna(lo_b) and pd.notna(hi_b) and hi_b > lo_b:
                ax.plot([lo_b, hi_b], [y, y], color=col, lw=7, alpha=0.18, solid_capstyle='butt', zorder=1)
            ax.plot([r.ci_low if r.ci_low > 0 else 0.005, r.ci_high], [y, y], color=col, lw=1.6, zorder=2)
            ax.plot(r.rate_per_million, y, 'o', color=col, ms=8, markeredgecolor=SURF, markeredgewidth=1.2, zorder=3)
            s = '' if r.count_days == 1 else 's'
            lab = t['lab'].format(k=int(r.collisions_within_100m_2012_2021), m=r.mean_daily_crossings, n=int(r.count_days), s=s)
            if lang == 'fr': lab = lab.replace(',', ' ')
            ax.text(0.0045, y + 0.3, f"{NAME[name]}, {t[mode]}: {lab}", fontsize=8.5, color=INK2, va='bottom')
            ys.append(y); y -= 1
        y -= 0.6
    ax.set_xscale('log'); ax.set_xlim(0.004, 1500); ax.set_yticks([]); ax.set_ylim(y + 0.3, 0.9)
    ax.set_xlabel(t['x'], fontsize=9, color=INK2)
    for sp in ['top', 'right', 'left']: ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color(GRID); ax.tick_params(colors=INK2, labelsize=9, length=0); ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f'{v:g}'.replace('.', ',' if lang == 'fr' else '.')))
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', color=INK2, lw=1.6, markersize=7, label=t['leg_pt']), Line2D([], [], color=INK2, lw=7, alpha=0.18, label=t['leg_band'])]
    leg = ax.legend(handles=h, loc='lower right', frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.suptitle(t['title'], x=0.045, y=0.975, ha='left', fontsize=12.5, color=INK, fontweight='bold')
    fig.text(0.045, 0.905, t['sub'], fontsize=8.5, color=INK2)
    import textwrap
    fig.text(0.045, 0.1, textwrap.fill(t['note'], 175), fontsize=8, color=INK2, va='top', linespacing=1.4); fig.text(0.045, 0.012, t['src'], fontsize=7.5, color=INK2)
    fig.subplots_adjust(left=0.045, right=0.97, top=0.86, bottom=0.26)
    fig.savefig(HERE / f'point_rates_{lang}.png', dpi=150, facecolor=SURF); fig.savefig(HERE / f'point_rates_{lang}.svg', facecolor=SURF); plt.close(fig)
print('written')
