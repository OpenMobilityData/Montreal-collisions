"""Bar chart of the City's count-days at Terrebonne / Girouard (bicycles, pedestrians, motor vehicles per day), EN/FR."""
from pathlib import Path
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
SURF = '#fcfcfb'; INK = '#0b0b0b'; INK2 = '#52514e'; GRID = '#e6e5e1'; C = {'bikes': '#1baf7a', 'peds': '#2a78d6', 'autos': '#eb6834'}
T = {'en': dict(title='Terrebonne / Girouard: the City’s count-days, 2018–2023', sub='Crossings per day at the intersection, all approaches and movements; 24-hour counts (Comptages aux intersections, Ville de Montréal, CC BY 4.0)',
                bikes='bicycles', peds='pedestrians', autos='motor vehicles', y='per day', note='Police-reported collisions within 100 m, 2012–2021: 44 in all, 3 involving a bicycle, 4 with a pedestrian victim (1 fatal, 2019). The protected lane was built in 2024, after every count shown.',
                src='Source: Ville de Montréal open data « Comptages des véhicules, cyclistes et piétons aux intersections »; scripts/point_rates.py.'),
     'fr': dict(title='Terrebonne / Girouard : les journées de comptage de la Ville, 2018–2023', sub='Passages par jour à l’intersection, toutes approches et tous mouvements ; comptages de 24 heures (Comptages aux intersections, Ville de Montréal, CC BY 4.0)',
                bikes='bicyclettes', peds='piétons', autos='véhicules motorisés', y='par jour', note='Collisions déclarées à la police à 100 m ou moins, 2012–2021 : 44 en tout, 3 impliquant une bicyclette, 4 avec victime piétonne (1 mortelle, 2019). La piste protégée a été aménagée en 2024, après tous les comptages montrés.',
                src='Source : données ouvertes de la Ville de Montréal « Comptages des véhicules, cyclistes et piétons aux intersections » ; scripts/point_rates.py.')}
d = pd.read_csv(HERE / 'terrebonne_intersection_counts_by_day.csv'); d = d[(d.Nom_Intersection == 'Girouard / Terrebonne') & d.full_day].sort_values('Date')
labels = [pd.Timestamp(x).strftime('%Y-%m-%d %a') for x in d.Date]
for lang in ('en', 'fr'):
    t = T[lang]; fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), dpi=150, facecolor=SURF)
    for ax, key in zip(axes, ('bikes', 'peds', 'autos')):
        ax.set_facecolor(SURF); x = np.arange(len(d)); v = d[key].values
        ax.bar(x, v, color=C[key], width=0.72)
        for xi, vi in zip(x, v): ax.text(xi, vi + max(v) * 0.015, f'{int(vi):,}'.replace(',', ' ' if lang == 'fr' else ','), ha='center', fontsize=7, color=INK2, rotation=90 if key != 'bikes' else 0)
        ax.set_title(t[key], loc='left', fontsize=10.5, color=INK, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7, color=INK2); ax.set_ylim(0, max(v) * 1.22)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8, length=0); ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v_, p: f'{int(v_):,}'.replace(',', ' ' if lang == 'fr' else ',')))
        ax.axvspan(2.5, 5.5, color=GRID, alpha=0.5, lw=0); ax.axvspan(5.5, 12.5, color=GRID, alpha=0.25, lw=0)
    fig.suptitle(t['title'], x=0.045, y=0.98, ha='left', fontsize=12.5, color=INK, fontweight='bold')
    fig.text(0.045, 0.9, t['sub'], fontsize=8.5, color=INK2)
    fig.text(0.045, 0.045, t['note'], fontsize=8, color=INK2); fig.text(0.045, 0.012, t['src'], fontsize=7.5, color=INK2)
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.88])
    fig.savefig(HERE / f'girouard_count_days_{lang}.png', dpi=150, facecolor=SURF); fig.savefig(HERE / f'girouard_count_days_{lang}.svg', facecolor=SURF); plt.close(fig)
print('written')
