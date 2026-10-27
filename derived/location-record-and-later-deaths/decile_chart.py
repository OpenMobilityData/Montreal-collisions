"""Deaths per 100 cells in 2017–2021 by decile of the cells' 2012–2016 injury-collision count (100 m cells), with 95 % Poisson intervals."""
from pathlib import Path
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
SURF = '#fcfcfb'; INK = '#0b0b0b'; INK2 = '#52514e'; GRID = '#e6e5e1'; C = {'bike': '#1baf7a', 'ped': '#2a78d6'}
T = {'en': dict(title='Deaths in 2017–2021 per 100 cells, by decile of the cells’ 2012–2016 injury-collision count', sub='100 m grid cells with at least one collision in 2012–2016 (17,593 cells); bars with 95 % Poisson intervals; the deaths at cells with no earlier record (1 cyclist, 2 pedestrians) are outside the ranking',
                bike='cyclist killed', ped='pedestrian killed', of='of', x='decile of the earlier record (1 = fewest injury collisions, 10 = most)', y='deaths per 100 cells', src='Source: Ville de Montréal open data « Collisions routières » (SPVM reports, CC BY 4.0); scripts/location_record_test.py.'),
     'fr': dict(title='Décès de 2017–2021 par 100 cellules, selon le décile du nombre de collisions avec blessés de 2012–2016', sub='Cellules de 100 m avec au moins une collision en 2012–2016 (17 593 cellules) ; barres avec intervalles de Poisson à 95 % ; les décès dans des cellules sans dossier antérieur (1 cycliste, 2 piétons) sont hors classement',
                bike='cycliste tué(e)', ped='piéton(ne) tué(e)', of='sur', x='décile du dossier antérieur (1 = le moins de collisions avec blessés, 10 = le plus)', y='décès par 100 cellules', src='Source : données ouvertes de la Ville de Montréal « Collisions routières » (rapports du SPVM, CC BY 4.0) ; scripts/location_record_test.py.')}
D = pd.read_csv(HERE / 'deaths_by_decile.csv'); D = D[(D.split == '2012-2016 | 2017-2021') & (D.cell_m == 100)]
for lang in ('en', 'fr'):
    t = T[lang]; fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, facecolor=SURF, sharex=True)
    for ax, mode in zip(axes, ('bike', 'ped')):
        s = D[D['mode'] == mode]; ax.set_facecolor(SURF)
        ax.bar(s.decile, s.deaths_per_100_cells, color=C[mode], width=0.72)
        ax.errorbar(s.decile, s.deaths_per_100_cells, yerr=[s.deaths_per_100_cells - s.ci_low, s.ci_high - s.deaths_per_100_cells], fmt='none', ecolor=INK2, elinewidth=1, capsize=2)
        for x, v, k in zip(s.decile, s.deaths_per_100_cells, s.deaths):
            ax.text(x, v + 0.02 * s.ci_high.max(), f'{int(k)}', ha='center', fontsize=8, color=INK2)
        ax.set_title(f"{t[mode]} ({int(s.deaths.sum())} {t['of']} {13 if mode == 'bike' else 87})", loc='left', fontsize=10.5, color=INK, fontweight='bold')
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9, length=0); ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.set_xticks(range(1, 11)); ax.set_xlabel(t['x'], fontsize=9, color=INK2); ax.set_ylabel(t['y'], fontsize=9, color=INK2)
    fig.suptitle(t['title'], x=0.045, y=0.98, ha='left', fontsize=12.5, color=INK, fontweight='bold')
    fig.text(0.045, 0.905, t['sub'], fontsize=9, color=INK2)
    fig.text(0.045, 0.015, t['src'], fontsize=8, color=INK2); fig.text(0.955, 0.015, 'github.com/OpenMobilityData/Montreal-collisions', fontsize=8, color=INK2, ha='right')
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.88])
    fig.savefig(HERE / f'deaths_by_decile_{lang}.png', dpi=150, facecolor=SURF); fig.savefig(HERE / f'deaths_by_decile_{lang}.svg', facecolor=SURF); plt.close(fig)
print('charts written')
