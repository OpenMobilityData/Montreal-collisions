"""Charts from the two series, English and French, PNG and SVG in derived/figures/charts/.

  saaq_montreal_severity_<lang>   collisions per year in the Montréal region by severity, SAAQ 2011–2022 (small multiples)
  saaq_montreal_type_<lang>       the same by type of collision (small multiples)
  spvm_<extent>_modes_<lang>      collisions per year involving pedestrians, cyclists, motorcyclists, SPVM 2012–2021 (island, NDG box)
  spvm_<extent>_shares_<lang>     the same as a share of all collisions, with a fitted linear trend
Colours: the validated light-surface slots of the reference palette (dataviz skill), fixed per series, never cycled.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spvm, saaq

OUT = Path(__file__).resolve().parent.parent / 'derived' / 'figures' / 'charts'
SURF = '#fcfcfb'; INK = '#0b0b0b'; INK2 = '#52514e'; GRID = '#e6e5e1'
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4', green='#008300', red='#e34948')
MODES = {'ped': ('blue', {'en': 'pedestrian victim', 'fr': 'victime piétonne'}), 'bike': ('aqua', {'en': 'bicycle involved', 'fr': 'bicyclette impliquée'}), 'moto': ('orange', {'en': 'motorcycle involved', 'fr': 'motocyclette impliquée'})}
T = {'en': dict(src_saaq='Source: SAAQ open data « Rapports d’accident » (Données Québec, CC BY 4.0), administrative region Montréal (06).',
                src_spvm='Source: Ville de Montréal open data « Collisions routières » (SPVM reports, CC BY 4.0).',
                sev_title='Police-reported collisions per year in the Montréal region, by severity', type_title='Police-reported collisions per year in the Montréal region, by type of collision',
                modes_title='Collisions per year involving a pedestrian, a bicycle or a motorcycle', shares_title='Share of all collisions involving a pedestrian, a bicycle or a motorcycle',
                period_saaq='2011–2022', period_spvm='2012–2021', of='of', collisions='collisions', trend='dotted: fitted linear trend', per_year='per year', total='total'),
     'fr': dict(src_saaq='Source : données ouvertes de la SAAQ « Rapports d’accident » (Données Québec, CC BY 4.0), région administrative de Montréal (06).',
                src_spvm='Source : données ouvertes de la Ville de Montréal « Collisions routières » (rapports du SPVM, CC BY 4.0).',
                sev_title='Collisions déclarées à la police par année, région de Montréal, selon la gravité', type_title='Collisions déclarées à la police par année, région de Montréal, selon le type',
                modes_title='Collisions par année impliquant un piéton, une bicyclette ou une motocyclette', shares_title='Part de toutes les collisions impliquant un piéton, une bicyclette ou une motocyclette',
                period_saaq='2011–2022', period_spvm='2012–2021', of='sur', collisions='collisions', trend='pointillé : tendance linéaire ajustée', per_year='par année', total='total')}
SEV_SAAQ = [('total', 'blue', {'en': 'all collisions', 'fr': 'toutes les collisions'}), ('damage_under_threshold', 'aqua', {'en': 'damage under the reporting threshold', 'fr': 'dommages sous le seuil de rapportage'}),
            ('damage', 'yellow', {'en': 'property damage only', 'fr': 'dommages matériels seulement'}), ('light', 'orange', {'en': 'light injury', 'fr': 'blessure légère'}), ('fatal_or_severe', 'red', {'en': 'fatal or severe injury', 'fr': 'mortelle ou blessure grave'})]
TYPE_SAAQ = [('genre_motor_vehicle', 'blue', {'en': 'motor vehicle', 'fr': 'véhicule routier'}), ('genre_fixed_object', 'orange', {'en': 'fixed object', 'fr': 'objet fixe'}), ('genre_pedestrian', 'aqua', {'en': 'pedestrian', 'fr': 'piéton'}),
             ('genre_bicycle', 'yellow', {'en': 'cyclist', 'fr': 'cycliste'}), ('genre_no_collision', 'magenta', {'en': 'no collision', 'fr': 'sans collision'}), ('genre_other', 'green', {'en': 'animal or other', 'fr': 'animal ou autre'})]
NAME = {'island': {'en': 'Montréal agglomeration (SPVM territory)', 'fr': 'Agglomération de Montréal (territoire du SPVM)'}, 'ndg': {'en': 'NDG box (50.3 km²)', 'fr': 'Rectangle NDG (50,3 km²)'}}


def fmt(n, lang):
    s = f'{int(round(n)):,}'
    return s.replace(',', ' ') if lang == 'fr' else s


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{stem}.png', dpi=150, facecolor=SURF); fig.savefig(OUT / f'{stem}.svg', facecolor=SURF); plt.close(fig)


def small_multiples(df, series, title, src, lang, stem):
    years = df.index.values
    n = len(series); cols = 3; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.9 * rows + 1.1), dpi=150, facecolor=SURF)
    for ax, (col, hue, label) in zip(axes.flat, series):
        style(ax); y = df[col].values
        ax.plot(years, y, color=C[hue], lw=2, marker='o', ms=4, markerfacecolor=SURF, markeredgewidth=1.5)
        ax.set_ylim(0, max(y) * 1.18); ax.set_xlim(years[0] - 0.4, years[-1] + 0.4); ax.set_xticks(years[::2])
        ax.set_title(f"{label[lang]}", loc='left', fontsize=10.5, color=INK, fontweight='bold', pad=6)
        ax.text(years[0], y[0] + max(y) * 0.06, fmt(y[0], lang), fontsize=8.5, color=INK2, ha='left'); ax.text(years[-1], y[-1] + max(y) * 0.06, fmt(y[-1], lang), fontsize=8.5, color=INK2, ha='right')
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: fmt(v, lang)))
    for ax in list(axes.flat)[n:]: ax.set_visible(False)
    fig.suptitle(title, x=0.045, y=0.985, ha='left', fontsize=13.5, color=INK, fontweight='bold')
    fig.text(0.045, 0.012, src, fontsize=8, color=INK2); fig.text(0.955, 0.012, 'github.com/OpenMobilityData/Montreal-collisions', fontsize=8, color=INK2, ha='right')
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.95]); save(fig, stem)


def modes_chart(y, lang, extent, shares):
    years = y.index.values; txt = T[lang]
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150, facecolor=SURF); style(ax)
    for key, (hue, label) in MODES.items():
        v = (100 * y[key] / y['total']).values if shares else y[key].values
        ax.plot(years, v, color=C[hue], lw=2, marker='o', ms=4.5, markerfacecolor=SURF, markeredgewidth=1.5, label=f"{label[lang]}: {fmt(y[key].sum(), lang)}" if not shares else label[lang])
        if shares:
            p = np.polyfit(years, v, 1); ax.plot(years, np.polyval(p, years), color=C[hue], lw=1.2, ls=(0, (1.5, 2.5)))
            ax.text(years[-1] + 0.15, v[-1], f"{v[-1]:.1f} %".replace('.', ',' if lang == 'fr' else '.'), color=INK2, fontsize=9, va='center')
        else:
            ax.text(years[-1] + 0.15, v[-1], fmt(v[-1], lang), color=INK2, fontsize=9, va='center')
    ax.set_xlim(years[0] - 0.3, years[-1] + 1.1); ax.set_xticks(years); ax.set_ylim(0, None)
    if shares: ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f'{v:g} %'))
    else: ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: fmt(v, lang)))
    leg = ax.legend(loc='upper right' if not shares else 'upper left', frameon=False, fontsize=9.5, labelcolor=INK2, title=f"{txt['total']}: {fmt(y['total'].sum(), lang)} {txt['collisions']}" if not shares else txt['trend'], title_fontsize=9.5, alignment='left')
    leg.get_title().set_color(INK2)
    fig.suptitle(txt['shares_title'] if shares else txt['modes_title'], x=0.06, y=0.975, ha='left', fontsize=13.5, color=INK, fontweight='bold')
    fig.text(0.06, 0.915, f"{NAME[extent][lang]} · {txt['period_spvm']}", fontsize=10, color=INK2)
    fig.text(0.06, 0.02, txt['src_spvm'], fontsize=8, color=INK2); fig.text(0.96, 0.02, 'github.com/OpenMobilityData/Montreal-collisions', fontsize=8, color=INK2, ha='right')
    fig.subplots_adjust(left=0.08, right=0.96, top=0.86, bottom=0.12); save(fig, f"spvm_{extent}_{'shares' if shares else 'modes'}_{lang}")


if __name__ == '__main__':
    m = saaq.yearly(saaq.load(montreal_only=True)); m['genre_other'] = m['genre_animal'] + m['genre_other']
    d = spvm.load()
    tables = {}
    for extent in ['island', 'ndg']:
        s = d if extent == 'island' else d[spvm.in_box(d, spvm.EXTENTS['ndg']['box'])]
        g = s.groupby('year'); y = pd.DataFrame({'total': g.size(), 'ped': g['ped'].sum(), 'bike': g['bike'].sum(), 'moto': g['moto'].sum()}).astype(int)
        y.to_csv(OUT.parent.parent / 'tables' / f'spvm_{extent}_modes_yearly_2012-2021.csv'); tables[extent] = y
    for lang in ['en', 'fr']:
        small_multiples(m, SEV_SAAQ, f"{T[lang]['sev_title']}, {T[lang]['period_saaq']}", T[lang]['src_saaq'], lang, f'saaq_montreal_severity_{lang}')
        small_multiples(m, TYPE_SAAQ, f"{T[lang]['type_title']}, {T[lang]['period_saaq']}", T[lang]['src_saaq'], lang, f'saaq_montreal_type_{lang}')
        for extent, y in tables.items():
            modes_chart(y, lang, extent, False); modes_chart(y, lang, extent, True)
    print('charts written to', OUT); print(tables['island'].to_string())
