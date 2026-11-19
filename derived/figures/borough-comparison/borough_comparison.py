"""Côte-des-Neiges–Notre-Dame-de-Grâce among the City's 19 boroughs, SPVM collisions 2012–2021: six ranked panels,
three counts (collisions with an injury or a death; with a pedestrian victim; involving a bicycle) and the same three
set against a denominator (per 100 km of street; per 10,000 residents). English and French, PNG and SVG.

  python3 derived/figures/borough-comparison/borough_comparison.py

Every value is read from derived/tables/boroughs_2012-2021.csv (scripts/cdn_ndg_tables.py). Rates stay by mode. Residents and
street length are not exposure: nobody counted the trips made on foot, by bicycle or by car in each borough."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURF = '#fcfcfb'; INK = '#0b0b0b'; INK2 = '#52514e'; GRID = '#e6e5e1'; HI = '#2a78d6'; LO = '#c9c8c1'
B = 'Côte-des-Neiges–Notre-Dame-de-Grâce'
SHORT = {B: 'CDN–NDG', 'Villeray–Saint-Michel–Parc-Extension': 'Villeray–St-Michel–Parc-Ex', 'Rosemont–La Petite-Patrie': 'Rosemont–La Petite-Patrie', 'Mercier–Hochelaga-Maisonneuve': 'Mercier–Hochelaga-Maisonn.',
         'Rivière-des-Prairies–Pointe-aux-Trembles': 'RDP–Pointe-aux-Trembles', 'L’Île-Bizard–Sainte-Geneviève': 'L’Île-Bizard–Ste-Geneviève', 'Le Plateau-Mont-Royal': 'Le Plateau-Mont-Royal'}
PANELS = [('injury', 0), ('ped', 0), ('bike', 0), ('injury_per_100km', 0), ('ped_per_10k_res', 1), ('bike_per_10k_res', 1)]
T = {'en': dict(title='Côte-des-Neiges–Notre-Dame-de-Grâce among Montréal’s 19 boroughs', sub='Police-reported collisions, January 1, 2012 to December 31, 2021 · each panel ranked on its own · counts above, the same counts against a denominator below',
                p={'injury': 'Collisions with an injury or a death', 'ped': 'Collisions with a pedestrian victim', 'bike': 'Collisions involving a bicycle', 'injury_per_100km': '… per 100 km of street',
                   'ped_per_10k_res': '… per 10,000 residents', 'bike_per_10k_res': '… per 10,000 residents'},
                rank='CDN–NDG: {v} · rank {r} of 19', foot='Counts are collisions whose recorded point falls inside the borough (rows placed at a police district’s centroid left out). Street length: City géobase, local, collector and arterial streets. Residents: 2021 Census, dissemination areas.\nResidents and street length are not exposure: a borough that draws many workers, shoppers or through traffic has more collisions per resident. These are not risks per trip.',
                src='Source: Ville de Montréal open data « Collisions routières » (SPVM reports, CC BY 4.0); Statistics Canada, 2021 Census.'),
     'fr': dict(title='Côte-des-Neiges–Notre-Dame-de-Grâce parmi les 19 arrondissements de Montréal', sub='Collisions déclarées à la police, 1er janvier 2012 au 31 décembre 2021 · chaque panneau classé séparément · les comptes en haut, les mêmes comptes rapportés à un dénominateur en bas',
                p={'injury': 'Collisions avec blessés ou décès', 'ped': 'Collisions avec victime piétonne', 'bike': 'Collisions impliquant une bicyclette', 'injury_per_100km': '… par 100 km de rue',
                   'ped_per_10k_res': '… par 10 000 résidents', 'bike_per_10k_res': '… par 10 000 résidents'},
                rank='CDN–NDG : {v} · rang {r} sur 19', foot='Les comptes sont les collisions dont le point enregistré tombe dans l’arrondissement (lignes placées au centroïde d’un poste de quartier exclues). Longueur de rue : géobase de la Ville, rues locales, collectrices et artères. Résidents : Recensement de 2021, aires de diffusion.\nLes résidents et la longueur de rue ne sont pas une exposition : un arrondissement qui attire travailleurs, clients ou transit compte plus de collisions par résident. Ce ne sont pas des risques par déplacement.',
                src='Source : données ouvertes de la Ville de Montréal « Collisions routières » (rapports du SPVM, CC BY 4.0) ; Statistique Canada, Recensement de 2021.')}


def num(v, nd, lang):
    s = f'{v:,.{nd}f}'
    return s.replace(',', ' ').replace('.', ',') if lang == 'fr' else s


def draw(t, lang):
    tx = T[lang]
    fig, axes = plt.subplots(2, 3, figsize=(14, 10.2), dpi=150, facecolor=SURF)
    for ax, (col, nd) in zip(axes.flat, PANELS):
        s = t.sort_values(col); hi = (s.territory == B).values
        ax.set_facecolor(SURF)
        ax.barh(range(len(s)), s[col], color=[HI if h else LO for h in hi], height=0.72)
        ax.set_yticks(range(len(s))); ax.set_yticklabels([SHORT.get(n, n) for n in s.territory], fontsize=8)
        for lab, h in zip(ax.get_yticklabels(), hi): lab.set_color(INK if h else INK2); lab.set_fontweight('bold' if h else 'normal')
        for sp in ['top', 'right', 'left']: ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color(GRID); ax.tick_params(axis='x', colors=INK2, labelsize=8, length=0); ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_ylim(-0.6, len(s) - 0.4)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: num(v, 0, lang)))
        mx = s[col].max()
        for i, (v, h) in enumerate(zip(s[col], hi)):
            if h or i == len(s) - 1 or i == 0: ax.text(v + mx * 0.012, i, num(v, nd, lang), va='center', fontsize=8, color=INK if h else INK2, fontweight='bold' if h else 'normal')
        ax.set_xlim(0, mx * 1.13)
        me = t[t.territory == B].iloc[0]
        ax.set_title(tx['p'][col], loc='left', fontsize=10.5, color=INK, fontweight='bold', pad=20)
        ax.text(0, 1.022, tx['rank'].format(v=num(me[col], nd, lang), r=int(me[f'rank_{col}'])), transform=ax.transAxes, fontsize=9, color=HI, fontweight='bold')
    fig.suptitle(tx['title'], x=0.035, y=0.982, ha='left', fontsize=15, color=INK, fontweight='bold')
    fig.text(0.035, 0.944, tx['sub'], fontsize=9.5, color=INK2)
    fig.text(0.035, 0.036, tx['foot'], fontsize=8.2, color=INK2, linespacing=1.45, va='bottom')
    fig.text(0.035, 0.012, tx['src'], fontsize=8, color=INK2); fig.text(0.965, 0.012, 'github.com/OpenMobilityData/Montreal-collisions', fontsize=8, color=INK2, ha='right')
    fig.subplots_adjust(left=0.165, right=0.975, top=0.885, bottom=0.115, wspace=0.62, hspace=0.3)
    fig.savefig(HERE / f'borough_comparison_{lang}.png', dpi=150, facecolor=SURF); fig.savefig(HERE / f'borough_comparison_{lang}.svg', facecolor=SURF); plt.close(fig)


if __name__ == '__main__':
    t = pd.read_csv(ROOT / 'derived/tables/boroughs_2012-2021.csv'); t = t[t.territory_type == 'borough']
    for lang in ('en', 'fr'): draw(t, lang)
