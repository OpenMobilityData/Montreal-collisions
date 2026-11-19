"""Maps of Côte-des-Neiges–Notre-Dame-de-Grâce from the SPVM collisions 2012–2021, English and French, PNG.

  python3 derived/figures/cdn-ndg-maps/cdn_ndg_maps.py [--lang en fr] [--view ...]

Extent: the borough's polygon (City open data, administrative limits), its five electoral districts of 2025 outlined;
collisions whose recorded point falls inside the polygon, rows placed at a police district's centroid left out.
  heat, ped_heat, bike_heat   smoothed density, collisions per km² (Gaussian kernel, σ = 100 m, 20 m raster), as maps.py
  ped_serious, bike_serious   discrete markers: collisions with a pedestrian victim / involving a bicycle in which
                              someone was killed or severely injured (the collision's severity)
"""
import sys, argparse, textwrap
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[2] / 'scripts'))
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spvm, areas, areamaps as am

BOX = ((45.4490, 45.5185), (-73.6800, -73.5915))
T = {'en': dict(name='Côte-des-Neiges–Notre-Dame-de-Grâce', heat={'heat': 'Density of police-reported collisions', 'ped_heat': 'Density of collisions with a pedestrian victim', 'bike_heat': 'Density of collisions involving a bicycle'},
                serious={'ped_serious': 'Collisions with a pedestrian victim: killed or severely injured', 'bike_serious': 'Collisions involving a bicycle: killed or severely injured'},
                sub='borough polygon, {a} km² · January 1, 2012 to December 31, 2021', total='in the borough', bar='collisions per km², 2012–2021, smoothed (Gaussian kernel, σ = 100 m)',
                note='Each collision is spread over a Gaussian kernel of standard deviation 100 m; the colour bar is linear and its top is the 99.5th percentile of the map. The police locate most collisions at an intersection, so the bright spots are the intersections with the longest records. A long record is a count, not a risk: no traffic, pedestrian or bicycle volume enters. Thin lines: the electoral districts of 2025.',
                note_s='One marker per collision, at its recorded point (collisions recorded at one point are spread in a small spiral). Severity is the collision’s own: the person killed or severely injured is not always the pedestrian or the cyclist. Outside the borough the map is dimmed and no collision is drawn. Thin lines: the electoral districts of 2025.',
                lane='rue de Terrebonne', qm='ch. Queen-Mary', km='1 km', sev='severity of the collision'),
     'fr': dict(name='Côte-des-Neiges–Notre-Dame-de-Grâce', heat={'heat': 'Densité des collisions déclarées à la police', 'ped_heat': 'Densité des collisions avec victime piétonne', 'bike_heat': 'Densité des collisions impliquant une bicyclette'},
                serious={'ped_serious': 'Collisions avec victime piétonne : décès ou blessure grave', 'bike_serious': 'Collisions impliquant une bicyclette : décès ou blessure grave'},
                sub='polygone de l’arrondissement, {a} km² · 1er janvier 2012 au 31 décembre 2021', total='dans l’arrondissement', bar='collisions par km², 2012–2021, lissées (noyau gaussien, σ = 100 m)',
                note='Chaque collision est étalée sur un noyau gaussien d’écart-type 100 m ; l’échelle de couleur est linéaire et son sommet est le 99,5e centile de la carte. La police situe la plupart des collisions à une intersection : les points clairs sont les intersections aux dossiers les plus longs. Un long dossier est un compte, pas un risque : aucun débit de véhicules, de piétons ou de vélos n’entre dans le calcul. Traits fins : les districts électoraux de 2025.',
                note_s='Un marqueur par collision, au point enregistré (les collisions enregistrées au même point sont étalées en petite spirale). La gravité est celle de la collision : la personne tuée ou gravement blessée n’est pas toujours le piéton ou le cycliste. Hors de l’arrondissement, la carte est assombrie et aucune collision n’est tracée. Traits fins : les districts électoraux de 2025.',
                lane='rue de Terrebonne', qm='ch. Queen-Mary', km='1 km', sev='gravité de la collision')}
DLAB = {'Snowdon': (-73.655, 45.4895), 'Notre-Dame-de-Grâce': (-73.6235, 45.4705), 'Loyola': (-73.6455, 45.4585), 'Darlington': (-73.6375, 45.5075), 'Côte-des-Neiges': (-73.6125, 45.5035)}
VIEWS = ['heat', 'ped_heat', 'bike_heat', 'ped_serious', 'bike_serious']


def draw(s, terr, dist, qm, view, lang):
    t = T[lang]; W = 11.0; axw = 0.66 * W; axh = axw * am.Map.aspect(BOX); H = axh + 1.9
    fig = plt.figure(figsize=(W, H), dpi=150, facecolor=am.BG); top = (1.05 + axh) / H
    M = am.Map(fig, [0.03, 1.05 / H, 0.66, axh / H], BOX, 15, brighten=1.45 if 'heat' in view else 1.15)
    mode = view.split('_')[0] if '_' in view else None
    q = s if mode is None else s[s[mode]]
    M.shade_outside(terr, 0.5)
    if 'heat' in view:
        im = M.heat(q.LOC_LONG.values, q.LOC_LAT.values, 100, 20)
        cax = fig.add_axes([0.72, top - 0.95 / H, 0.22, 0.14 / H]); cb = fig.colorbar(im, cax=cax, orientation='horizontal'); cb.outline.set_edgecolor(am.EDGE); cb.ax.tick_params(colors=am.INK2, labelsize=8, length=2)
        cb.ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: am.fmt(round(v), lang)))
        fig.text(0.72, top - 0.72 / H, textwrap.fill(t['bar'], 44), color=am.INK, fontsize=9, va='bottom', linespacing=1.3)
        fig.text(0.705, top - 1.3 / H, textwrap.fill(t['note'], 44), color=am.INK2, fontsize=8.5, va='top', linespacing=1.4); title = t['heat'][view]
    else:
        q = q[q.severity.isin(['fatal', 'severe'])]
        M.severity_markers(q, scale=1.6, by_mode=False)
        c = q.severity.value_counts().to_dict()
        h = [matplotlib.lines.Line2D([], [], color='none', label=t['sev'])] + am.sev_handles(lang, {k: c.get(k, 0) for k in ('fatal', 'severe')})
        fig.legend(handles=h, loc='upper left', bbox_to_anchor=(0.705, top - 0.3 / H), frameon=False, labelcolor=am.INK, fontsize=10, handletextpad=0.6, borderaxespad=0)
        fig.text(0.705, top - 1.35 / H, textwrap.fill(t['note_s'], 44), color=am.INK2, fontsize=8.5, va='top', linespacing=1.4); title = t['serious'][view]
    M.outline(dist, color=am.INK2, lw=0.7, ls=(0, (3, 2)), alpha=0.8, z=5); M.outline(terr, color=am.INK, lw=1.4, z=6)
    for name, (lo, la) in DLAB.items(): M.label(lo, la, name, dx=0, dy=0, size=8, color=am.INK2, ha='center', alpha=0.55)
    ax_, ay_ = M.xy([-73.64080, -73.62487], [45.45974, 45.47833]); M.ax.plot(ax_, ay_, color=am.INK, lw=1.0, ls=(0, (4, 3)), zorder=6, alpha=0.9)
    M.label(-73.64080, 45.45974, t['lane'], dx=-4, dy=-11, size=7.5, ha='right')
    M.lines(qm, color=am.INK, lw=1.0, alpha=0.9, z=6, ls=(0, (1, 2))); M.label(-73.6194, 45.4946, t['qm'], dx=6, dy=2, size=7.5)
    M.scalebar(1000, t['km'])
    fig.text(0.705, top - 0.15 / H, f"{am.fmt(len(q), lang)} {t['total']}", color=am.INK, fontsize=12, fontweight='bold', va='bottom')
    am.titles(fig, title, f"{t['name']} · " + t['sub'].format(a=am.dec(terr.to_crs(areas.MTM8).area.sum() / 1e6, lang)), top); am.footer(fig, lang)
    am.save(fig, HERE, f'cdn_ndg_{view}_{lang}')
    return len(q)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--lang', nargs='*', default=['en', 'fr']); ap.add_argument('--view', nargs='*', default=VIEWS); a = ap.parse_args()
    d = areas.assign(spvm.load()); s = d[(d.territory == areas.CDN_NDG) & (d.LOC_COTE_PD != '4')]
    terr = areas.territories(); terr = terr[terr.territory == areas.CDN_NDG]
    dist = areas.districts(); dist = dist[dist.district_borough.str.contains('Neiges')]
    st = areas.streets(); qm = st[st.NOM_VOIE == 'Queen-Mary']; import geopandas as gpd; qm = gpd.clip(qm, terr.to_crs(areas.MTM8))
    for v in a.view:
        for l in a.lang: print(v, l, draw(s, terr, dist, qm, v, l))
