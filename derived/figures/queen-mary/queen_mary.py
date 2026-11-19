"""Chemin Queen-Mary inside Côte-des-Neiges–Notre-Dame-de-Grâce, SPVM collisions 2012–2021: a map with one marker per
collision (colour and size = severity, shape = road user) beside the record of each intersection, east to west.
English and French, PNG. Two versions: queen_mary_<lang> (map and intersections) and
hero_queen_mary_corridor_<lang> (the same with the headline numbers above and the record by year below).

  python3 derived/figures/queen-mary/queen_mary.py

Selection: collisions within 30 m of the street's géobase segments, inside the borough polygon (derived/tables/
queen_mary_corridor_collisions_2012-2021.csv); intersections and their counts from queen_mary_intersections_2012-2021.csv
(collisions recorded within 30 m of the intersection's nodes)."""
import sys, textwrap
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]; sys.path.insert(0, str(ROOT / 'scripts')); TB = ROOT / 'derived/tables'
import numpy as np, pandas as pd, geopandas as gpd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import areas, areamaps as am

BOX = ((45.4780, 45.4985), (-73.6370, -73.6120))
T = {'en': dict(title='Chemin Queen-Mary: the police record, 2012–2021', sub='Collisions within 30 m of the street, inside Côte-des-Neiges–Notre-Dame-de-Grâce, from Decelles to Macdonald', total='{n} collisions',
                kpi=['collisions within 30 m of the street', 'with an injury', 'with a severe injury', 'fatal', 'with a pedestrian victim', 'involving a bicycle', 'intersections'], year='By year, by severity', sev='severity (colour, size)', mode='road users (shape)', bars='Recorded at each intersection, east to west', dmg='property damage', inj='injury (light)', sf='severe or fatal', pb='ped. · bike', km='500 m',
                note='One marker per collision at its recorded point; collisions recorded at one point are spread in a small spiral. The pedestrian (▽) and bicycle (□) columns count collisions, whatever their severity. Counts, not risks: Queen-Mary’s traffic, pedestrian and bicycle volumes are not in the record.'),
     'fr': dict(title='Chemin Queen-Mary : le dossier policier, 2012–2021', sub='Collisions à 30 m ou moins de la rue, dans Côte-des-Neiges–Notre-Dame-de-Grâce, de Decelles à Macdonald', total='{n} collisions',
                kpi=['collisions à 30 m ou moins de la rue', 'avec blessés', 'avec blessure grave', 'mortelles', 'avec victime piétonne', 'impliquant une bicyclette', 'intersections'], year='Par année, selon la gravité', sev='gravité (couleur, taille)', mode='usagers (forme)', bars='Enregistrées à chaque intersection, d’est en ouest', dmg='dommages matériels', inj='blessure (légère)', sf='grave ou mortelle', pb='piét. · vélo', km='500 m',
                note='Un marqueur par collision, au point enregistré ; les collisions enregistrées au même point sont étalées en petite spirale. Les colonnes piéton (▽) et vélo (□) comptent des collisions, quelle que soit leur gravité. Des comptes, pas des risques : les débits de véhicules, de piétons et de vélos de Queen-Mary ne sont pas dans le fichier.')}


def draw(c, Q, terr, qm, corr, lang, hero=False):
    t = T[lang]; W = 14.0; axw = 0.44 * W; axh = axw * am.Map.aspect(BOX); K = 1.25 if hero else 0.0; H = axh + 1.9 + K
    fig = plt.figure(figsize=(W, H), dpi=150, facecolor=am.BG); top = (1.05 + axh) / H
    if hero:
        k = [len(c), int((c.severity != 'damage').sum()), int((c.severity == 'severe').sum()), int((c.severity == 'fatal').sum()), int(c.ped.sum()), int(c.bike.sum()), int(Q.name.str.contains(' / ').sum())]
        for i, (v, lab) in enumerate(zip(k, t['kpi'])):
            x = 0.03 + i * 0.135
            fig.text(x, top + 0.52 / H, am.fmt(v, lang), color=am.INK, fontsize=27, fontweight='bold'); fig.text(x, top + 0.42 / H, textwrap.fill(lab, 22), color=am.INK2, fontsize=8.8, va='top', linespacing=1.25)
        top_t = top + K / H
    else: top_t = top
    M = am.Map(fig, [0.03, 1.05 / H, 0.44, axh / H], BOX, 16, brighten=1.15)
    M.shade_outside(terr, 0.5); M.outline(terr, color=am.INK2, lw=1.0, z=6); M.lines(qm, color=am.INK, lw=0.8, alpha=0.5, z=4)
    M.severity_markers(c, step_px=4.6, scale=1.25)
    for r in Q.itertuples():
        if r.rec_all >= 60: M.label(r.lon, r.lat, r.name.replace('Queen-Mary / ', '').replace(' / Queen-Mary', ''), dx=14, dy=-14, size=8)
    M.scalebar(500, t['km'])
    sc = c.severity.value_counts().to_dict(); mc = {'ped': int(c.ped.sum()), 'bike': int((c.bike & ~c.ped).sum()), 'moto': int((c.moto & ~c.ped & ~c.bike).sum())}; mc['motor_only'] = len(c) - sum(mc.values())
    h = [Line2D([], [], color='none', label=t['sev'])] + am.sev_handles(lang, {k: sc.get(k, 0) for k in ('fatal', 'severe', 'light', 'damage')}) + [Line2D([], [], color='none', label=' '), Line2D([], [], color='none', label=t['mode'])] + am.mode_handles(lang, mc)
    M.ax.legend(handles=h, loc='lower right', frameon=True, facecolor=am.BG, edgecolor=am.EDGE, framealpha=0.85, labelcolor=am.INK, fontsize=8.5, handletextpad=0.5)
    # bars: one row per intersection, east (top) to west
    yb = 3.75 if hero else 2.0; ax = fig.add_axes([0.66, yb / H, 0.25, (axh + 0.5 - yb) / H]); ax.set_facecolor(am.BG)
    Q = Q[Q.name.str.contains(' / ')].iloc[::-1].reset_index(drop=True); y = np.arange(len(Q))
    dmg = Q.rec_all - Q.rec_injury; light = Q.rec_injury - Q.rec_severe_fatal
    ax.barh(y, dmg, color=am.SEV['damage']['color'], height=0.66, edgecolor=am.BG, lw=1); ax.barh(y, light, left=dmg, color=am.SEV['light']['color'], height=0.66, edgecolor=am.BG, lw=1)
    ax.barh(y, Q.rec_severe_fatal, left=dmg + light, color=am.SEV['severe']['color'], height=0.66, edgecolor=am.BG, lw=1)
    ax.set_yticks(y); ax.set_yticklabels([n.replace('Queen-Mary / ', '').replace(' / Queen-Mary', '') for n in Q.name], color=am.INK, fontsize=8.5)
    for yy, r in zip(y, Q.itertuples()):
        ax.text(r.rec_all + 3, yy, str(r.rec_all), color=am.INK, fontsize=8.5, va='center')
        ax.text(1.16, yy, f'{r.rec_ped} · {r.rec_bike}', color=am.INK2, fontsize=8.5, va='center', ha='center', transform=ax.get_yaxis_transform())
    ax.text(1.16, len(Q) - 0.1, t['pb'], color=am.INK2, fontsize=8.5, ha='center', va='bottom', transform=ax.get_yaxis_transform())
    ax.set_xlim(0, Q.rec_all.max() * 1.12); ax.set_ylim(-0.6, len(Q) - 0.4); ax.tick_params(colors=am.INK2, labelsize=8, length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.grid(axis='x', color=am.EDGE, lw=0.6, alpha=0.6); ax.set_axisbelow(True)
    fig.text(0.50, top - 0.12 / H, t['bars'], color=am.INK, fontsize=11, fontweight='bold', va='top')
    hb = [matplotlib.patches.Patch(color=am.SEV['damage']['color'], label=t['dmg']), matplotlib.patches.Patch(color=am.SEV['light']['color'], label=t['inj']), matplotlib.patches.Patch(color=am.SEV['severe']['color'], label=t['sf'])]
    fig.legend(handles=hb, loc='upper left', bbox_to_anchor=(0.50, top - 0.38 / H), ncol=3, frameon=False, labelcolor=am.INK2, fontsize=8.5, handlelength=1.2, columnspacing=1.2, borderaxespad=0)
    if hero:
        a2 = fig.add_axes([0.53, 2.0 / H, 0.38, 1.05 / H]); a2.set_facecolor(am.BG); yr = c.assign(year=pd.to_datetime(c.date).dt.year).groupby(['year', 'severity']).size().unstack(fill_value=0).reindex(range(2012, 2022), fill_value=0); b = np.zeros(10)
        for sv in ('damage', 'light', 'severe', 'fatal'):
            if sv in yr: a2.bar(yr.index, yr[sv], bottom=b, color=am.SEV[sv]['color'], width=0.7, edgecolor=am.BG, lw=1); b += yr[sv].values
        for x, v in zip(yr.index, b): a2.text(x, v + b.max() * 0.03, str(int(v)), color=am.INK2, fontsize=7.5, ha='center')
        a2.set_xticks(range(2012, 2022)); a2.set_ylim(0, b.max() * 1.2); a2.set_yticks([]); a2.tick_params(colors=am.INK2, labelsize=8, length=0)
        for sp in a2.spines.values(): sp.set_visible(False)
        fig.text(0.50, 3.2 / H, t['year'], color=am.INK, fontsize=11, fontweight='bold', va='bottom')
    fig.text(0.50, 1.7 / H, textwrap.fill(t['note'], 105), color=am.INK2, fontsize=8.3, va='top', linespacing=1.4)
    am.titles(fig, t['title'], t['sub'] + ' · ' + t['total'].format(n=am.fmt(len(c), lang)), top_t); am.footer(fig, lang, 100)
    am.save(fig, HERE, f'hero_queen_mary_corridor_{lang}' if hero else f'queen_mary_{lang}')


if __name__ == '__main__':
    c = pd.read_csv(TB / 'queen_mary_corridor_collisions_2012-2021.csv'); c['motor_only'] = ~(c.ped | c.bike | c.moto)
    Q = pd.read_csv(TB / 'queen_mary_intersections_2012-2021.csv'); corr = pd.read_csv(TB / 'queen_mary_corridor_2012-2021.csv').iloc[0]
    terr = areas.territories(); terr = terr[terr.territory == areas.CDN_NDG]
    st = areas.streets(); qm = gpd.clip(st[st.NOM_VOIE == 'Queen-Mary'], terr.to_crs(areas.MTM8))
    for lang in ('en', 'fr'): draw(c, Q, terr, qm, corr, lang); draw(c, Q, terr, qm, corr, lang, hero=True)
