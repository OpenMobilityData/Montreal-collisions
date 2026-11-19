"""Overview graphic: collision patterns at chemin Queen-Mary / boulevard Décarie, SPVM open data 2012–2021. English and French, PNG.

  python3 derived/figures/queen-mary-decarie-overview/overview.py

The crossing is one intersection: chemin Queen-Mary meets the two service roads of the Décarie expressway, which
passes underneath; the police record places the collisions reported there on a single point and flags them imprecise
(LOC_IMPRECISION), so it cannot say at which service road each one happened. Selection: collisions within 100 m of the
intersection's four géobase nodes (queen_mary_decarie_collisions_2012-2021.csv); « recorded at the intersection » =
within 30 m. Ranks from intersection_ranks_queen_mary_decarie.csv. Every number is read from derived/tables/."""
import sys, textwrap
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]; sys.path.insert(0, str(ROOT / 'scripts')); TB = ROOT / 'derived/tables'
import numpy as np, pandas as pd, geopandas as gpd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import areas, areamaps as am

C = (45.48397, -73.62898); DLAT = 0.00185; DLON = 0.00290
BOX = ((C[0] - DLAT, C[0] + DLAT), (C[1] - DLON, C[1] + DLON))
SEVS = ['damage', 'light', 'severe', 'fatal']
MODES = ['motor vehicles only', 'pedestrian victim', 'bicycle involved', 'motorcycle involved']
T = {'en': dict(title='Queen-Mary / Décarie: what the police record holds, 2012–2021',
                sub='Chemin Queen-Mary where it crosses the two service roads of the Décarie expressway, counted as one intersection · collisions within 100 m',
                kpi=['collisions within 100 m', 'recorded at the crossing itself', 'with an injury', 'severe injury', 'deaths'],
                year='By year', hour='By hour ({n} with no hour stated)', who='Who was involved', rank='Where it ranks among intersections (collisions recorded at the intersection)',
                modes=dict(zip(MODES, ['motor vehicles only', 'pedestrian victim', 'bicycle involved', 'motorcycle involved'])), inj='{n} with an injury',
                rk=['measure', 'here', 'in the borough', 'in the City'], of='of', first='first: {n} ({v})',
                meas={'all': 'all collisions', 'injury': 'with an injury or a death', 'severe_fatal': 'severe injury or death', 'ped': 'pedestrian victim', 'bike': 'bicycle involved'},
                map_note='The police place a collision reported « at Queen-Mary and Décarie » on one point: the {n} markers around it are spread in a spiral and their position inside the cluster carries no meaning. The record cannot say at which service road a collision happened. One of the {m} records names the expressway, which passes below street level.',
                foot='A count is not a risk: the record holds no traffic, pedestrian or bicycle volume, only collisions involving a motor vehicle that police attended, and ends in 2021. Within 100 m the count includes the next intersection, Queen-Mary / Trans Island.',
                sev='severity', mode='road users', m100='100 m', m50='50 m', aut='Décarie expressway (below street level)'),
     'fr': dict(title='Queen-Mary / Décarie : ce que contient le dossier policier, 2012–2021',
                sub='Le chemin Queen-Mary à la traversée des deux voies de service de l’autoroute Décarie, comptée comme une seule intersection · collisions à 100 m ou moins',
                kpi=['collisions à 100 m ou moins', 'enregistrées à la traversée même', 'avec blessés', 'blessure grave', 'décès'],
                year='Par année', hour='Par heure ({n} sans heure)', who='Qui était impliqué', rank='Son rang parmi les intersections (collisions enregistrées à l’intersection)',
                modes=dict(zip(MODES, ['véhicules motorisés seulement', 'victime piétonne', 'bicyclette impliquée', 'motocyclette impliquée'])), inj='{n} avec blessés',
                rk=['mesure', 'ici', 'dans l’arrondissement', 'dans la Ville'], of='sur', first='1re : {n} ({v})',
                meas={'all': 'toutes les collisions', 'injury': 'avec blessés ou décès', 'severe_fatal': 'blessure grave ou décès', 'ped': 'victime piétonne', 'bike': 'bicyclette impliquée'},
                map_note='La police place une collision déclarée « à Queen-Mary et Décarie » en un seul point : les {n} marqueurs qui l’entourent sont étalés en spirale et leur position dans le groupe n’a pas de signification. Le fichier ne dit pas à quelle voie de service une collision a eu lieu. Un seul des {m} dossiers nomme l’autoroute, qui passe en contrebas.',
                foot='Un compte n’est pas un risque : le fichier ne contient aucun débit de véhicules, de piétons ou de vélos, seulement les collisions impliquant un véhicule motorisé où la police est intervenue, et s’arrête en 2021. À 100 m, le compte inclut l’intersection voisine, Queen-Mary / Trans Island.',
                sev='gravité', mode='usagers', m100='100 m', m50='50 m', aut='autoroute Décarie (en contrebas)')}


def dark(ax):
    ax.set_facecolor(am.BG); ax.tick_params(colors=am.INK2, labelsize=8, length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.grid(axis='y', color=am.EDGE, lw=0.6, alpha=0.6); ax.set_axisbelow(True)


def draw(h, R, st, nodes, lang):
    t = T[lang]; W, H = 15.0, 9.6
    fig = plt.figure(figsize=(W, H), dpi=150, facecolor=am.BG)
    fig.text(0.03, 1 - 0.50 / H, t['title'], color=am.INK, fontsize=18, fontweight='bold'); fig.text(0.03, 1 - 0.80 / H, t['sub'], color=am.INK2, fontsize=10.5)
    # headline numbers
    k = [len(h), int(h.at_recorded_point.sum()), int(h.severity.isin(['light', 'severe', 'fatal']).sum()), int((h.severity == 'severe').sum()), int(h.NB_MORTS.sum())]
    for i, (v, lab) in enumerate(zip(k, t['kpi'])):
        x = 0.03 + i * 0.084
        fig.text(x, 1 - 1.52 / H, str(v), color=am.INK, fontsize=27, fontweight='bold'); fig.text(x, 1 - 1.78 / H, textwrap.fill(lab, 15), color=am.INK2, fontsize=8.6, va='top', linespacing=1.25)
    # map: no tiles at this scale (the basemap stops at zoom 16); the City's street centre lines, drawn wide
    mw = 0.40; mh = mw * W * am.Map.aspect(BOX) / H; my = 1.55 / H
    M = am.Map(fig, [0.03, my, mw, mh], BOX, 16, tiles=False)
    M.lines(st[st.CLASSE == 8], color='#0e0e0e', lw=9, alpha=1, z=1)
    loc = st[st.CLASSE != 8]
    M.lines(loc[loc.CLASSE == 0], color='#3b3b39', lw=5, alpha=1, z=2); M.lines(loc[loc.CLASSE != 0], color='#474744', lw=8, alpha=1, z=2)
    from shapely.geometry import box as sbox
    view = gpd.GeoSeries([sbox(M.x0, M.y0, M.x1, M.y1)], crs='EPSG:3857').to_crs(areas.MTM8).iloc[0].buffer(-25)
    seen = set()
    for r in st.assign(L=st.intersection(view).length).sort_values('L', ascending=False).itertuples():
        if r.NOM_VOIE in seen or r.L < 55 or (r.CLASSE == 8 and 'aut' in seen): continue
        seen.add(r.NOM_VOIE); seen.add('aut' if r.CLASSE == 8 else r.NOM_VOIE)
        g = gpd.GeoSeries([r.geometry.intersection(view)], crs=areas.MTM8).to_crs('EPSG:3857').iloc[0]
        if g.geom_type != 'LineString': g = max(g.geoms, key=lambda q: q.length)
        cx, cy = (M.x0 + M.x1) / 2, (M.y0 + M.y1) / 2
        f = max((0.14, 0.86), key=lambda q: np.hypot(g.interpolate(q, normalized=True).x - cx, g.interpolate(q, normalized=True).y - cy))
        p0 = g.interpolate(f - 0.04, normalized=True); p1 = g.interpolate(f + 0.04, normalized=True); pm = g.interpolate(f, normalized=True)
        ang = np.degrees(np.arctan2(p1.y - p0.y, p1.x - p0.x)); ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
        name = (t['aut'] if r.CLASSE == 8 else str(r.ODONYME).strip())
        M.ax.text(pm.x, pm.y, name, color=am.INK3, fontsize=7.5, rotation=ang, rotation_mode='anchor', ha='center', va='center', zorder=4)
    s = h.copy(); s['motor_only'] = ~(s.ped | s.bike | s.moto)
    M.severity_markers(s, step_px=7.2, scale=2.4)
    M.scalebar(50, t['m50'])
    sc = s.severity.value_counts().to_dict(); mc = {'ped': int(s.ped.sum()), 'bike': int((s.bike & ~s.ped).sum()), 'moto': int((s.moto & ~s.ped & ~s.bike).sum())}; mc['motor_only'] = len(s) - sum(mc.values())
    hd = am.sev_handles(lang, {x: sc.get(x, 0) for x in ('fatal', 'severe', 'light', 'damage')}) + am.mode_handles(lang, mc)
    M.ax.legend(handles=hd, loc='upper left', ncol=2, frameon=True, facecolor=am.BG, edgecolor=am.EDGE, framealpha=0.88, labelcolor=am.INK, fontsize=8, handletextpad=0.4, columnspacing=1.0)
    fig.text(0.03, my - 0.12 / H, textwrap.fill(t['map_note'].format(n=k[1], m=k[0]), 108), color=am.INK2, fontsize=8.2, va='top', linespacing=1.4)
    # right column
    X0 = 0.485; top = 1 - 1.25 / H
    def head(y, text): fig.text(X0, y, text, color=am.INK, fontsize=11, fontweight='bold', va='bottom')
    # by year, stacked by severity
    ax = fig.add_axes([X0 + 0.02, top - 1.75 / H, 0.225, 1.5 / H]); dark(ax); head(top - 0.12 / H, t['year'])
    yr = h.groupby(['year', 'severity']).size().unstack(fill_value=0).reindex(range(2012, 2022), fill_value=0); b = np.zeros(10)
    for sv in SEVS:
        if sv in yr: ax.bar(yr.index, yr[sv], bottom=b, color=am.SEV[sv]['color'], width=0.7, edgecolor=am.BG, lw=1); b += yr[sv].values
    for x, v in zip(yr.index, b): ax.text(x, v + 0.8, str(int(v)), color=am.INK2, fontsize=7.5, ha='center')
    ax.set_xticks([2012, 2015, 2018, 2021]); ax.set_ylim(0, b.max() * 1.15)
    # by hour
    ax = fig.add_axes([X0 + 0.285, top - 1.75 / H, 0.205, 1.5 / H]); dark(ax); fig.text(X0 + 0.265, top - 0.12 / H, t['hour'].format(n=int(h.hour.isna().sum())), color=am.INK, fontsize=11, fontweight='bold', va='bottom')
    hr = h.groupby(['hour', 'severity']).size().unstack(fill_value=0).reindex(range(24), fill_value=0); b = np.zeros(24)
    for sv in SEVS:
        if sv in hr: ax.bar(hr.index, hr[sv], bottom=b, color=am.SEV[sv]['color'], width=0.75, edgecolor=am.BG, lw=0.8); b += hr[sv].values
    ax.set_xticks([0, 6, 12, 18, 23]); ax.set_xticklabels(['0 h', '6 h', '12 h', '18 h', '23 h']); ax.set_ylim(0, b.max() * 1.12)
    fig.legend(handles=[Patch(color=am.SEV[x]['color'], label=am.SEV_LABEL[lang][x]) for x in ('damage', 'light', 'severe')], loc='upper left', bbox_to_anchor=(X0 + 0.02, top - 1.92 / H), ncol=3, frameon=False, labelcolor=am.INK2, fontsize=8.5, handlelength=1.2, borderaxespad=0)
    # who was involved
    y0 = top - 2.6 / H; head(y0, t['who'])
    ax = fig.add_axes([X0 + 0.16, y0 - 1.2 / H, 0.24, 1.1 / H]); dark(ax); ax.grid(False)
    g = h.groupby('mode'); n = g.size().reindex(MODES, fill_value=0); inj = h[h.severity != 'damage'].groupby('mode').size().reindex(MODES, fill_value=0)
    yy = np.arange(len(MODES))[::-1]
    ax.barh(yy, n - inj, color=am.SEV['damage']['color'], height=0.6, edgecolor=am.BG, lw=1); ax.barh(yy, inj, left=n - inj, color=am.SEV['light']['color'], height=0.6, edgecolor=am.BG, lw=1)
    ax.set_yticks(yy); ax.set_yticklabels([t['modes'][m] for m in MODES], color=am.INK, fontsize=9); ax.set_xticks([]); ax.set_xlim(0, n.max() * 1.02)
    for y_, a, b_ in zip(yy, n, inj): ax.text(1.03, y_, f"{a}  ·  " + t['inj'].format(n=b_), color=am.INK2, fontsize=9, va='center', transform=ax.get_yaxis_transform())
    # ranks
    y1 = y0 - 1.62 / H; head(y1, t['rank'])
    r = R[R['count'] == 'rec'].set_index(['measure', 'comparison_set'])
    cols = [X0, X0 + 0.165, X0 + 0.215, X0 + 0.355]; yy = y1 - 0.28 / H
    for x, lab in zip(cols, t['rk']): fig.text(x, yy, lab, color=am.INK3, fontsize=8.5, va='center')
    for i, m in enumerate(['all', 'injury', 'severe_fatal', 'ped', 'bike']):
        yy = y1 - (0.62 + 0.40 * i) / H; a = r.loc[(m, 'cdn_ndg')]; c = r.loc[(m, 'city')]
        fig.text(cols[0], yy, t['meas'][m], color=am.INK, fontsize=9.5, va='center'); fig.text(cols[1], yy, str(int(a.value)), color=am.INK, fontsize=11, fontweight='bold', va='center')
        for x, q in ((cols[2], a), (cols[3], c)):
            tie = '=' if q.tied_at_rank > 1 else ''
            fig.text(x, yy + 0.07 / H, f"{tie}{int(q['rank'])} {t['of']} {am.fmt(q.n_in_set, lang)}", color=am.INK, fontsize=10, fontweight='bold', va='center')
            fig.text(x, yy - 0.11 / H, t['first'].format(n=q.first_name if len(q.first_name) < 30 else q.first_name[:28] + '…', v=int(q.first_value)), color=am.INK3, fontsize=7.3, va='center')
    fig.text(X0, 1.38 / H, textwrap.fill(t['foot'], 120), color=am.INK2, fontsize=8.6, va='top', linespacing=1.4)
    fig.text(0.03, 0.38 / H, textwrap.fill(am.SRC[lang].split(' Basemap')[0].split(' Fond de carte')[0], 210), color=am.INK3, fontsize=7.8, va='bottom', linespacing=1.4)
    fig.text(0.03, 0.16 / H, am.REPO, color=am.EDGE, fontsize=8, va='bottom')
    am.save(fig, HERE, f'overview_queen_mary_decarie_{lang}')


if __name__ == '__main__':
    h = pd.read_csv(TB / 'queen_mary_decarie_collisions_2012-2021.csv', parse_dates=['date']); h['year'] = h.date.dt.year
    R = pd.read_csv(TB / 'intersection_ranks_queen_mary_decarie.csv'); nodes = pd.read_csv(TB / 'queen_mary_decarie_nodes.csv')
    st = gpd.read_file(areas.GEOBASE).to_crs(areas.MTM8); st = st[st.CLASSE.isin((0, 5, 6, 7, 8))]; st = st[st.intersects(gpd.GeoSeries(gpd.points_from_xy([nodes.x.mean()], [nodes.y.mean()]), crs=areas.MTM8).buffer(450).iloc[0])]
    for lang in ('en', 'fr'): draw(h, R, st, nodes, lang)
