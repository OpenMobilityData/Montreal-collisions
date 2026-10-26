"""Collision maps from the SPVM open data 2012–2021: four extents, two views, English and French.

  python3 scripts/maps.py [--extent island|ndg|h4c|corridor ...] [--view severity|vehicle ...] [--lang en|fr ...]

View « severity »: every collision of the extent, marker shape = mode (pedestrian, bicycle, motorcycle, occupants only),
colour and size = severity of the collision (its GRAVITE); fatal collisions drawn last, white with a ring.
View « vehicle »: collisions involving a bicycle, coloured by the other vehicle recorded (car or light truck, heavy
truck, bus, other). Legend with the count per class and the total; title with the extent's name, its area and the
period; scale bar; dark basemap credited on the map. Output: derived/figures/maps/<extent>_<view>_<lang>.png (+ .svg
except for the island, whose point count makes the SVG impractical).
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import spvm, basemap

OUT = Path(__file__).resolve().parent.parent / 'derived' / 'figures' / 'maps'
PERIOD = {'en': 'January 1, 2012 to December 31, 2021', 'fr': '1er janvier 2012 au 31 décembre 2021'}
# dark-surface slots of the validated palette (dataviz skill, reference palette): blue, aqua, orange; fatal = white + ring
SEV = {'damage': dict(color='#3987e5', size=5, z=1), 'light': dict(color='#199e70', size=9, z=2),
       'severe': dict(color='#d95926', size=22, z=3), 'fatal': dict(color='#f0efec', size=60, z=4)}
SEV_LABEL = {'en': {'damage': 'property damage', 'light': 'light injury', 'severe': 'severe injury', 'fatal': 'fatal'},
             'fr': {'damage': 'dommages matériels', 'light': 'blessure légère', 'severe': 'blessure grave', 'fatal': 'mortelle'}}
MODE = {'ped': 'v', 'bike': 's', 'moto': 'D', 'occupants_only': 'o'}
MODE_LABEL = {'en': {'ped': 'pedestrian', 'bike': 'bicycle', 'moto': 'motorcycle', 'occupants_only': 'occupants only'},
              'fr': {'ped': 'piéton', 'bike': 'bicyclette', 'moto': 'motocyclette', 'occupants_only': 'occupants seulement'}}
VEH = {'car': '#3987e5', 'heavy': '#d95926', 'bus': '#199e70', 'other': '#f0efec'}
VEH_LABEL = {'en': {'car': 'car or light truck', 'heavy': 'heavy truck or tractor-trailer', 'bus': 'bus or minibus', 'other': 'other or none recorded'},
             'fr': {'car': 'automobile ou camion léger', 'heavy': 'camion lourd ou tracteur routier', 'bus': 'autobus ou minibus', 'other': 'autre ou non précisé'}}
TEXT = {'en': dict(title_sev='Police-reported collisions by severity', title_veh='Collisions involving a bicycle, by the other vehicle',
                   total='total', area='area', scale='1 km', source='Source: Ville de Montréal, open data « Collisions routières » (SPVM reports), CC BY 4.0. Basemap', corridor_note='within 25 m of the lane axis',
                   mode='mode (shape)', severity='severity (colour, size)', filter_name='by street name', n_fmt='{:,}'),
        'fr': dict(title_sev='Collisions déclarées à la police, par gravité', title_veh='Collisions impliquant une bicyclette, selon l’autre véhicule',
                   total='total', area='superficie', scale='1 km', source='Source : Ville de Montréal, données ouvertes « Collisions routières » (rapports du SPVM), CC BY 4.0. Fond de carte', corridor_note='à 25 m ou moins de l’axe de la piste',
                   mode='mode (forme)', severity='gravité (couleur, taille)', filter_name='par nom de rue', n_fmt='{:,}')}
NAME = {'island': {'en': 'Montréal agglomeration (SPVM territory)', 'fr': 'Agglomération de Montréal (territoire du SPVM)'},
        'ndg': {'en': 'NDG box', 'fr': 'Rectangle NDG'}, 'h4c': {'en': 'H4C box', 'fr': 'Rectangle H4C'},
        'corridor': {'en': 'Rue de Terrebonne corridor', 'fr': 'Corridor de la rue de Terrebonne'}}
ZOOM = {'island': 12, 'ndg': 15, 'h4c': 16, 'corridor': 16}


def fmt(n, lang):
    s = f'{n:,}'
    return s.replace(',', ' ') if lang == 'fr' else s


def spread(X, Y, step, rank=None):
    """Deterministic offsets for points that share a recorded location (the police fix a collision at the intersection):
    each group is laid out on a Fermat spiral with the given spacing (map units), so that no marker hides another;
    with a rank array, the highest rank (the most severe) takes the outermost position. Returns new X, Y arrays."""
    X = X.copy(); Y = Y.copy()
    if step <= 0:
        return X, Y
    key = np.round(X / 2.0).astype(np.int64) * 1_000_003 + np.round(Y / 2.0).astype(np.int64)
    order = np.argsort(key, kind='stable'); ks = key[order]
    golden = np.pi * (3 - np.sqrt(5))
    start = 0
    for i in range(1, len(ks) + 1):
        if i == len(ks) or ks[i] != ks[start]:
            n = i - start
            if n > 1:
                idx = order[start:i]
                if rank is not None:
                    idx = idx[np.argsort(rank[idx], kind='stable')]
                k = np.arange(n)
                r = step * np.sqrt(k); th = k * golden
                X[idx] = X[idx] + r * np.cos(th); Y[idx] = Y[idx] + r * np.sin(th)
            start = i
    return X, Y


def select(d, extent):
    if extent == 'island':
        return d[d.LOC_LAT.notna()], None
    if extent in ('ndg', 'h4c'):
        return d[spvm.in_box(d, spvm.EXTENTS[extent]['box'])], spvm.EXTENTS[extent]['box']
    A = spvm.mtm8(45.45974, -73.64080); B = spvm.mtm8(45.47833, -73.62487)
    m = spvm.dist_to_segment_m(d, A, B) <= 25
    return d[m], ((45.4574, 45.4800), (-73.6440, -73.6216))


def draw(d, extent, view, lang):
    s, box = select(d, extent)
    if view == 'vehicle':
        s = s[s.bike]
    if box is None:
        lat0, lat1, lon0, lon1 = spvm.extent_bounds(s); pad = 0.01
        lat0 -= pad; lat1 += pad; lon0 -= pad * 1.4; lon1 += pad * 1.4
    else:
        (lat0, lat1), (lon0, lon1) = box
    area = spvm.box_area_km2(lat0, lat1, lon0, lon1) if extent != 'island' else spvm.box_area_km2(*spvm.extent_bounds(s))
    img, ext = basemap.basemap(lon0, lon1, lat0, lat1, ZOOM[extent])
    x0, y0 = basemap.merc(lon0, lat0); x1, y1 = basemap.merc(lon1, lat1)
    aspect = (y1 - y0) / (x1 - x0)
    W = 11.0; axw = 0.66 * W; axh = axw * aspect; H = axh + 1.9
    fig = plt.figure(figsize=(W, H), dpi=150, facecolor='#1a1a19')
    ax = fig.add_axes([0.03, 1.05 / H, 0.66, axh / H]); ax.set_facecolor('#1a1a19')
    ax.imshow(img, extent=ext, interpolation='bilinear', zorder=0)
    X, Y = basemap.merc(s.LOC_LONG.values, s.LOC_LAT.values)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor('#52514e')
    T = TEXT[lang]; handles = []; counts = {}
    scale_pts = 2.2 if extent in ('corridor', 'h4c') else 1.0
    if extent == 'corridor':
        ax_, ay_ = basemap.merc(np.array([-73.64080, -73.62487]), np.array([45.45974, 45.47833]))
        ax.plot(ax_, ay_, color='#c3c2b7', lw=1.0, ls=(0, (4, 3)), zorder=1, alpha=0.8)
    px = (x1 - x0) / (axw * 150)  # map units per pixel at 150 dpi
    step = 0 if extent == 'island' else 6.5 * px  # spacing between markers that share one recorded point
    rank = s.severity.map({'damage': 0, 'light': 1, 'severe': 2, 'fatal': 3}).fillna(0).values
    SX, SY = spread(X, Y, step, rank)
    if view == 'severity':
        for sev in ['damage', 'light', 'severe', 'fatal']:
            st = SEV[sev]
            for mode, marker in MODE.items():
                m = (s.severity == sev).values & s[mode].values if mode != 'occupants_only' else (s.severity == sev).values & s.occupants_only.values
                n = int(m.sum()); counts[(sev, mode)] = n
                if n == 0: continue
                ax.scatter(SX[m], SY[m], s=st['size'] * scale_pts, c=st['color'], marker=marker, linewidths=0.8 if sev in ('fatal', 'severe') else 0,
                           edgecolors='#1a1a19' if sev in ('fatal', 'severe') else 'none', alpha=0.9 if sev != 'damage' else 0.6, zorder=st['z'], rasterized=True)
        # legend: severity (colour) then mode (shape)
        handles.append(Line2D([], [], color='none', label=T['severity']))
        for sev in ['fatal', 'severe', 'light', 'damage']:
            n = sum(v for (a, b), v in counts.items() if a == sev)
            handles.append(Line2D([], [], marker='o', color='none', markerfacecolor=SEV[sev]['color'], markeredgecolor='#1a1a19' if sev == 'fatal' else 'none',
                                  markersize=max(4, math.sqrt(SEV[sev]['size']) * 1.1), label=f"{SEV_LABEL[lang][sev]}: {fmt(n, lang)}"))
        handles.append(Line2D([], [], color='none', label=' '))
        handles.append(Line2D([], [], color='none', label=T['mode']))
        for mode, marker in MODE.items():
            n = sum(v for (a, b), v in counts.items() if b == mode)
            handles.append(Line2D([], [], marker=marker, color='none', markerfacecolor='#c3c2b7', markersize=7, label=f"{MODE_LABEL[lang][mode]}: {fmt(n, lang)}"))
        title = T['title_sev']; total = len(s)
    else:
        cls = np.where(s.nb_camionLourd_tractRoutier.fillna(0) > 0, 'heavy', np.where(s.nb_tous_autobus_minibus.fillna(0) > 0, 'bus', np.where(s.nb_automobile_camion_leger.fillna(0) > 0, 'car', 'other')))
        for k in ['car', 'other', 'bus', 'heavy']:
            m = cls == k; n = int(m.sum()); counts[k] = n
            if n == 0: continue
            ax.scatter(SX[m], SY[m], s=(7 if k == 'car' else 18) * scale_pts, c=VEH[k], marker='o', linewidths=0, alpha=0.8, zorder=2 if k == 'car' else 3, rasterized=True)
        for k in ['car', 'heavy', 'bus', 'other']:
            handles.append(Line2D([], [], marker='o', color='none', markerfacecolor=VEH[k], markersize=7, label=f"{VEH_LABEL[lang][k]}: {fmt(counts[k], lang)}"))
        title = T['title_veh']; total = len(s)
    top = (1.05 + axh) / H
    leg = fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.705, top - 0.3 / H), frameon=False, labelcolor='#f0efec', fontsize=10, handletextpad=0.6, borderaxespad=0)
    fig.text(0.705, top - 0.15 / H, f"{T['total']}: {fmt(total, lang)}", color='#f0efec', fontsize=12, fontweight='bold', va='bottom')
    if extent == 'corridor':
        byname = d[spvm.on_street(d, 'TERREBONNE', 'TERREBONE')]
        if view == 'vehicle': byname = byname[byname.bike]
        cmp = {'en': f"Comparison, by street name (RUE_ACCDN or ACCDN_PRES_DE contains TERREBONNE): {fmt(len(byname), lang)} collisions, {int(byname.fatal.sum())} fatal, {int((byname.severity == 'severe').sum())} severe injury, {int(byname.bike.sum())} with a bicycle, {int(byname.ped.sum())} with a pedestrian victim. Dashed line: the lane axis, Belmore to Girouard.",
               'fr': f"Comparaison, par nom de rue (RUE_ACCDN ou ACCDN_PRES_DE contient TERREBONNE) : {fmt(len(byname), lang)} collisions, {int(byname.fatal.sum())} mortelle(s), {int((byname.severity == 'severe').sum())} blessure(s) grave(s), {int(byname.bike.sum())} avec bicyclette, {int(byname.ped.sum())} avec victime piétonne. Tireté : l’axe de la piste, de Belmore à Girouard."}[lang]
        import textwrap
        fig.text(0.705, top - 4.2 / H, textwrap.fill(cmp, 42), color='#c3c2b7', fontsize=8.5, va='top', linespacing=1.4)
    sub = f"{NAME[extent][lang]}" + (f" ({T['corridor_note']})" if extent == 'corridor' else '') + f" · {T['area']} {area:,.1f} km² · {PERIOD[lang]}"
    fig.text(0.03, top + 0.42 / H, title, ha='left', va='bottom', color='#f0efec', fontsize=15, fontweight='bold')
    fig.text(0.03, top + 0.12 / H, sub.replace(',', '\u202f') if lang == 'fr' else sub, color='#c3c2b7', fontsize=10.5, ha='left', va='bottom')
    # scale bar: 1 km (Web Mercator scale factor at the latitude)
    k = 1 / math.cos(math.radians((lat0 + lat1) / 2)); L = 1000 * k if extent != 'island' else 10000 * k
    bx = x0 + (x1 - x0) * 0.04; by = y0 + (y1 - y0) * 0.05
    ax.plot([bx, bx + L], [by, by], color='#f0efec', lw=2, solid_capstyle='butt', zorder=10)
    ax.text(bx + L / 2, by + (y1 - y0) * 0.015, '10 km' if extent == 'island' else T['scale'], color='#f0efec', ha='center', fontsize=9, zorder=10, bbox=dict(boxstyle='round,pad=0.15', fc='#1a1a19', ec='none', alpha=0.7))
    import textwrap
    fig.text(0.03, 0.5 / H, textwrap.fill(f"{T['source']} {basemap.ATTRIBUTION}.", 150), color='#c3c2b7', fontsize=8.5, va='bottom', linespacing=1.4)
    fig.text(0.03, 0.25 / H, 'github.com/OpenMobilityData/Montreal-collisions', color='#52514e', fontsize=8, va='bottom')
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f'{extent}_{view}_{lang}'
    fig.savefig(stem.with_suffix('.png'), dpi=150, facecolor=fig.get_facecolor())
    if extent != 'island':
        fig.savefig(stem.with_suffix('.svg'), facecolor=fig.get_facecolor())
    plt.close(fig)
    return total, counts


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--extent', nargs='*', default=['island', 'ndg', 'h4c', 'corridor'])
    ap.add_argument('--view', nargs='*', default=['severity', 'vehicle']); ap.add_argument('--lang', nargs='*', default=['en', 'fr'])
    a = ap.parse_args()
    d = spvm.load()
    for e in a.extent:
        for v in a.view:
            for l in a.lang:
                t, c = draw(d, e, v, l); print(e, v, l, 'total', t, {k: v for k, v in c.items() if v} if v == 'vehicle' else '')
