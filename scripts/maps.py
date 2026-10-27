"""Collision maps from the SPVM open data 2012–2021, English and French, PNG (+ SVG except the island).

  python3 scripts/maps.py [--extent island|ndg|corridor|h4c ...] [--view ...] [--lang en|fr ...]

Views (the default run draws every valid combination):
  severity        every collision of the extent; shape = road users involved (pedestrian, bicycle, motorcycle, motor
                  vehicles only), colour and size = the collision's severity; fatal drawn last, white with a ring
  bike_vehicle    collisions involving a bicycle, coloured by the other vehicle recorded (car or light truck, heavy
  ped_vehicle     truck, bus, other or none); the same for collisions with a pedestrian victim
  bike_fatal      island only: the collisions in which a cyclist (31) or a pedestrian (156) was killed, coloured by the
  ped_fatal       other vehicle; numbered in date order with a key under the map when there are 40 or fewer, and in
                  every case listed in derived/tables/<mode>_fatalities_2012-2021.csv
  bike, ped       corridor only: every collision involving a bicycle / a pedestrian on rue de Terrebonne (within 25 m
                  of the lane axis or with Terrebonne in a street field), each labelled with its date, by severity
  vru_vehicle, vru_fatal, vru   the same three for the two together (a bicycle involved or a pedestrian victim)
  bike_severity, ped_severity, vru_severity   the severity view restricted to that mode, at every extent
  heat, bike_heat, ped_heat, vru_heat   island and NDG box: smoothed density of collisions per km² (Gaussian kernel,
                  σ = 250 m on the island, 100 m in the box), single-hue ramp with a colour bar; for the dense extents
Extents: island (every collision with coordinates), ndg (the NDG box), corridor (25 m from the lane axis, Belmore to
Girouard; the street-name selection printed beside as the comparison), h4c (the 2024 H4C box; on request only).
Collisions recorded at one point are spread on a Fermat spiral (spacing in pixels, the most severe outermost) and a
note on the map says so. Legend with the count per class and the total; title with the extent, its area and the
period; scale bar; dark basemap credited on the map.
"""
from __future__ import annotations
import argparse, math, textwrap
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import spvm, basemap

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'derived' / 'figures' / 'maps'
TABLES = ROOT / 'derived' / 'tables'
PERIOD = {'en': 'January 1, 2012 to December 31, 2021', 'fr': '1er janvier 2012 au 31 décembre 2021'}
# dark-surface slots of the validated palette (dataviz skill, reference palette): blue, aqua, orange; fatal = white + ring
SEV = {'damage': dict(color='#3987e5', size=5, z=1), 'light': dict(color='#199e70', size=9, z=2),
       'severe': dict(color='#d95926', size=22, z=3), 'fatal': dict(color='#f0efec', size=60, z=4)}
SEV_LABEL = {'en': {'damage': 'property damage', 'light': 'light injury', 'severe': 'severe injury', 'fatal': 'fatal'},
             'fr': {'damage': 'dommages matériels', 'light': 'blessure légère', 'severe': 'blessure grave', 'fatal': 'mortelle'}}
MODE = {'ped': 'v', 'bike': 's', 'moto': 'D', 'motor_only': 'o'}
MODE_LABEL = {'en': {'ped': 'pedestrian', 'bike': 'bicycle', 'moto': 'motorcycle', 'motor_only': 'motor vehicles only'},
              'fr': {'ped': 'piéton', 'bike': 'bicyclette', 'moto': 'motocyclette', 'motor_only': 'véhicules motorisés seulement'}}
VEH = {'car': '#3987e5', 'heavy': '#d95926', 'bus': '#199e70', 'other': '#f0efec'}
VEH_LABEL = {'en': {'car': 'car or light truck', 'heavy': 'heavy truck or tractor-trailer', 'bus': 'bus or minibus', 'other': 'other or none recorded'},
             'fr': {'car': 'automobile ou camion léger', 'heavy': 'camion lourd ou tracteur routier', 'bus': 'autobus ou minibus', 'other': 'autre ou non précisé'}}
TEXT = {'en': dict(title_sev='Police-reported collisions by severity', title_heat={'all': 'Density of police-reported collisions', 'bike': 'Density of collisions involving a bicycle', 'ped': 'Density of collisions with a pedestrian victim', 'vru': 'Density of collisions involving a bicycle or a pedestrian victim'}, heat_bar='collisions per km², 2012–2021, smoothed (Gaussian kernel, σ = {} m)', heat_note='Each collision is spread over a Gaussian kernel of standard deviation {} m and the result is expressed as collisions per km²; the colour bar is linear. Collisions recorded at one point (an intersection) add up at that point.', title_msev={'bike': 'Collisions involving a bicycle, by severity', 'ped': 'Collisions with a pedestrian victim, by severity', 'vru': 'Collisions involving a bicycle or a pedestrian victim, by severity'},
                   title_veh={'bike': 'Collisions involving a bicycle, by the other vehicle', 'ped': 'Collisions with a pedestrian victim, by the vehicle', 'vru': 'Collisions involving a bicycle or a pedestrian victim, by the other vehicle'},
                   title_fatal={'bike': 'Collisions in which a cyclist was killed', 'ped': 'Collisions in which a pedestrian was killed', 'vru': 'Collisions in which a cyclist or a pedestrian was killed'},
                   title_corr={'bike': 'Collisions involving a bicycle on rue de Terrebonne', 'ped': 'Collisions with a pedestrian victim on rue de Terrebonne', 'vru': 'Collisions involving a bicycle or a pedestrian victim on rue de Terrebonne'},
                   corr_sel='within 25 m of the lane axis or with Terrebonne in the street fields; label = date',
                   fatal_key='by date; colour = the other vehicle', fatal_csv='the list, by date, is in derived/tables/{}_fatalities_2012-2021.csv',
                   total='total', area='area', scale='1 km', corridor_note='within 25 m of the lane axis',
                   source='Source: Ville de Montréal, open data « Collisions routières » (SPVM reports), CC BY 4.0. Basemap',
                   mode='road users involved (shape)', severity='severity (colour, size)', severity_only='severity (colour)',
                   spread='Collisions recorded at the same point (the police locate a collision at the intersection) are spread in a small spiral around it so that every marker is visible; the position of a marker within a cluster carries no meaning.',
                   cmp='Comparison, by street name (RUE_ACCDN or ACCDN_PRES_DE contains TERREBONNE): {n} collisions, {f} fatal, {s} severe injury, {b} with a bicycle, {p} with a pedestrian victim. Dashed line: the lane axis, Belmore to Girouard.'),
        'fr': dict(title_sev='Collisions déclarées à la police, par gravité', title_heat={'all': 'Densité des collisions déclarées à la police', 'bike': 'Densité des collisions impliquant une bicyclette', 'ped': 'Densité des collisions avec victime piétonne', 'vru': 'Densité des collisions impliquant une bicyclette ou une victime piétonne'}, heat_bar='collisions par km², 2012–2021, lissées (noyau gaussien, σ = {} m)', heat_note='Chaque collision est étalée sur un noyau gaussien d’écart-type {} m et le résultat est exprimé en collisions par km² ; l’échelle de couleur est linéaire. Les collisions enregistrées au même point (une intersection) s’additionnent en ce point.', title_msev={'bike': 'Collisions impliquant une bicyclette, par gravité', 'ped': 'Collisions avec victime piétonne, par gravité', 'vru': 'Collisions impliquant une bicyclette ou une victime piétonne, par gravité'},
                   title_veh={'bike': 'Collisions impliquant une bicyclette, selon l’autre véhicule', 'ped': 'Collisions avec victime piétonne, selon le véhicule', 'vru': 'Collisions impliquant une bicyclette ou une victime piétonne, selon l’autre véhicule'},
                   title_fatal={'bike': 'Collisions où un ou une cycliste a été tué(e)', 'ped': 'Collisions où un piéton ou une piétonne a été tué(e)', 'vru': 'Collisions où un ou une cycliste ou un piéton ou une piétonne a été tué(e)'},
                   title_corr={'bike': 'Collisions impliquant une bicyclette sur la rue de Terrebonne', 'ped': 'Collisions avec victime piétonne sur la rue de Terrebonne', 'vru': 'Collisions impliquant une bicyclette ou une victime piétonne sur la rue de Terrebonne'},
                   corr_sel='à 25 m ou moins de l’axe de la piste ou avec Terrebonne dans les champs de rue ; étiquette = date',
                   fatal_key='par date ; couleur = l’autre véhicule', fatal_csv='la liste, par date, est dans derived/tables/{}_fatalities_2012-2021.csv',
                   total='total', area='superficie', scale='1 km', corridor_note='à 25 m ou moins de l’axe de la piste',
                   source='Source : Ville de Montréal, données ouvertes « Collisions routières » (rapports du SPVM), CC BY 4.0. Fond de carte',
                   mode='usagers impliqués (forme)', severity='gravité (couleur, taille)', severity_only='gravité (couleur)',
                   spread='Les collisions enregistrées au même point (la police situe une collision à l’intersection) sont étalées en une petite spirale autour de ce point pour que chaque marqueur soit visible ; la position d’un marqueur dans un groupe n’a pas de signification.',
                   cmp='Comparaison, par nom de rue (RUE_ACCDN ou ACCDN_PRES_DE contient TERREBONNE) : {n} collisions, {f} mortelle(s), {s} blessure(s) grave(s), {b} avec bicyclette, {p} avec victime piétonne. Tireté : l’axe de la piste, de Belmore à Girouard.')}
NAME = {'island': {'en': 'Montréal agglomeration (SPVM territory)', 'fr': 'Agglomération de Montréal (territoire du SPVM)'},
        'ndg': {'en': 'NDG box', 'fr': 'Rectangle NDG'}, 'h4c': {'en': 'H4C box', 'fr': 'Rectangle H4C'},
        'corridor': {'en': 'Rue de Terrebonne corridor', 'fr': 'Corridor de la rue de Terrebonne'}}
ZOOM = {'island': 12, 'ndg': 15, 'h4c': 16, 'corridor': 16}
AXIS = (spvm.mtm8(45.45974, -73.64080), spvm.mtm8(45.47833, -73.62487))  # Belmore → Girouard, MTM 8
VIEWS = ['severity', 'bike_severity', 'ped_severity', 'vru_severity', 'heat', 'bike_heat', 'ped_heat', 'vru_heat', 'bike_vehicle', 'ped_vehicle', 'vru_vehicle', 'bike_fatal', 'ped_fatal', 'vru_fatal', 'bike', 'ped', 'vru']
MARK = {'bike': 's', 'ped': 'v', 'vru': 'o'}
DEATHS = {'bike': ['NB_DECES_VELO'], 'ped': ['NB_DECES_PIETON'], 'vru': ['NB_DECES_VELO', 'NB_DECES_PIETON']}


def fmt(n, lang):
    s = f'{n:,}'
    return s.replace(',', ' ') if lang == 'fr' else s


def valid(extent, view):
    if view.endswith('_fatal'):
        return extent == 'island'
    if view.endswith('heat'):
        return extent in ('island', 'ndg')
    if view in ('bike', 'ped', 'vru'):
        return extent == 'corridor'
    return True


def place_labels(ax, fig, X, Y, texts, fontsize, dx=5, dy=4):
    """Annotate points, pushing a label down in 11-pt steps while it would overlap one already placed (greedy, in pixels)."""
    fig.canvas.draw()
    placed = []
    k = fig.dpi / 72
    for x, y, t in zip(X, Y, texts):
        bx, by = ax.transData.transform((x, y))
        w = len(t) * fontsize * 0.58 * k; h = fontsize * 1.3 * k
        off = 0
        for _ in range(12):
            x0, y0 = bx + dx * k, by + (dy - off) * k
            if all(x0 + w < qx or qx + qw < x0 or y0 + h < qy or qy + qh < y0 for qx, qy, qw, qh in placed):
                break
            off += 11
        placed.append((x0, y0, w, h))
        ax.annotate(t, (x, y), xytext=(dx, dy - off), textcoords='offset points', color='#f0efec', fontsize=fontsize, zorder=5,
                    bbox=dict(boxstyle='round,pad=0.12', fc='#1a1a19', ec='none', alpha=0.75))


def spread(X, Y, step, rank=None):
    """Deterministic offsets for points that share a recorded location: each group is laid out on a Fermat spiral
    with the given spacing (map units); with a rank array the highest rank takes the outermost position."""
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
                k = np.arange(n); r = step * np.sqrt(k); th = k * golden
                X[idx] = X[idx] + r * np.cos(th); Y[idx] = Y[idx] + r * np.sin(th)
            start = i
    return X, Y


def other_vehicle(s):
    return np.where(s.nb_camionLourd_tractRoutier.fillna(0) > 0, 'heavy', np.where(s.nb_tous_autobus_minibus.fillna(0) > 0, 'bus',
                    np.where(s.nb_automobile_camion_leger.fillna(0) > 0, 'car', 'other')))


def select(d, extent, view):
    """Rows and the lat/long box of the extent (None = bounds of all located collisions)."""
    if extent == 'island':
        s, box = d[d.LOC_LAT.notna()], None
    elif extent in ('ndg', 'h4c'):
        s, box = d[spvm.in_box(d, spvm.EXTENTS[extent]['box'])], spvm.EXTENTS[extent]['box']
    else:
        near = spvm.dist_to_segment_m(d, *AXIS) <= 25
        s, box = d[near], ((45.4574, 45.4800), (-73.6440, -73.6216))
        if view in ('bike', 'ped', 'vru'):
            s = d[(near | spvm.on_street(d, 'TERREBONNE', 'TERREBONE'))]
    mode = view.split('_')[0] if view not in ('severity', 'heat') else None
    if view.endswith('_vehicle') or view.endswith('_severity') or view.endswith('_heat') or view in ('bike', 'ped', 'vru'):
        s = s[s[mode]]
    if view.endswith('_fatal'):
        s = s[sum(s[c].fillna(0) for c in DEATHS[mode]) > 0]
    if view != 'severity':
        s = s.sort_values('date')
    return s, box, mode


def fatal_key_rows(s):
    return [(r.date.date().isoformat(), spvm.street_label(r.RUE_ACCDN), spvm.street_label(r.ACCDN_PRES_DE) if isinstance(r.ACCDN_PRES_DE, str) and r.ACCDN_PRES_DE not in ('FACE', 'OPP') else '',
             r.NO_CIVIQ_ACCDN if isinstance(r.NO_CIVIQ_ACCDN, str) else '') for r in s.itertuples()]


def draw(d, extent, view, lang):
    s, box, mode = select(d, extent, view)
    T = TEXT[lang]
    located = d[d.LOC_LAT.notna()]
    if box is None:
        lat0, lat1, lon0, lon1 = spvm.extent_bounds(located); pad = 0.01
        lat0 -= pad; lat1 += pad; lon0 -= pad * 1.4; lon1 += pad * 1.4
    else:
        (lat0, lat1), (lon0, lon1) = box
    area = spvm.box_area_km2(lat0, lat1, lon0, lon1) if extent != 'island' else spvm.box_area_km2(*spvm.extent_bounds(located))
    img, ext = basemap.basemap(lon0, lon1, lat0, lat1, ZOOM[extent])
    x0, y0 = basemap.merc(lon0, lat0); x1, y1 = basemap.merc(lon1, lat1)
    aspect = (y1 - y0) / (x1 - x0)
    numbered = view.endswith('_fatal') and len(s) <= 40
    keyh = 2.3 if numbered else 0.0
    W = 11.0; axw = 0.66 * W; axh = axw * aspect; H = axh + 1.9 + keyh
    fig = plt.figure(figsize=(W, H), dpi=150, facecolor='#1a1a19')
    ax = fig.add_axes([0.03, (1.05 + keyh) / H, 0.66, axh / H]); ax.set_facecolor('#1a1a19')
    ax.imshow(img, extent=ext, interpolation='bilinear', zorder=0)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor('#52514e')
    top = (1.05 + keyh + axh) / H
    X, Y = basemap.merc(s.LOC_LONG.values, s.LOC_LAT.values)
    px = (x1 - x0) / (axw * 150)
    step = 0 if extent == 'island' else 5.5 * px
    rank = s.severity.map({'damage': 0, 'light': 1, 'severe': 2, 'fatal': 3}).fillna(0).values
    SX, SY = spread(X, Y, step, rank)
    scale_pts = 2.2 if extent in ('corridor', 'h4c') else 1.0
    if extent == 'corridor':
        ax_, ay_ = basemap.merc(np.array([-73.64080, -73.62487]), np.array([45.45974, 45.47833]))
        ax.plot(ax_, ay_, color='#c3c2b7', lw=1.0, ls=(0, (4, 3)), zorder=1, alpha=0.8)
    handles = []; counts = {}; notes = []
    if view.endswith('heat'):
        sigma_m = 250 if extent == 'island' else 100; res_m = 50 if extent == 'island' else 20
        k_m = 1 / math.cos(math.radians((lat0 + lat1) / 2))  # Web Mercator metres per ground metre
        nx = int((x1 - x0) / (res_m * k_m)) + 1; ny = int((y1 - y0) / (res_m * k_m)) + 1
        Hc, xe, ye = np.histogram2d(X, Y, bins=[nx, ny], range=[[x0, x1], [y0, y1]])
        dens = gaussian_filter(Hc.T, sigma=sigma_m / res_m) / (res_m / 1000) ** 2  # collisions per km²
        cmap = LinearSegmentedColormap.from_list('heat', [(0, (0.1, 0.1, 0.1, 0.0)), (0.15, (0.22, 0.53, 0.90, 0.55)), (0.6, (0.22, 0.53, 0.90, 0.95)), (1.0, (0.94, 0.94, 0.93, 1.0))])
        vmax = float(np.percentile(dens[dens > 0], 99.5)) if (dens > 0).any() else 1.0
        im = ax.imshow(dens, extent=[x0, x1, y0, y1], origin='lower', cmap=cmap, vmin=0, vmax=vmax, interpolation='bilinear', zorder=2)
        cax = fig.add_axes([0.72, top - 0.95 / H, 0.22, 0.14 / H])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal'); cb.outline.set_edgecolor('#52514e'); cb.ax.tick_params(colors='#c3c2b7', labelsize=8, length=2)
        cb.ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: fmt(int(round(v)), lang)))
        fig.text(0.72, top - 0.72 / H, textwrap.fill(T['heat_bar'].format(sigma_m), 44), color='#f0efec', fontsize=9, va='bottom', linespacing=1.3)
        notes.append(T['heat_note'].format(sigma_m))
        title = T['title_heat'][mode or 'all']
    elif view == 'severity' or view.endswith('_severity'):
        for sev in ['damage', 'light', 'severe', 'fatal']:
            st = SEV[sev]
            for m_, marker in MODE.items():
                m = (s.severity == sev).values & s[m_].values; n = int(m.sum()); counts[(sev, m_)] = n
                if n == 0: continue
                edge = sev in ('fatal', 'severe')
                ax.scatter(SX[m], SY[m], s=st['size'] * scale_pts, c=st['color'], marker=marker, linewidths=0.8 if edge else 0,
                           edgecolors='#1a1a19' if edge else 'none', alpha=0.9 if sev != 'damage' else 0.6, zorder=st['z'], rasterized=True)
        handles.append(Line2D([], [], color='none', label=T['severity']))
        for sev in ['fatal', 'severe', 'light', 'damage']:
            n = sum(v for (a, b), v in counts.items() if a == sev)
            handles.append(Line2D([], [], marker='o', color='none', markerfacecolor=SEV[sev]['color'], markeredgecolor='#1a1a19' if sev == 'fatal' else 'none',
                                  markersize=max(4, math.sqrt(SEV[sev]['size']) * 1.1), label=f"{SEV_LABEL[lang][sev]}: {fmt(n, lang)}"))
        handles.append(Line2D([], [], color='none', label=' '))
        handles.append(Line2D([], [], color='none', label=T['mode']))
        for m_, marker in MODE.items():
            n = sum(v for (a, b), v in counts.items() if b == m_)
            if n == 0 and mode: continue
            handles.append(Line2D([], [], marker=marker, color='none', markerfacecolor='#c3c2b7', markersize=7, label=f"{MODE_LABEL[lang][m_]}: {fmt(n, lang)}"))
        title = T['title_msev'][mode] if mode else T['title_sev']
    elif view.endswith('_vehicle') or view.endswith('_fatal'):
        cls = other_vehicle(s); fatal = view.endswith('_fatal')
        for k in ['car', 'other', 'bus', 'heavy']:
            m = cls == k; n = int(m.sum()); counts[k] = n
            if n == 0: continue
            if fatal:
                ax.scatter(X[m], Y[m], s=70 if numbered else 26, c=VEH[k], marker='o', linewidths=0.8, edgecolors='#1a1a19', zorder=3 if k == 'car' else 4)
            else:
                ax.scatter(SX[m], SY[m], s=(7 if k == 'car' else 18) * scale_pts, c=VEH[k], marker='o', linewidths=0, alpha=0.8, zorder=2 if k == 'car' else 3, rasterized=True)
        for k in ['car', 'heavy', 'bus', 'other']:
            handles.append(Line2D([], [], marker='o', color='none', markerfacecolor=VEH[k], markeredgecolor='#1a1a19' if fatal else 'none', markersize=8 if fatal else 7, label=f"{VEH_LABEL[lang][k]}: {fmt(counts[k], lang)}"))
        if fatal:
            rows = fatal_key_rows(s)
            TABLES.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([dict(n=i, date=a, street=b, near=c, civic=e, other_vehicle=v) for i, ((a, b, c, e), v) in enumerate(zip(rows, cls), 1)]).to_csv(TABLES / f'{mode}_fatalities_2012-2021.csv', index=False)
            if numbered:
                place_labels(ax, fig, X, Y, [str(i) for i in range(1, len(X) + 1)], 7.5)
                key = [f"{i}. {a}  {b}" + (f" / {c}" if c else '') for i, (a, b, c, e) in enumerate(rows, 1)]
                fig.text(0.03, (1.0 + keyh) / H, T['fatal_key'], color='#f0efec', fontsize=8.5, va='top')
                per = math.ceil(len(key) / 3)
                for c in range(3):
                    fig.text(0.03 + c * 0.323, (0.78 + keyh) / H, '\n'.join(key[c * per:(c + 1) * per]), color='#c3c2b7', fontsize=7.6, va='top', linespacing=1.35)
            else:
                notes.append(T['fatal_csv'].format(mode))
            title = T['title_fatal'][mode]
        else:
            title = T['title_veh'][mode]
    else:  # bike / ped on the corridor, dated
        for sev in ['damage', 'light', 'severe', 'fatal']:
            st = SEV[sev]; m = (s.severity == sev).values; n = int(m.sum()); counts[sev] = n
            if n: ax.scatter(SX[m], SY[m], s=max(st['size'], 14) * scale_pts, c=st['color'], marker=MARK[mode], linewidths=0.8, edgecolors='#1a1a19', zorder=st['z'])
        place_labels(ax, fig, SX, SY, [dt.date().isoformat() for dt in s.date], 7.2, dx=6, dy=-3)
        handles.append(Line2D([], [], color='none', label=T['severity_only']))
        for sev in ['fatal', 'severe', 'light', 'damage']:
            handles.append(Line2D([], [], marker=MARK[mode], color='none', markerfacecolor=SEV[sev]['color'], markeredgecolor='#1a1a19', markersize=7, label=f"{SEV_LABEL[lang][sev]}: {fmt(counts[sev], lang)}"))
        notes.append(T['corr_sel'])
        title = T['title_corr'][mode]
    total = len(s)
    if handles: fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.705, top - 0.3 / H), frameon=False, labelcolor='#f0efec', fontsize=10, handletextpad=0.6, borderaxespad=0)
    fig.text(0.705, top - 0.15 / H, f"{T['total']}: {fmt(total, lang)}", color='#f0efec', fontsize=12, fontweight='bold', va='bottom')
    if extent != 'island' and view not in ('bike', 'ped', 'vru') and not view.endswith('heat'):
        notes.append(T['spread'])
    if extent == 'corridor' and (view == 'severity' or view.endswith('_severity') or view.endswith('_vehicle')):
        byname = d[spvm.on_street(d, 'TERREBONNE', 'TERREBONE')]
        if mode: byname = byname[byname[mode]]
        notes.append(T['cmp'].format(n=fmt(len(byname), lang), f=int(byname.fatal.sum()), s=int((byname.severity == 'severe').sum()), b=int(byname.bike.sum()), p=int(byname.ped.sum())))
    y = min(top - (0.75 + 0.235 * len(handles)) / H, top - 1.2 / H) if handles else top - 1.25 / H  # below the legend, within the map's height
    for note in notes:
        fig.text(0.705, y, textwrap.fill(note, 44), color='#8f8e87' if note == T['spread'] else '#c3c2b7', fontsize=8.5, va='top', linespacing=1.4)
        y -= (0.19 * (len(textwrap.wrap(note, 44)) + 1.2)) / H
    sub = NAME[extent][lang] + (f" ({T['corridor_note']})" if extent == 'corridor' else '') + f" · {T['area']} {area:,.1f} km² · {PERIOD[lang]}"
    fig.text(0.03, top + 0.42 / H, title, ha='left', va='bottom', color='#f0efec', fontsize=15, fontweight='bold')
    fig.text(0.03, top + 0.12 / H, sub.replace(',', ' ') if lang == 'fr' else sub, color='#c3c2b7', fontsize=10.5, ha='left', va='bottom')
    k = 1 / math.cos(math.radians((lat0 + lat1) / 2)); L = 1000 * k if extent != 'island' else 10000 * k
    bx = x0 + (x1 - x0) * 0.04; by = y0 + (y1 - y0) * 0.05
    ax.plot([bx, bx + L], [by, by], color='#f0efec', lw=2, solid_capstyle='butt', zorder=10)
    ax.text(bx + L / 2, by + (y1 - y0) * 0.015, '10 km' if extent == 'island' else T['scale'], color='#f0efec', ha='center', fontsize=9, zorder=10, bbox=dict(boxstyle='round,pad=0.15', fc='#1a1a19', ec='none', alpha=0.7))
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
    ap = argparse.ArgumentParser(); ap.add_argument('--extent', nargs='*', default=['island', 'ndg', 'corridor'], choices=['island', 'ndg', 'corridor', 'h4c'])
    ap.add_argument('--view', nargs='*', default=VIEWS, choices=VIEWS); ap.add_argument('--lang', nargs='*', default=['en', 'fr'])
    a = ap.parse_args()
    d = spvm.load()
    for e in a.extent:
        for v in a.view:
            if not valid(e, v): continue
            for l in a.lang:
                t, c = draw(d, e, v, l); print(e, v, l, 'total', t, {k: n for k, n in c.items() if n} if 'vehicle' in v or 'fatal' in v else '')
