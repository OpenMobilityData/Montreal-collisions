"""Density maps by severity: small multiples (one panel per severity class) and a bivariate map (brightness = density,
hue = share of collisions with an injury), island and NDG box, English and French.

  python3 scripts/heatmaps.py [--extent island ndg] [--kind multiples bivariate] [--lang en fr]

Kernel and raster as in maps.py (σ = 250 m / 100 m; 50 m / 20 m cells). The bivariate share is shrunk toward the
extent's overall share with a prior of PRIOR collisions, so that cells with few collisions read as average.
"""
from __future__ import annotations
import argparse, math, textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from scipy.ndimage import gaussian_filter
import spvm, basemap, maps

OUT = maps.OUT
CLASSES = [('damage', ['damage']), ('light', ['light']), ('severe_fatal', ['severe', 'fatal'])]
LABEL = {'en': {'damage': 'property damage only', 'light': 'light injury', 'severe_fatal': 'severe injury or fatal'},
         'fr': {'damage': 'dommages matériels seulement', 'light': 'blessure légère', 'severe_fatal': 'blessure grave ou mortelle'}}
T = {'en': dict(title_m='Density of police-reported collisions, by severity', title_b='Density of police-reported collisions and the share with an injury',
                bar='collisions per km², smoothed (σ = {} m)', bar_b='density (collisions per km², smoothed, σ = {} m)', share='share with an injury',
                note_b='Brightness follows the smoothed density of all collisions; the hue follows the smoothed share of collisions with an injury (fatal, severe or light), shrunk toward the extent’s overall share ({:.0f} %) with a prior of {} collisions so that sparse cells read as average. The hue range runs from the 10th to the 90th percentile of the share over the populated cells; legend: density across, injury share down.',
                note_m='Each panel is the smoothed density of one severity class, with its own colour bar; the kernel is the same in every panel.'),
     'fr': dict(title_m='Densité des collisions déclarées à la police, selon la gravité', title_b='Densité des collisions déclarées à la police et part avec blessés',
                bar='collisions par km², lissées (σ = {} m)', bar_b='densité (collisions par km², lissées, σ = {} m)', share='part avec blessés',
                note_b='La luminosité suit la densité lissée de toutes les collisions ; la teinte suit la part lissée des collisions avec blessés (mortelle, grave ou légère), ramenée vers la part d’ensemble du territoire ({:.0f} %) avec un a priori de {} collisions pour que les cellules peu fournies se lisent comme moyennes. La gamme de teintes va du 10e au 90e centile de la part sur les cellules fournies ; légende : densité en abscisse, part avec blessés en ordonnée.',
                note_m='Chaque panneau est la densité lissée d’une classe de gravité, avec sa propre échelle ; le noyau est le même partout.')}
PRIOR = 20
BLUE = np.array(to_rgb('#3987e5')); ORANGE = np.array(to_rgb('#d95926')); WHITE = np.array([0.97, 0.97, 0.95])


def raster(X, Y, x0, x1, y0, y1, res, sigma):
    nx = int((x1 - x0) / res) + 1; ny = int((y1 - y0) / res) + 1
    Hc, _, _ = np.histogram2d(X, Y, bins=[nx, ny], range=[[x0, x1], [y0, y1]])
    return gaussian_filter(Hc.T, sigma=sigma / res)


def frame(extent, lang, view_label, ncols, width=13.0):
    d = maps.spvm.load() if not hasattr(frame, 'd') else frame.d
    frame.d = d
    s, box, _ = maps.select(d, extent, 'severity')
    located = d[d.LOC_LAT.notna()]
    if box is None:
        lat0, lat1, lon0, lon1 = spvm.extent_bounds(located); pad = 0.01; lat0 -= pad; lat1 += pad; lon0 -= pad * 1.4; lon1 += pad * 1.4
    else:
        (lat0, lat1), (lon0, lon1) = box
    area = spvm.box_area_km2(lat0, lat1, lon0, lon1) if extent != 'island' else spvm.box_area_km2(*spvm.extent_bounds(located))
    img, ext = basemap.basemap(lon0, lon1, lat0, lat1, maps.ZOOM[extent]); img = np.clip(img * 1.45, 0, 1)
    x0, y0 = basemap.merc(lon0, lat0); x1, y1 = basemap.merc(lon1, lat1)
    k_m = 1 / math.cos(math.radians((lat0 + lat1) / 2))
    sigma_m = 250 if extent == 'island' else 100; res_m = 50 if extent == 'island' else 20
    return dict(d=d, s=s, img=img, ext=ext, x0=x0, x1=x1, y0=y0, y1=y1, k=k_m, sigma=sigma_m * k_m, res=res_m * k_m, sigma_m=sigma_m, res_m=res_m, area=area, lat0=lat0, lat1=lat1)


def style_axes(ax, F):
    ax.set_facecolor('#1a1a19'); ax.imshow(F['img'], extent=F['ext'], interpolation='bilinear', zorder=0)
    ax.set_xlim(F['x0'], F['x1']); ax.set_ylim(F['y0'], F['y1']); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor('#52514e')


def terrebonne(ax, extent):
    if extent == 'ndg':
        ax_, ay_ = basemap.merc(np.array([-73.64080, -73.62487]), np.array([45.45974, 45.47833]))
        ax.plot(ax_, ay_, color='#f0efec', lw=1.0, ls=(0, (4, 3)), zorder=6, alpha=0.9)
        ax.annotate('Terrebonne', (ax_[1], ay_[1]), xytext=(4, 2), textcoords='offset points', color='#f0efec', fontsize=7, zorder=7, bbox=dict(boxstyle='round,pad=0.15', fc='#1a1a19', ec='none', alpha=0.75))


def footer(fig, F, extent, lang, title, H):
    t = maps.TEXT[lang]
    sub = maps.NAME[extent][lang] + f" · {t['area']} {F['area']:,.1f} km² · {maps.PERIOD[lang]}"
    fig.text(0.03, 1 - 0.45 / H, title, ha='left', va='bottom', color='#f0efec', fontsize=15, fontweight='bold')
    fig.text(0.03, 1 - 0.72 / H, sub.replace(',', ' ') if lang == 'fr' else sub, color='#c3c2b7', fontsize=10.5, va='bottom')
    fig.text(0.03, 0.5 / H, textwrap.fill(f"{t['source']} {basemap.ATTRIBUTION}.", 170), color='#c3c2b7', fontsize=8.5, va='bottom', linespacing=1.4)
    fig.text(0.03, 0.25 / H, 'github.com/OpenMobilityData/Montreal-collisions', color='#52514e', fontsize=8, va='bottom')


def multiples(extent, lang):
    F = frame(extent, lang, 'multiples', 3); s = F['s']; X, Y = basemap.merc(s.LOC_LONG.values, s.LOC_LAT.values)
    aspect = (F['y1'] - F['y0']) / (F['x1'] - F['x0'])
    W = 15.0; gap = 0.25; axw = (W - 0.6 - 2 * gap) / 3; axh = axw * aspect; H = axh + 2.7
    fig = plt.figure(figsize=(W, H), dpi=150, facecolor='#1a1a19')
    cmap = LinearSegmentedColormap.from_list('heat', [(0, (0.22, 0.53, 0.90, 0.0)), (0.12, (0.22, 0.53, 0.90, 0.25)), (0.45, (0.22, 0.53, 0.90, 0.6)), (0.8, (0.55, 0.75, 0.97, 0.9)), (1.0, (0.97, 0.97, 0.95, 1.0))])
    for i, (key, classes) in enumerate(CLASSES):
        m = s.severity.isin(classes).values
        dens = raster(X[m], Y[m], F['x0'], F['x1'], F['y0'], F['y1'], F['res'], F['sigma']) / (F['res_m'] / 1000) ** 2
        ax = fig.add_axes([(0.3 + i * (axw + gap)) / W, 1.55 / H, axw / W, axh / H]); style_axes(ax, F)
        vmax = float(np.percentile(dens[dens > 0], 99.5)) if (dens > 0).any() else 1.0
        im = ax.imshow(dens, extent=[F['x0'], F['x1'], F['y0'], F['y1']], origin='lower', cmap=cmap, vmin=0, vmax=vmax, interpolation='bilinear', zorder=2)
        terrebonne(ax, extent)
        ax.set_title(f"{LABEL[lang][key]}: {maps.fmt(int(m.sum()), lang)}", loc='left', color='#f0efec', fontsize=11, fontweight='bold', pad=6)
        cax = fig.add_axes([(0.3 + i * (axw + gap)) / W, 1.15 / H, axw * 0.6 / W, 0.12 / H])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal'); cb.outline.set_edgecolor('#52514e'); cb.ax.tick_params(colors='#c3c2b7', labelsize=8, length=2)
        cb.ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: maps.fmt(int(round(v)), lang)))
        fig.text((0.3 + i * (axw + gap)) / W, 0.95 / H, T[lang]['bar'].format(F['sigma_m']), color='#c3c2b7', fontsize=8.5, va='top')
    footer(fig, F, extent, lang, T[lang]['title_m'], H)
    fig.text(0.97, 1 - 0.45 / H, textwrap.fill(T[lang]['note_m'], 80), color='#8f8e87', fontsize=8.5, va='bottom', ha='right', linespacing=1.4)
    stem = OUT / f'{extent}_heat_severity_{lang}'
    fig.savefig(stem.with_suffix('.png'), dpi=150, facecolor=fig.get_facecolor())
    if extent != 'island': fig.savefig(stem.with_suffix('.svg'), facecolor=fig.get_facecolor())
    plt.close(fig)


NEUTRAL = np.array(to_rgb('#c3c2b7'))


def share_colour(h):
    """Diverging colour for a share position h in [0, 1]: blue (0) through the light neutral (0.5) to orange (1)."""
    h = np.asarray(h, float)[..., None]
    low = BLUE * (1 - 2 * h) + NEUTRAL * (2 * h); high = NEUTRAL * (2 - 2 * h) + ORANGE * (2 * h - 1)
    return np.where(h < 0.5, low, high)


def bivariate(extent, lang):
    F = frame(extent, lang, 'bivariate', 1); s = F['s']; X, Y = basemap.merc(s.LOC_LONG.values, s.LOC_LAT.values)
    inj = s.severity.isin(['fatal', 'severe', 'light']).values
    N = raster(X, Y, F['x0'], F['x1'], F['y0'], F['y1'], F['res'], F['sigma']); I = raster(X[inj], Y[inj], F['x0'], F['x1'], F['y0'], F['y1'], F['res'], F['sigma'])
    p0 = float(inj.mean()); share = (I + PRIOR * p0) / (N + PRIOR)  # shrunk share
    dens = N / (F['res_m'] / 1000) ** 2
    vmax = float(np.percentile(dens[dens > 0], 99.5)); v = np.clip(dens / vmax, 0, 1)
    pop = v > 0.12
    lo, hi = (float(np.percentile(share[pop], 10)), float(np.percentile(share[pop], 90))) if pop.any() else (p0 * 0.6, p0 * 1.6)
    h = np.clip((share - lo) / (hi - lo), 0, 1)
    rgb = share_colour(h)
    rgb = rgb + (WHITE - rgb) * (np.clip(v - 0.85, 0, 0.15) / 0.15)[..., None] * 0.7  # white cores at the top of the density
    alpha = np.clip(v / 0.12, 0, 1) * (0.55 + 0.4 * np.clip(v, 0, 1))
    rgba = np.dstack([np.clip(rgb, 0, 1), alpha])
    aspect = (F['y1'] - F['y0']) / (F['x1'] - F['x0'])
    W = 11.0; axw = 0.66 * W; axh = axw * aspect; H = axh + 1.9
    fig = plt.figure(figsize=(W, H), dpi=150, facecolor='#1a1a19')
    ax = fig.add_axes([0.03, 1.05 / H, 0.66, axh / H]); style_axes(ax, F)
    ax.imshow(rgba, extent=[F['x0'], F['x1'], F['y0'], F['y1']], origin='lower', interpolation='bilinear', zorder=2)
    terrebonne(ax, extent)
    top = (1.05 + axh) / H
    fig.text(0.705, top - 0.15 / H, f"{maps.TEXT[lang]['total']}: {maps.fmt(len(s), lang)}", color='#f0efec', fontsize=12, fontweight='bold', va='bottom')
    # 3 x 3 legend: density across (low, mid, high), injury share down (high, average, low), composited on the dark surface
    L = 0.3 / H; lw = 0.045; lx = 0.76; ly = top - 0.75 / H
    surf = np.array(to_rgb('#1a1a19'))
    for r_, hh in enumerate([1.0, 0.5, 0.0]):
        for c_, vv in enumerate([0.2, 0.55, 1.0]):
            col = share_colour(hh); col = col + (WHITE - col) * (max(0, vv - 0.85) / 0.15) * 0.7
            a = min(1, vv / 0.12) * (0.55 + 0.4 * vv); col = col * a + surf * (1 - a)
            lax = fig.add_axes([lx + c_ * lw, ly - (r_ + 1) * L, lw, L]); lax.set_facecolor(np.clip(col, 0, 1)); lax.set_xticks([]); lax.set_yticks([])
            for sp in lax.spines.values(): sp.set_edgecolor('#1a1a19')
    rows = [f'{hi*100:.1f} %', f'{p0*100:.1f} %', f'{lo*100:.1f} %']
    if lang == 'fr': rows = [r.replace('.', ',') for r in rows]
    for r_, lab in enumerate(rows):
        fig.text(lx - 0.008, ly - (r_ + 0.5) * L, lab, color='#c3c2b7', fontsize=8.5, va='center', ha='right')
    fig.text(lx - 0.045, ly - 1.5 * L, T[lang]['share'], color='#c3c2b7', fontsize=8.5, va='center', ha='center', rotation=90)
    cols = {'en': ['low', 'mid', 'high'], 'fr': ['faible', 'moyenne', 'forte']}[lang]
    for c_, lab in enumerate(cols):
        fig.text(lx + (c_ + 0.5) * lw, ly - 3 * L - 0.04 / H, lab, color='#c3c2b7', fontsize=8.5, va='top', ha='center')
    fig.text(lx + 1.5 * lw, ly - 3 * L - 0.22 / H, textwrap.fill(T[lang]['bar_b'].format(F['sigma_m']), 36), color='#c3c2b7', fontsize=8.5, va='top', ha='center', linespacing=1.3)
    fig.text(0.705, ly - 3 * L - 0.75 / H, textwrap.fill(T[lang]['note_b'].format(100 * p0, PRIOR), 44), color='#8f8e87', fontsize=8.5, va='top', linespacing=1.4)
    footer(fig, F, extent, lang, T[lang]['title_b'], H)
    stem = OUT / f'{extent}_heat_bivariate_{lang}'
    fig.savefig(stem.with_suffix('.png'), dpi=150, facecolor=fig.get_facecolor())
    if extent != 'island': fig.savefig(stem.with_suffix('.svg'), facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--extent', nargs='*', default=['island', 'ndg']); ap.add_argument('--kind', nargs='*', default=['multiples', 'bivariate']); ap.add_argument('--lang', nargs='*', default=['en', 'fr'])
    a = ap.parse_args()
    for e in a.extent:
        for k in a.kind:
            for l in a.lang:
                (multiples if k == 'multiples' else bivariate)(e, l); print(e, k, l)
