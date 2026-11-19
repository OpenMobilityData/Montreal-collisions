"""Shared drawing code for the maps of a territory or a small area (borough, street, intersection): dark basemap,
polygon outlines, a density layer, severity markers. Used by the figure scripts under derived/figures/<type>/.
Palette, kernel, spiral spread and credits are those of maps.py."""
from __future__ import annotations
import math, textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
import basemap, maps, areas

BG = '#1a1a19'; INK = '#f0efec'; INK2 = '#c3c2b7'; INK3 = '#8f8e87'; EDGE = '#52514e'
SEV = maps.SEV; SEV_LABEL = maps.SEV_LABEL
MODE_MARK = {'ped': 'v', 'bike': 's', 'moto': 'D', 'motor_only': 'o'}
SRC = {'en': 'Source: Ville de Montréal, open data « Collisions routières » (SPVM reports, 2012–2021), CC BY 4.0. Boundaries and streets: Ville de Montréal open data, CC BY 4.0. Basemap: ',
       'fr': 'Source : Ville de Montréal, données ouvertes « Collisions routières » (rapports du SPVM, 2012–2021), CC BY 4.0. Limites et rues : données ouvertes de la Ville de Montréal, CC BY 4.0. Fond de carte : '}
REPO = 'github.com/OpenMobilityData/Montreal-collisions'
CMAP = maps.HEAT_CMAP


def fmt(n, lang):
    s = f'{int(n):,}'
    return s.replace(',', ' ') if lang == 'fr' else s


def dec(v, lang, nd=1):
    s = f'{v:,.{nd}f}'
    return s.replace(',', ' ').replace('.', ',') if lang == 'fr' else s


class Map:
    """One map panel on a dark figure. box = ((lat0, lat1), (lon0, lon1))."""
    def __init__(self, fig, rect, box, zoom, brighten=1.0, tiles=True):
        (self.lat0, self.lat1), (self.lon0, self.lon1) = box
        self.x0, self.y0 = basemap.merc(self.lon0, self.lat0); self.x1, self.y1 = basemap.merc(self.lon1, self.lat1)
        self.fig = fig; self.ax = ax = fig.add_axes(rect); ax.set_facecolor('#232322' if not tiles else BG)
        if tiles:  # tiles=False: a plain dark ground for scales finer than the basemap's last zoom level (16); draw streets with lines()
            img, ext = basemap.basemap(self.lon0, self.lon1, self.lat0, self.lat1, zoom)
            ax.imshow(np.clip(img * brighten, 0, 1), extent=ext, interpolation='bicubic', zorder=0)
        ax.set_xlim(self.x0, self.x1); ax.set_ylim(self.y0, self.y1); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(EDGE)
        self.k = 1 / math.cos(math.radians((self.lat0 + self.lat1) / 2))  # Web Mercator metres per ground metre
        W, H = fig.get_size_inches(); self.px = (self.x1 - self.x0) / (rect[2] * W * fig.dpi)  # mercator metres per pixel

    @staticmethod
    def aspect(box):
        (lat0, lat1), (lon0, lon1) = box
        x0, y0 = basemap.merc(lon0, lat0); x1, y1 = basemap.merc(lon1, lat1)
        return (y1 - y0) / (x1 - x0)

    def xy(self, lon, lat):
        return basemap.merc(np.asarray(lon, float), np.asarray(lat, float))

    def outline(self, gdf, color=INK, lw=1.2, ls='-', alpha=0.9, z=6):
        for geom in gdf.to_crs('EPSG:3857').geometry:
            for poly in (geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]):
                x, y = poly.exterior.xy; self.ax.plot(x, y, color=color, lw=lw, ls=ls, alpha=alpha, zorder=z, solid_joinstyle='round')

    def shade_outside(self, gdf, alpha=0.45):
        from shapely.geometry import box as sbox
        inside = gdf.to_crs('EPSG:3857').union_all()
        out = sbox(self.x0, self.y0, self.x1, self.y1).difference(inside)
        for poly in (out.geoms if out.geom_type == 'MultiPolygon' else [out]):
            x, y = poly.exterior.xy; self.ax.fill(x, y, color=BG, alpha=alpha, zorder=3, lw=0)

    def lines(self, gdf, color=INK2, lw=0.8, alpha=0.6, z=1, ls='-'):
        for geom in gdf.to_crs('EPSG:3857').geometry:
            for part in (geom.geoms if geom.geom_type == 'MultiLineString' else [geom]):
                x, y = part.xy; self.ax.plot(x, y, color=color, lw=lw, alpha=alpha, zorder=z, ls=ls, solid_capstyle='round')

    def heat(self, lon, lat, sigma_m, res_m, vmax=None):
        X, Y = self.xy(lon, lat); r = res_m * self.k
        nx = int((self.x1 - self.x0) / r) + 1; ny = int((self.y1 - self.y0) / r) + 1
        Hc, _, _ = np.histogram2d(X, Y, bins=[nx, ny], range=[[self.x0, self.x1], [self.y0, self.y1]])
        dens = gaussian_filter(Hc.T, sigma=sigma_m / res_m) / (res_m / 1000) ** 2
        if vmax is None: vmax = float(np.percentile(dens[dens > 0], 99.5)) if (dens > 0).any() else 1.0
        dens = np.minimum(dens, vmax)  # values above the bar's top take its last colour
        return self.ax.imshow(dens, extent=[self.x0, self.x1, self.y0, self.y1], origin='lower', cmap=CMAP, vmin=0, vmax=vmax, interpolation='bilinear', zorder=4)

    def severity_markers(self, s, step_px=5.5, scale=1.0, by_mode=True, z0=7):
        """Markers coloured and sized by severity; shape by road user when by_mode. Returns the spread positions."""
        X, Y = self.xy(s.LOC_LONG.values, s.LOC_LAT.values)
        rank = s.severity.map({'damage': 0, 'light': 1, 'severe': 2, 'fatal': 3}).fillna(0).values
        SX, SY = maps.spread(X, Y, step_px * self.px, rank)
        for sev in ['damage', 'light', 'severe', 'fatal']:
            st = SEV[sev]
            for m_, marker in (MODE_MARK.items() if by_mode else [(None, 'o')]):
                m = (s.severity == sev).values & (s[m_].values if m_ else True)
                if by_mode and m_ == 'bike': m &= ~s.ped.values
                if by_mode and m_ == 'moto': m &= ~(s.ped.values | s.bike.values)
                if not m.any(): continue
                edge = sev in ('fatal', 'severe')
                self.ax.scatter(SX[m], SY[m], s=st['size'] * scale, c=st['color'], marker=marker, linewidths=0.8 if edge else 0, edgecolors=BG if edge else 'none',
                                alpha=0.92 if sev != 'damage' else 0.65, zorder=z0 + st['z'])
        return SX, SY

    def scalebar(self, metres, label):
        L = metres * self.k; bx = self.x0 + (self.x1 - self.x0) * 0.04; by = self.y0 + (self.y1 - self.y0) * 0.045
        self.ax.plot([bx, bx + L], [by, by], color=INK, lw=2, solid_capstyle='butt', zorder=20)
        self.ax.text(bx + L / 2, by + (self.y1 - self.y0) * 0.015, label, color=INK, ha='center', fontsize=8.5, zorder=20, bbox=dict(boxstyle='round,pad=0.15', fc=BG, ec='none', alpha=0.7))

    def label(self, lon, lat, text, dx=5, dy=4, size=8, color=INK, ha='left', z=21, alpha=0.75):
        x, y = self.xy(lon, lat)
        self.ax.annotate(text, (float(x), float(y)), xytext=(dx, dy), textcoords='offset points', color=color, fontsize=size, zorder=z, ha=ha,
                         bbox=dict(boxstyle='round,pad=0.18', fc=BG, ec='none', alpha=alpha))


def sev_handles(lang, counts, marker='o'):
    return [Line2D([], [], marker=marker, color='none', markerfacecolor=SEV[k]['color'], markeredgecolor=BG if k in ('fatal', 'severe') else 'none',
                   markersize=max(4.5, math.sqrt(SEV[k]['size']) * 1.1), label=f"{SEV_LABEL[lang][k]}: {fmt(counts.get(k, 0), lang)}") for k in ['fatal', 'severe', 'light', 'damage'] if k in counts]


def mode_handles(lang, counts):
    return [Line2D([], [], marker=MODE_MARK[k], color='none', markerfacecolor=INK2, markersize=7, label=f"{maps.MODE_LABEL[lang][k]}: {fmt(counts[k], lang)}") for k in MODE_MARK if k in counts]


def titles(fig, title, sub, top, left=0.03):
    H = fig.get_size_inches()[1]
    fig.text(left, top + 0.42 / H, title, ha='left', va='bottom', color=INK, fontsize=15, fontweight='bold')
    fig.text(left, top + 0.12 / H, sub, color=INK2, fontsize=10.5, ha='left', va='bottom')


def footer(fig, lang, width=150, left=0.03):
    H = fig.get_size_inches()[1]
    fig.text(left, 0.5 / H, textwrap.fill(SRC[lang] + basemap.ATTRIBUTION + '.', width), color=INK2, fontsize=8.5, va='bottom', linespacing=1.4)
    fig.text(left, 0.25 / H, REPO, color=EDGE, fontsize=8, va='bottom')


def save(fig, out: Path, stem, svg=False):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{stem}.png', dpi=150, facecolor=fig.get_facecolor())
    if svg: fig.savefig(out / f'{stem}.svg', facecolor=fig.get_facecolor())
    plt.close(fig)
