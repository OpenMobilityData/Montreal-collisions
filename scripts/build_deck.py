"""Slide deck (beamer 16:9, English): road safety in Côte-des-Neiges–Notre-Dame-de-Grâce from the SPVM collisions 2012–2021.

  /opt/local/bin/python3.13 scripts/build_deck.py [--final]   ->  docs/cdn_ndg_road_safety_deck_en.pdf

Every number on a slide is read from derived/tables/ at build time; the figures come from derived/figures/. Run first:
intersections.py, cdn_ndg_tables.py and the figure scripts under derived/figures/{borough-comparison,cdn-ndg-maps,
cdn-ndg-charts,queen-mary,queen-mary-decarie-hero}/ (and maps.py, point_rates_time_chart.py for the Terrebonne slides). The figure of each slide is copied to docs/deck_figures_en/NN_<figure>.png (NN = slide number). DRAFT footer until --final, which prints « Preliminary results,
not peer reviewed » instead."""
import datetime, subprocess, sys
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

FINAL = '--final' in sys.argv
ROOT = Path(__file__).resolve().parent.parent; DOCS = ROOT / 'docs'; FIG = ROOT / 'derived/figures'; T = ROOT / 'derived/tables'
now = datetime.datetime.now(ZoneInfo('America/Toronto')); TAG = now.strftime('v%Y-%m-%d-%H-%M-%S')
n = lambda x, d=0: f'{x:,.{d}f}'.replace(',', '{,}')
B = 'Côte-des-Neiges–Notre-Dame-de-Grâce'

bo = pd.read_csv(T / 'boroughs_2012-2021.csv'); city = bo[bo.territory_type == 'borough']; me = city[city.territory == B].iloc[0]
run = (T / 'cdn_ndg_tables_run.txt').read_text(); irun = (T / 'intersections_run.txt').read_text()
ov = pd.read_csv(T / 'cdn_ndg_other_vehicle_2012-2021.csv').set_index(['mode', 'other_vehicle'])
R = pd.read_csv(T / 'intersection_ranks_queen_mary_decarie.csv'); RG = pd.read_csv(T / 'intersection_ranks_terrebonne_girouard.csv')
I = pd.read_csv(T / 'intersections_2012-2021.csv'); IB = I[I.territory == B]
qc = pd.read_csv(T / 'queen_mary_corridor_2012-2021.csv').iloc[0]
h = pd.read_csv(T / 'queen_mary_decarie_collisions_2012-2021.csv')
cat = pd.read_csv(T / 'queen_mary_decarie_by_road_category.csv').set_index('road_category').collisions
share = lambda c: 100 * me[c] / city[c].sum()
rk = lambda c: int(me[f'rank_{c}'])
centroid_b = int(run.split('of which in')[1].split(':')[1].split()[0])
a_all_rank = int(R[(R['count'] == 'rec') & (R.measure == 'all') & (R.comparison_set == 'cdn_ndg')]['rank'].iloc[0])
n_int_b = len(IB); n_int_c = int((I.territory_type == 'borough').sum())
# injury share among the borough's intersections with at least 100 recorded collisions
big = IB[IB.rec_all >= 100].assign(sh=lambda d: 100 * d.rec_injury / d.rec_all).sort_values('sh', ascending=False).reset_index(drop=True)
qrow = big[big.name == 'Décarie / Queen-Mary']; q_sh = float(qrow.sh.iloc[0]); q_sh_rank = int(qrow.index[0]) + 1
MEAS = [('all', 'all collisions'), ('injury', 'with an injury or a death'), ('severe_fatal', 'with a severe injury or a death'), ('ped', 'with a pedestrian victim'), ('ped_severe_fatal', 'pedestrian victim, severe or fatal'), ('bike', 'involving a bicycle'), ('bike_severe_fatal', 'bicycle, severe or fatal')]


def rank_cell(r):
    return ('=' if r.tied_at_rank > 1 else '') + f"{int(r['rank'])}" + (rf" \textcolor{{gray}}{{\scriptsize({int(r.tied_at_rank)} tied)}}" if r.tied_at_rank > 1 else '')


short = lambda s_: s_ if len(s_) <= 20 else s_[:18] + '…'
top3 = IB.sort_values('rec_all', ascending=False).head(3).name.tolist(); assert all('Décarie' in x for x in top3) and a_all_rank == 4
assert city.residents_2021.max() == me.residents_2021


def rank_rows(Rk, count):
    out = []
    for m, lab in MEAS[:4] + MEAS[5:6]:
        a = Rk[(Rk['count'] == count) & (Rk.measure == m) & (Rk.comparison_set == 'cdn_ndg')].iloc[0]; c = Rk[(Rk['count'] == count) & (Rk.measure == m) & (Rk.comparison_set == 'city')].iloc[0]
        out.append(f"{lab} & {int(a.value)} & {rank_cell(a)} & {short(a.first_name)} ({int(a.first_value)}) & {rank_cell(c)} & {short(c.first_name)} ({int(c.first_value)}) \\\\")
    return '\n'.join(out)


a_all = R[(R['count'] == 'rec') & (R.measure == 'all') & (R.comparison_set == 'cdn_ndg')].iloc[0]; a_inj = R[(R['count'] == 'rec') & (R.measure == 'injury') & (R.comparison_set == 'cdn_ndg')].iloc[0]
c_all = R[(R['count'] == 'rec') & (R.measure == 'all') & (R.comparison_set == 'city')].iloc[0]
g_all = RG[(RG['count'] == 'rec') & (RG.measure == 'all') & (RG.comparison_set == 'cdn_ndg')].iloc[0]; g_inj = RG[(RG['count'] == 'rec') & (RG.measure == 'injury') & (RG.comparison_set == 'cdn_ndg')].iloc[0]
g_ped = RG[(RG['count'] == 'rec') & (RG.measure == 'ped') & (RG.comparison_set == 'cdn_ndg')].iloc[0]; g_bike = RG[(RG['count'] == 'rec') & (RG.measure == 'bike') & (RG.comparison_set == 'cdn_ndg')].iloc[0]


def ovrow(mode, k, lab):
    r = ov.loc[(mode, k)]; sf = r.fatal + r.severe
    return f"{lab} & {n(r.collisions)} & {int(r.fatal)} & {int(r.severe)} & {n(r.light)} & {100 * sf / r.collisions:.0f}~\\% \\\\" if r.collisions else f"{lab} & 0 & 0 & 0 & 0 & \\\\"


VEH = [('car', 'car or light truck'), ('heavy', 'heavy truck or tractor-trailer'), ('bus', 'bus or minibus'), ('other', 'other or none recorded')]
MARK = r"Preliminary results, not peer reviewed" if FINAL else r"DRAFT for the author\textquoteright{}s review"
BANNER = r"\setbeamertemplate{footline}{\hfill{\tiny\textcolor{red!70!black}{" + MARK + ", " + TAG + r"}\quad\insertframenumber}\hspace{1em}\vspace{0.4em}}"
fig = lambda folder, stem, hgt=0.84: rf"\begin{{center}}\includegraphics[width=\linewidth,height={hgt}\textheight,keepaspectratio]{{../derived/figures/{folder}/{stem}_en.png}}\end{{center}}"

DECK = r"""\documentclass[aspectratio=169,10pt]{beamer}\usetheme{default}\usecolortheme{seahorse}\setbeamertemplate{navigation symbols}{}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage{helvet}\usepackage{booktabs,amsmath}
\DeclareUnicodeCharacter{2019}{\textquoteright}\DeclareUnicodeCharacter{2013}{--}
\setbeamersize{text margin left=0.6cm,text margin right=0.6cm}\setbeamerfont{frametitle}{size=\large,series=\bfseries}
""" + BANNER + r"""
\title{Road safety in Côte-des-Neiges--Notre-Dame-de-Grâce}
\subtitle{What the police collision record shows, 2012--2021}
\author{Rick Hoge, PhD}\date{""" + ("" if FINAL else "Draft, ") + now.strftime('%B %-d, %Y') + r""" \quad {\footnotesize """ + TAG + r"""}}
\begin{document}
\begin{frame}\titlepage
{\footnotesize I am a researcher, not an engineer (not a member of the Ordre des ingénieurs du Québec): this is a data analysis, not an engineering opinion, done on my own initiative and under no professional mandate.}
\end{frame}

\begin{frame}{What the record is, and what it is not}
\begin{itemize}\setlength\itemsep{5pt}
\item \textbf{Source.} Ville de Montréal open data « Collisions routières »: the SPVM's accident reports, January 1, 2012 to December 31, 2021, one row per collision (""" + n(218272) + r""" rows), with a recorded point.
\item \textbf{Only collisions involving a motor vehicle that police attended.} A fall, a bicycle striking a pedestrian, or a collision nobody reported is not in it.
\item \textbf{It ends in 2021.} Nothing here describes a street as it is laid out today.
\item \textbf{No age and no exposure.} The record does not say how many people walked, cycled or drove through a place. \textbf{A count at a place is not a risk}: busy places have long records.
\item \textbf{Location.} The police place most collisions on the nearest intersection. """ + n(centroid_b) + r""" rows placed at a police district's centroid inside the borough (« précision nulle ») are left out of every count here.
\item \textbf{Cyclists and pedestrians are kept apart} throughout: no pooled figure.
\end{itemize}
\end{frame}

\begin{frame}{The borough's record in ten years}
\begin{columns}[T]\begin{column}{0.47\linewidth}
\begin{tabular}{@{}lr@{}}\toprule
Collisions inside the borough & """ + n(me.collisions) + r""" \\
\quad property damage only & """ + n(me.damage) + r""" \\
\quad light injury & """ + n(me.light) + r""" \\
\quad severe injury & """ + n(me.severe) + r""" \\
\quad fatal & """ + n(me.fatal) + r""" \\\midrule
With a pedestrian victim & """ + n(me.ped) + r""" \\
\quad severe or fatal collision & """ + n(me.ped_severe_fatal) + r""" \\
\quad pedestrians killed & """ + n(me.deaths_ped) + r""" \\
Involving a bicycle & """ + n(me.bike) + r""" \\
\quad severe or fatal collision & """ + n(me.bike_severe_fatal) + r""" \\
\quad cyclists killed & """ + n(me.deaths_bike) + r""" \\
Involving a motorcycle & """ + n(me.moto) + r""" \\\bottomrule
\end{tabular}
\end{column}\begin{column}{0.5\linewidth}
\begin{itemize}\setlength\itemsep{6pt}
\item """ + n(me.deaths_all) + r""" people were killed in the borough's """ + n(me.fatal) + r""" fatal collisions; """ + n(me.deaths_ped) + r""" of them were pedestrians.
\item The borough holds """ + f"{share('collisions'):.1f}" + r"""~\% of the collisions recorded in the City's 19 boroughs, """ + f"{share('ped'):.1f}" + r"""~\% of those with a pedestrian victim and """ + f"{share('bike'):.1f}" + r"""~\% of those involving a bicycle.
\item It is the City's most populous borough (""" + n(me.residents_2021) + r""" residents in the 2021 Census, by dissemination area) with """ + n(me.street_km) + r""" km of street.
\item Severity is the collision's own: in a « severe or fatal » collision with a pedestrian victim, the person most hurt is not always the pedestrian.
\end{itemize}
\end{column}\end{columns}
\end{frame}

\begin{frame}{How the borough compares with the rest of Montréal}
""" + fig('borough-comparison', 'borough_comparison', 0.86) + r"""
\end{frame}

\begin{frame}{Reading the comparison}
\small
\begin{tabular}{@{}lrrr@{}}\toprule
Rank among the 19 boroughs (1 = most) & count & per 100 km of street & per 10,000 residents \\\midrule
Collisions with an injury or a death & """ + f"{rk('injury')} & {rk('injury_per_100km')} & {rk('injury_per_10k_res')}" + r""" \\
\quad of which severe or fatal & """ + f"{rk('severe_fatal')} & {rk('severe_fatal_per_100km')} & {rk('severe_fatal_per_10k_res')}" + r""" \\
With a pedestrian victim & """ + f"{rk('ped')} & & {rk('ped_per_10k_res')}" + r""" \\
\quad of which severe or fatal & """ + f"{rk('ped_severe_fatal')} & & {rk('ped_severe_fatal_per_10k_res')}" + r""" \\
Involving a bicycle & """ + f"{rk('bike')} & & {rk('bike_per_10k_res')}" + r""" \\
\quad of which severe or fatal & """ + f"{rk('bike_severe_fatal')} & & {rk('bike_severe_fatal_per_10k_res')}" + r""" \\\bottomrule
\end{tabular}
\medskip
\begin{itemize}\setlength\itemsep{4pt}
\item By count the borough is near the top because it is the largest. Set against its streets or its residents it moves toward the middle of the 19.
\item \textbf{There is no fair single ranking.} Residents and street length are not exposure: Ville-Marie's counts per resident reflect the people who come downtown, not only those who live there. Nobody has counted trips on foot, by bicycle or by car by borough.
\item A lower bicycle count per resident may mean safer streets or fewer people cycling. The record cannot tell which.
\end{itemize}
\end{frame}

\begin{frame}{Where the collisions are recorded}
""" + fig('cdn-ndg-maps', 'cdn_ndg_heat') + r"""
\end{frame}

\begin{frame}{By electoral district}
""" + fig('cdn-ndg-charts', 'cdn_ndg_districts', 0.8) + r"""
\end{frame}

\begin{frame}{By year}
""" + fig('cdn-ndg-charts', 'cdn_ndg_yearly', 0.8) + r"""
\end{frame}

\begin{frame}{Pedestrians: where the collisions are recorded}
""" + fig('cdn-ndg-maps', 'cdn_ndg_ped_heat') + r"""
\end{frame}

\begin{frame}{Pedestrians: collisions with a death or a severe injury}
""" + fig('cdn-ndg-maps', 'cdn_ndg_ped_serious') + r"""
\end{frame}

\begin{frame}{Bicycles: where the collisions are recorded}
""" + fig('cdn-ndg-maps', 'cdn_ndg_bike_heat') + r"""
\end{frame}

\begin{frame}{Bicycles: collisions with a death or a severe injury}
""" + fig('cdn-ndg-maps', 'cdn_ndg_bike_serious') + r"""
\end{frame}

\begin{frame}{By the other vehicle}
\small
\begin{tabular}{@{}lrrrrr@{}}\toprule
 & collisions & fatal & severe & light & severe or fatal \\\midrule
\multicolumn{6}{@{}l}{\textbf{With a pedestrian victim}} \\
""" + '\n'.join(ovrow('ped', k, lab) for k, lab in VEH) + r"""
\midrule
\multicolumn{6}{@{}l}{\textbf{Involving a bicycle}} \\
""" + '\n'.join(ovrow('bike', k, lab) for k, lab in VEH) + r"""
\bottomrule
\end{tabular}
\medskip
\begin{itemize}\setlength\itemsep{4pt}
\item Order of precedence when several vehicles are recorded: heavy truck, bus, car or light truck.
\item Collisions, by the collision's severity. The heavy-vehicle rows are small: read them as counts, not as rates.
\item Bicycle rows include collisions with property damage only, so the four severity columns do not sum to the first.
\end{itemize}
\end{frame}

\begin{frame}{The intersections with the longest records}
""" + fig('cdn-ndg-charts', 'cdn_ndg_intersections', 0.8) + r"""
\end{frame}

\begin{frame}{Chemin Queen-Mary}
""" + fig('queen-mary', 'hero_queen_mary_corridor', 0.86) + r"""
\end{frame}

\begin{frame}{Queen-Mary / Décarie}
""" + fig('queen-mary-decarie-hero', 'hero_queen_mary_decarie', 0.86) + r"""
\end{frame}

\begin{frame}{Rue de Terrebonne: every collision within 25 m of the lane's axis}
""" + fig('maps', 'corridor_severity') + r"""
\end{frame}

\begin{frame}{Rue de Terrebonne: collisions involving a bicycle or a pedestrian victim}
""" + fig('maps', 'corridor_vru') + r"""
\end{frame}

\begin{frame}{Rue de Terrebonne: what can and cannot be said}
\begin{itemize}\setlength\itemsep{6pt}
\item The record predates the protected lane (built 2024): it describes the street as it was.
\item Terrebonne / Girouard, collisions recorded at the intersection: """ + f"{int(g_all.value)} (rank {int(g_all['rank'])} of {n(n_int_b)} in the borough), {int(g_inj.value)} with an injury (rank {int(g_inj['rank'])}), {int(g_ped.value)} with a pedestrian victim, {int(g_bike.value)} involving a bicycle" + r""".
\item Rates per crossing (next slide) exist only where the City counted, on a few days. For bicycles the counts differ by more than an order of magnitude between visits, so no single rate is quotable: each figure goes with its interval and its exposure method.
\item City-wide, an earlier fatal collision at the same place was not a precursor of a cyclist's death; the volume of the earlier record was, in the aggregate, and it points to the busy tenth of the city, not to a place.
\end{itemize}
\end{frame}

\begin{frame}{Terrebonne / Girouard and rue Sherbrooke: collisions per million crossings}
""" + fig('charts', 'point_rates_time') + r"""
\end{frame}

\begin{frame}{Definitions}
\small
\begin{itemize}\setlength\itemsep{4pt}
\item \textbf{Borough and districts.} The City's administrative-limits polygon and its electoral districts of 2025 (open data). A collision belongs where its recorded point falls.
\item \textbf{Severity} is the collision's own (fatal, severe injury, light injury, property damage). \textbf{Pedestrian}: at least one pedestrian victim. \textbf{Bicycle}: a bicycle involved or a cyclist victim.
\item \textbf{Denominators.} Street length: City géobase, local, collector and arterial streets, a divided street counted once per carriageway. Residents: 2021 Census, dissemination areas placed by a point inside each.
\item \textbf{Intersection.} Géobase nodes where two street names or three segment ends meet; nodes within 30 m are one intersection, and so are same-name units within 120 m (a street crossing both Décarie service roads). """ + n(n_int_b) + r""" in the borough, """ + n(n_int_c) + r""" in the City.
\item \textbf{Two counts.} Recorded at the intersection: within 30 m of its nodes, each collision given to one intersection. Within 100 m: every collision in that radius, shared between neighbours.
\item \textbf{Queen-Mary / Décarie.} One intersection. Of """ + str(len(h)) + r""" collisions within 100 m, """ + str(int(h.at_recorded_point.sum())) + r""" sit on the crossing's recorded point; """ + str(int(cat.get('ramp, collector or service road of an expressway', 0))) + r""" are coded on a ramp, collector or service road. The record cannot separate the two service roads.
\end{itemize}
\end{frame}

\begin{frame}{Limits, and where everything is}
\small
\begin{itemize}\setlength\itemsep{5pt}
\item No exposure measure by borough, street or intersection: every figure here is a count or a count against residents or street length, never a risk per trip.
\item 2012--2021 only; no age; police-attended collisions with a motor vehicle only.
\item Small numbers: a rank built on fewer than ten events moves with one collision.
\item Recorded points are approximate: """ + n(19196) + r""" rows of the file are flagged imprecise, among them every collision recorded at a service-road crossing.
\item Not done: any before-and-after of a street change; any comparison of intersections by traffic volume.
\end{itemize}
\medskip
Code, tables and figures (English and French): \texttt{github.com/OpenMobilityData/Montreal-collisions}\\
Data: Ville de Montréal, « Collisions routières », géobase, administrative limits, electoral districts (CC BY 4.0); Statistics Canada, 2021 Census. Basemap: Esri, HERE, Garmin, © OpenStreetMap contributors.
\end{frame}
\end{document}
"""

DOCS.mkdir(exist_ok=True); name = 'cdn_ndg_road_safety_deck_en'
(DOCS / f'{name}.tex').write_text(DECK)
for _ in range(2):
    r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', f'{name}.tex'], cwd=DOCS, capture_output=True, text=True, errors='replace')
    if r.returncode: print(r.stdout[-2500:]); raise SystemExit('pdflatex failed')
# the figure of each slide, named by slide number, for copying elsewhere
import re, shutil
OUTF = DOCS / 'deck_figures_en'; shutil.rmtree(OUTF, ignore_errors=True); OUTF.mkdir()
for i, frame in enumerate(DECK.split(r'\begin{frame}')[1:], 1):
    for m in re.finditer(r'\{(\.\./derived/figures/[^}]+\.png)\}', frame):
        src = (DOCS / m.group(1)).resolve(); shutil.copyfile(src, OUTF / f"{i:02d}_{src.stem.removesuffix('_en')}.png")
for ext in ('aux', 'log', 'nav', 'out', 'snm', 'toc'): (DOCS / f'{name}.{ext}').unlink(missing_ok=True)
print('written', DOCS / f'{name}.pdf', TAG, '| overfull:', r.stdout.count('Overfull'))
