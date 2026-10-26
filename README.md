# Montreal-collisions: police-reported road collisions in Montréal, from open data

Two public series of police-reported collisions, kept with their provenance, the code that reads them, and the tables, maps and charts drawn from them. Companion to [Terrebonne-open-sources](https://github.com/OpenMobilityData/Terrebonne-open-sources) and [Terrebonne-access-requests](https://github.com/OpenMobilityData/Terrebonne-access-requests), which concern the rue de Terrebonne bike lane; the data here are city-wide and the repository is meant to outlast that question. The READMEs and notes describe and count; they do not evaluate.

**First deliverable (October 26, 2026):** the record preceding each of the 31 cyclist fatalities of 2012–2021 at the same location: [data note (PDF, 6 pp.)](derived/cyclist-fatality-locations/cyclist_fatality_locations_note_en.pdf) · [the tables in Markdown](derived/cyclist-fatality-locations/tables.md) · [one row per fatality (CSV)](derived/cyclist-fatality-locations/cyclist_fatality_locations_2012-2021.csv) · [rue de Terrebonne on the same scale (CSV)](derived/cyclist-fatality-locations/terrebonne_same_scale_2012-2021.csv) · [summaries](derived/cyclist-fatality-locations/summary_earlier_fatal.csv), [by severity and mode](derived/cyclist-fatality-locations/summary_earlier_record.csv).

## The two series

| | SPVM (City of Montréal open data) | SAAQ (Données Québec open data) |
|---|---|---|
| **Dataset** | *Collisions routières*, [donnees.montreal.ca/dataset/collisions-routieres](https://donnees.montreal.ca/dataset/collisions-routieres) | *Rapports d'accident*, [donneesquebec.ca › rapports-d-accident](https://www.donneesquebec.ca/recherche/dataset/rapports-d-accident) |
| **Issuer** | Ville de Montréal, from the SPVM's reports (SAAQ coding) | Société de l'assurance automobile du Québec |
| **Coverage** | Collisions reported to the police on the SPVM's territory (the agglomeration), January 1, 2012 to December 31, 2021; 218,272 rows, 68 columns, one row per collision, with a location (MTM zone 8 and latitude/longitude; 11 rows without, 19,196 flagged imprecise) | Every police-reported collision in Québec, one file per year, 2011 to 2022 (108,186 to 182,915 rows a year); 25 columns; no coordinates; the administrative region (`REG_ADM`, « Montréal (06) ») and the mode indicators are the only geography |
| **Files** | `raw/spvm/collisions_routieres_MTL.csv` (82.2 MB; the portal's CSV resource dated October 6, 2022; downloaded May 29, 2024; re-downloaded October 26, 2026 and found byte-identical, SHA-256 `d71904aa1bff6a4cee3a159294899fbad367fca7aeb4b41dd07074b525be38ce`) and the City's two data dictionaries (PDF, downloaded October 26, 2026) | `raw/saaq/Rapport_Accident_<year>.csv.gz` (the twelve CSVs downloaded June 10, 2024, stored gzipped and otherwise unmodified; 2022 checked against the server's size on October 26, 2026) and the SAAQ's variable documentation (PDF, December 2023) |
| **Licence** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (stated on the portal) | [CC BY 4.0](https://www.donneesquebec.ca/licence/#cc-by) (stated on the portal) |
| **Link status** | Both resources served on October 26, 2026 (the City portal answers a plain `curl` with a redirect to a signed storage URL; follow redirects) | Served October 26, 2026 |

The two series are never merged: different years, territory and reporting. Each figure names its series and its years.

## Definitions (SPVM series, `scripts/spvm.py`)

- **Severity** is the collision's own `GRAVITE`: *Mortel* (fatal), *Grave* (severe injury), *Léger* (light injury), and the two property-damage classes (*Dommages matériels seulement*, *… inférieurs au seuil de rapportage*) merged as *damage*. Fatal collisions: 263, the same rows as `NB_MORTS > 0`.
- **Mode.** A collision *involves a bicycle* when `nb_bicyclette > 0` or `NB_VICTIMES_VELO > 0` (8,685); *a pedestrian* when `NB_VICTIMES_PIETON > 0` (10,722); *a motorcycle* when `nb_motocyclette > 0` or `NB_VICTIMES_MOTO > 0` (2,947); *occupants only* otherwise. A collision may be in more than one of the first three.
- **Street key** for « same street name » matches: upper case, accents and hyphens removed, generic words dropped (rue, av, boul, ch, O, E, face, opp…), *St* expanded to *Saint*, *N-D* to *Notre Dame*, tokens sorted. A crude match, kept beside the distance definitions.
- **Distances** are planar, on the City's `LOC_X`/`LOC_Y` (MTM zone 8, NAD83); `spvm.mtm8()` projects a latitude/longitude to the same system (checked against the file's own pairs: within 5 cm).
- **Extents** of the earlier figures (latitude/longitude boxes, strict bounds): NDG box 45.4390–45.4875 N, 73.7005–73.5808 W (50.3 km²); H4C box 45.4645–45.4850 N, 73.5870–73.5640 W (4.1 km²); the Terrebonne & Somerled corridor is a street-name selection.

## Reconciliation with the 2024 figures (`scripts/reproduce_2024.py` → `derived/tables/reproduce_2024.txt`)

A set of maps and charts drawn from the same CSV in 2024 with a MATLAB script (retired; archived outside this repository) printed these totals: 8,697 cyclist collisions (31 fatal, 280 severe, 6,298 light, 2,088 damage), 218,685 collisions in all, 10,814 pedestrian and 2,963 motorcycle collisions, 22,101 in the NDG box, 3,736 in the H4C box, 516 on Terrebonne and Somerled. Every one of these is reproduced exactly as the **sum of overlapping classes**: that script built each severity class from several conditions (for instance, a collision with a lightly injured cyclist and a severely injured other person fell in both the light and the severe class), and its legend totals added the classes. Counted as distinct rows the figures are 8,685 (31 / 279 / 6,287 / 2,088), 218,272, 10,722, 2,947, 22,062, 3,733 and 515. The 2nd-vehicle counts of the 2024 cyclist map match (cars and light trucks 7,617; heavy trucks 257; buses 130; motorcycles 44; construction equipment 15; other 186) except taxis, printed as 2 from a `> 1` test where `> 0` gives 414. The area printed on each 2024 map is the box's diagonal multiplied by its width, not its area (NDG printed 100.6 km², rectangle 50.3; Montréal printed 1,942.3, bounding rectangle 1,273.6). This repository reports distinct collisions and rectangle areas.

## Layout

| Path | Content |
|---|---|
| `raw/spvm/`, `raw/saaq/` | The files as downloaded (gzipped for SAAQ), never edited, with the issuers' dictionaries |
| `scripts/spvm.py` | Loader for the SPVM file: typed columns, dates, severity and mode classes, street key, extents, distances, MTM projection |
| `scripts/saaq.py` | Loader for the SAAQ files (per year, Montréal filter) and the yearly table by severity, collision type and mode (`derived/tables/saaq_{montreal,quebec}_yearly_2011-2022.csv`) |
| `scripts/reproduce_2024.py` | The reconciliation above |
| `scripts/cyclist_fatality_locations.py` | The record preceding each cyclist fatality (three definitions of the same location, by severity and mode), rue de Terrebonne on the same scale, the summaries, `tables.md` |
| `derived/cyclist-fatality-locations/` | Its outputs, the note (`build.py` regenerates the LaTeX tables from the CSVs, stamps the version and compiles) and the run log |
| `derived/tables/` | Extent and yearly tables |
| `derived/figures/maps/`, `derived/figures/charts/` | Maps and charts (to come), PNG and SVG, English and French, the script beside its figures |

Python 3.13 with pandas, numpy and matplotlib; `pdflatex` for the note. Scripts run from anywhere (paths are relative to the repository).

## Licence

Data: CC BY 4.0, as published by the Ville de Montréal and the SAAQ (attribution above). Scripts, tables, figures and notes: CC BY 4.0, Rick Hoge, 2026. Basemap tiles used in the maps are credited on each map and not redistributed.

## About the author

The analyses were written by Rick Hoge, a resident of Côte-des-Neiges–Notre-Dame-de-Grâce who cycles in every season. He works as a data scientist and is a professor in the Faculty of Medicine at McGill University, where his research is in neuroscience and biomedical engineering, including the effects of physical activity on brain health. The work in this repository is personal: it was done as a resident, on his own time, from open data, and represents no position of McGill University or of any other organization; the affiliation is given for identification only. The commits are co-authored with Claude (Anthropic's model), which wrote the code and drafts under the author's direction and review; the full history is public by design. Corrections are welcome through the repository's issue tracker.

## Maps and charts (`scripts/maps.py`, `scripts/charts.py`, `scripts/basemap.py`; October 26, 2026)

Figures are in `derived/figures/maps/` and `derived/figures/charts/`, PNG and SVG (the island maps PNG only: 218,000 points), English and French (`_en`, `_fr`). Colours follow a validated palette (the `dataviz` skill's reference palette, fixed per class, never cycled); marks are thin, every legend carries the count per class and the total, every title names the extent, its area as a rectangle and the period.

**Maps** (dark basemap: Esri « World Dark Gray Base », credited on each map, tiles fetched into a cache outside the repository and not redistributed). Four extents: `island` (every collision with coordinates), `ndg` (the NDG box above), `h4c` (the H4C box), `corridor` (collisions within 25 m of the Terrebonne lane axis from Belmore to Girouard, with the street-name selection printed beside as the comparison). Two views: `severity` (every collision; shape = mode, with pedestrian ▽, bicycle □, motorcycle ◇, occupants only ○; colour and size = the collision's severity, fatal drawn last in white with a ring) and `vehicle` (collisions involving a bicycle, coloured by the other vehicle recorded, in this order of precedence: heavy truck or tractor-trailer, bus or minibus, car or light truck, other or none recorded). Collisions recorded at one point are spread on a small spiral so that every marker is visible, the most severe outermost, and a note on each map says so (not on the island map). Two further views: `island_bike_fatal` (the 31 collisions in which a cyclist was killed, numbered in date order with a three-column key under the map, coloured by the other vehicle recorded) and `corridor_bike` (every collision involving a bicycle within 25 m of the Terrebonne lane axis or with Terrebonne in a street field, 15 in 2012–2021, each labelled with its date and coloured by severity).

**Charts.** `saaq_montreal_severity` and `saaq_montreal_type`: collisions per year in the SAAQ's Montréal region 2011–2022 as small multiples (one panel per class, each from zero: the classes differ by two orders of magnitude, so one shared axis would flatten the injury series); `spvm_<island|ndg>_modes`: collisions per year involving a pedestrian victim, a bicycle or a motorcycle, SPVM 2012–2021, legend with the ten-year totals; `spvm_<island|ndg>_shares`: the same as a share of all collisions, with a fitted linear trend (dotted) and the last value labelled. Yearly tables behind them: `derived/tables/spvm_<extent>_modes_yearly_2012-2021.csv`, `saaq_montreal_yearly_2011-2022.csv`.
