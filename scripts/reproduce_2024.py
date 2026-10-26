"""Reproduce the counts of the 2024 MATLAB figures from the same CSV, and state the differences."""
import sys; sys.path.insert(0, 'port')
import spvm, pandas as pd, numpy as np
d = spvm.load()
z = lambda c: d[c].fillna(0)
print('rows', len(d), '| LOC_X/Y missing', int(d.LOC_X.isna().sum()), '| lat missing', int(d.LOC_LAT.isna().sum()))
bike = z('nb_bicyclette') > 0
# --- MATLAB cyclist severity classes (unions, overlapping) ---
fatal = (bike & (z('NB_MORTS') > 0)) | (z('NB_DECES_VELO') > 0)
severe = (bike & (z('NB_BLESSES_GRAVES') > 0)) | ((z('NB_VICTIMES_VELO') > 0) & (z('NB_BLESSES_GRAVES') > 0))
light = (bike & (z('NB_BLESSES_LEGERS') > 0)) | ((z('NB_BLESSES_VELO') > 0) & (z('NB_BLESSES_GRAVES') < 1)) | ((z('NB_VICTIMES_VELO') > 0) & (z('NB_BLESSES_GRAVES') < 1) & (z('NB_MORTS') < 1))
damage = bike & ((z('NB_VICTIMES_TOTAL') < 1) | d.GRAVITE.isin(['Dommages matériels seulement', 'Dommages matériels inférieurs au seuil de rapportage']))
print('2024 cyclist classes (sum of overlapping classes): fatal', fatal.sum(), 'severe', severe.sum(), 'light', light.sum(), 'damage', damage.sum(), 'sum', fatal.sum()+severe.sum()+light.sum()+damage.sum(), '(2024: 31/280/6298/2088 = 8697)')
print('distinct cyclist collisions', int(d.bike.sum()), '| by GRAVITE', d[d.bike].severity.value_counts().to_dict())
# 2nd vehicle
v = dict(car=bike & (z('nb_automobile_camion_leger') > 0), heavy=bike & (z('nb_camionLourd_tractRoutier') > 0), bus=bike & (z('nb_tous_autobus_minibus') > 0),
         moto=bike & (z('nb_motocyclette') > 0), equip=bike & (z('nb_outil_equipement') > 0), taxi_gt1=bike & (z('nb_taxi') > 1), taxi_gt0=bike & (z('nb_taxi') > 0),
         other=bike & ((z('nb_motoneige') > 0) | (z('nb_VHR') > 0) | (z('nb_autres_types') > 0) | (z('nb_veh_non_precise') > 0)))
print('2nd vehicle:', {k: int(m.sum()) for k, m in v.items()}, '(2024: 7617/257/130/44/15/2/186 = 8251)')
# pedestrians / motos (MATLAB classes)
pf = (z('NB_DECES_PIETON') > 0) | ((z('NB_VICTIMES_PIETON') > 0) & (z('NB_MORTS') > 0))
ps = (z('NB_VICTIMES_PIETON') > 0) & (z('NB_BLESSES_GRAVES') > 0)
pl = ((z('NB_VICTIMES_PIETON') > 0) & (z('NB_BLESSES_LEGERS') > 0)) | ((z('NB_BLESSES_PIETON') > 0) & (z('NB_BLESSES_GRAVES') < 1))
print('2024 pedestrian classes sum', pf.sum()+ps.sum()+pl.sum(), '(2024: 10814) | distinct', int(d.ped.sum()))
mo = z('nb_motocyclette') > 0
mf = (mo & (z('NB_MORTS') > 0)) | (z('NB_DECES_MOTO') > 0)
ms = (mo & (z('NB_BLESSES_GRAVES') > 0)) | ((z('NB_VICTIMES_MOTO') > 0) & (z('NB_BLESSES_GRAVES') > 0))
ml = (mo & (z('NB_BLESSES_LEGERS') > 0)) | ((z('NB_BLESSES_MOTO') > 0) & (z('NB_BLESSES_GRAVES') < 1)) | ((z('NB_VICTIMES_MOTO') > 0) & (z('NB_BLESSES_GRAVES') < 1) & (z('NB_MORTS') < 1))
md = mo & ((z('NB_VICTIMES_TOTAL') < 1) | d.GRAVITE.isin(['Dommages matériels seulement', 'Dommages matériels inférieurs au seuil de rapportage']))
print('2024 moto classes sum', mf.sum()+ms.sum()+ml.sum()+md.sum(), '(2024: 2963) | distinct', int(d.moto.sum()))
auto = (~bike) & (z('NB_VICTIMES_PIETON') < 1) & (~mo)
af = auto & (z('NB_MORTS') > 0); as_ = auto & (z('NB_BLESSES_GRAVES') > 0); al = auto & (z('NB_BLESSES_LEGERS') > 0)
ad = auto & ((z('NB_VICTIMES_TOTAL') < 1) | d.GRAVITE.isin(['Dommages matériels seulement', 'Dommages matériels inférieurs au seuil de rapportage']))
asum = af.sum()+as_.sum()+al.sum()+ad.sum()
print('2024 auto classes sum', asum, '| grand sum of all 2024 classes', asum + (fatal.sum()+severe.sum()+light.sum()+damage.sum()) + (pf.sum()+ps.sum()+pl.sum()) + (mf.sum()+ms.sum()+ml.sum()+md.sum()), '(2024: 218685) | distinct rows', len(d))
# --- extents ---
for key in ['ndg', 'h4c']:
    e = spvm.EXTENTS[key]; m = spvm.in_box(d, e['box']); s = d[m]
    b = spvm.extent_bounds(s)
    print(f"{e['name']}: {m.sum()} collisions | fatal by mode (collision GRAVITE): ped {int((s.fatal & s.ped).sum())}, bike {int((s.fatal & s.bike).sum())}, moto {int((s.fatal & s.moto).sum())}, motor vehicles only {int((s.fatal & s.motor_only).sum())}"
          f" | severe: ped {int(((s.severity=='severe') & s.ped).sum())}, bike {int(((s.severity=='severe') & s.bike).sum())}"
          f" | area of the points' bounds {spvm.box_area_km2(*b):.1f} km² (2024 formula {spvm.box_area_km2_2024(*b):.1f}); of the box itself {spvm.box_area_km2(*e['box'][0], *e['box'][1]):.1f}")
m = spvm.on_street(d, 'TERREBONNE', 'TERREBONE', 'SOMERLED'); s = d[m]; b = spvm.extent_bounds(s)
print(f"Terrebonne & Somerled by name: {m.sum()} (2024: 516) | fatal {int(s.fatal.sum())} | bounds area {spvm.box_area_km2(*b):.1f} km² (2024 formula {spvm.box_area_km2_2024(*b):.1f}; printed 7.0)")
print('  by severity x mode:', s.groupby(['severity']).agg(ped=('ped','sum'), bike=('bike','sum'), moto=('moto','sum'), occ=('motor_only','sum'), n=('bike','size')).to_dict('index'))
t = spvm.on_street(d, 'TERREBONNE', 'TERREBONE'); s = d[t]
print(f"Terrebonne by name: {t.sum()} | fatal {int(s.fatal.sum())} severe {int((s.severity=='severe').sum())} bike {int(s.bike.sum())} ped {int(s.ped.sum())} | TERREBONE-only rows {int((spvm.on_street(d,'TERREBONE') & ~spvm.on_street(d,'TERREBONNE')).sum())}")
b = spvm.extent_bounds(d); print(f"Montréal bounds area {spvm.box_area_km2(*b):.1f} km² (2024 formula {spvm.box_area_km2_2024(*b):.1f}; printed 1942.3)")
print('cyclist deaths per year', d[z('NB_DECES_VELO')>0].groupby('year').size().reindex(range(2012,2022), fill_value=0).tolist())
