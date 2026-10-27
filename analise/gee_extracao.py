"""Extração no Google Earth Engine, por município e safra (colheita H = out/H-1 a mar/H).

  clima    : ERA5-Land diário agregado por decêndio de calendário (média no polígono municipal)
  chuva    : CHIRPS diário agregado por decêndio
  modis    : MOD13Q1 + MYD13Q1 (SummaryQA <= 1) sobre a máscara de soja do MapBiomas (fração >= 0,5 a 250 m)
  landsat  : Landsat 5/7/8/9 C2 L2 com máscara QA_PIXEL (nuvem, sombra, cirro, neve) sobre a soja a 30 m

Saídas em cache: dados/gee/<produto>_<H>.csv (uma vez baixada, a safra não é refeita).
Uso: .venv/bin/python analise/gee_extracao.py [H_ini H_fim]
"""
import json
import sys
from pathlib import Path

import ee
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / 'dados/gee'
OUT.mkdir(parents=True, exist_ok=True)
ee.Initialize(project='mapbiomas-brazil')

MB = 'projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_integration_v2'
SOJA = 39

# ---------------------------------------------------------------- municípios (49 do estudo)
cods = set(pd.read_csv(RAIZ / 'resultados/base_tcc_corrigida.csv')['cod_ibge'].astype(int))
geo = json.load(open(RAIZ / 'dados/malha_municipios_pr.geojson'))


def decima(anel, max_pts=400):
    passo = max(1, len(anel) // max_pts)
    a = anel[::passo]
    return a + [a[0]] if a[-1] != a[0] else a


feats = []
for f in geo['features']:
    cod = int(f['properties']['codarea'])
    if cod not in cods:
        continue
    g = f['geometry']
    polis = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
    geom = ee.Geometry.MultiPolygon([[decima(p[0])] for p in polis])
    feats.append(ee.Feature(geom, {'cod_ibge': cod}))
MUN = ee.FeatureCollection(feats)
REGIAO = MUN.geometry().bounds()


def decendios(H):
    """18 decêndios de calendário de 1/out/H-1 a 31/mar/H: (k, início, fim exclusivo)."""
    out, k = [], 1
    for ano, mes in [(H - 1, 10), (H - 1, 11), (H - 1, 12), (H, 1), (H, 2), (H, 3)]:
        ini = ee.Date.fromYMD(ano, mes, 1)
        fim_mes = ini.advance(1, 'month')
        for d0, d1 in [(1, 11), (11, 21), (21, None)]:
            a = ee.Date.fromYMD(ano, mes, d0)
            b = ee.Date.fromYMD(ano, mes, d1) if d1 else fim_mes
            out.append((k, a, b)); k += 1
    return out


def baixa(fc, colunas):
    lin = [ft['properties'] for ft in fc.getInfo()['features']]
    return pd.DataFrame(lin)[colunas] if lin else pd.DataFrame(columns=colunas)


def mascara_soja(H):
    return ee.Image(MB).select(f'classification_{min(H, 2024)}').eq(SOJA)


# ---------------------------------------------------------------- clima e chuva
def clima(H):
    era = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    chi = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    bandas = []
    for k, a, b in decendios(H):
        e = era.filterDate(a, b)
        img = ee.Image.cat([
            e.select('temperature_2m').mean().rename('Tmed'),
            e.select('temperature_2m_max').mean().rename('Tmax'),
            e.select('temperature_2m_min').mean().rename('Tmin'),
            e.select('temperature_2m_max').max().rename('TmaxAbs'),
            e.select('temperature_2m_max').map(lambda i: i.gt(305.15)).sum().rename('dQuente'),
            e.select('dewpoint_temperature_2m').mean().rename('Td'),
            e.select('soil_temperature_level_1').mean().rename('Tsolo'),
            e.select('volumetric_soil_water_layer_1').mean().rename('Usolo1'),
            e.select('volumetric_soil_water_layer_2').mean().rename('Usolo2'),
            e.select('surface_solar_radiation_downwards_sum').sum().rename('Rg'),
            e.select('surface_net_solar_radiation_sum').sum().rename('Rn'),
            e.select('potential_evaporation_sum').sum().rename('ETp'),
            e.select('total_evaporation_sum').sum().rename('ET'),
            e.select('leaf_area_index_low_vegetation').mean().rename('IAF'),
        ])
        c = chi.filterDate(a, b).select('precipitation')
        img = img.addBands(c.sum().rename('P')).addBands(c.map(lambda i: i.lt(1)).sum().rename('dSeco'))
        bandas.append(img.rename(img.bandNames().map(lambda n: ee.String(n).cat(f'_d{k:02d}'))))
    stack = ee.Image.cat(bandas)
    fc = stack.reduceRegions(collection=MUN, reducer=ee.Reducer.mean(), scale=5000, tileScale=4)
    df = pd.DataFrame([ft['properties'] for ft in fc.getInfo()['features']])
    longo = df.melt(id_vars='cod_ibge')
    longo[['var', 'dec']] = longo['variable'].str.extract(r'(.+)_d(\d+)$')
    out = longo.pivot_table(index=['cod_ibge', 'dec'], columns='var', values='value').reset_index()
    out['dec'] = out['dec'].astype(int)
    out['colheita'] = H
    return out


# ---------------------------------------------------------------- índices de vegetação
RED = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)


def modis(H):
    proj = ee.Image(ee.ImageCollection('MODIS/061/MOD13Q1').first()).projection()
    soja = (mascara_soja(H).reduceResolution(ee.Reducer.mean(), maxPixels=1024)
            .reproject(proj).gte(0.5))
    a, b = ee.Date.fromYMD(H - 1, 9, 23), ee.Date.fromYMD(H, 4, 1)
    col = (ee.ImageCollection('MODIS/061/MOD13Q1').map(lambda i: i.set('sat', 'MOD13Q1'))
           .merge(ee.ImageCollection('MODIS/061/MYD13Q1').map(lambda i: i.set('sat', 'MYD13Q1')))
           .filterDate(a, b).filterBounds(REGIAO))

    def red(img):
        ok = img.select('SummaryQA').lte(1).And(soja)
        v = img.select(['NDVI', 'EVI']).multiply(0.0001).updateMask(ok)
        data = img.date().advance(8, 'day').format('YYYY-MM-dd')
        return (v.reduceRegions(collection=MUN, reducer=RED, scale=250, tileScale=4)
                .map(lambda f: f.set({'data': data, 'sensor': img.get('sat')})))

    df = baixa(col.map(red).flatten(), ['cod_ibge', 'data', 'sensor', 'NDVI_mean', 'EVI_mean', 'NDVI_count'])
    tot = soja.selfMask().reduceRegions(collection=MUN, reducer=ee.Reducer.count(), scale=250, tileScale=4)
    tot = baixa(tot, ['cod_ibge', 'count']).rename(columns={'count': 'n_soja'})
    return df.merge(tot, on='cod_ibge', how='left').assign(colheita=H)


LS = {'LT05': ('SR_B4', 'SR_B3', 'SR_B1'), 'LE07': ('SR_B4', 'SR_B3', 'SR_B1'),
      'LC08': ('SR_B5', 'SR_B4', 'SR_B2'), 'LC09': ('SR_B5', 'SR_B4', 'SR_B2')}


def landsat(H):
    soja = mascara_soja(H)
    a, b = ee.Date.fromYMD(H - 1, 10, 1), ee.Date.fromYMD(H, 4, 1)
    partes = []
    for sat, (nir, red_, blue) in LS.items():
        col = ee.ImageCollection(f'LANDSAT/{sat}/C02/T1_L2').filterDate(a, b).filterBounds(REGIAO)

        def red(img, nir=nir, red_=red_, blue=blue, sat=sat):
            qa = img.select('QA_PIXEL')
            limpo = qa.bitwiseAnd(0b111110).eq(0)        # bits 1-5: nuvem dilatada, cirro, nuvem, sombra, neve
            sr = img.select([nir, red_, blue]).multiply(0.0000275).add(-0.2)
            n, r, bl = sr.select(0), sr.select(1), sr.select(2)
            ndvi = n.subtract(r).divide(n.add(r)).rename('NDVI')
            evi = n.subtract(r).multiply(2.5).divide(n.add(r.multiply(6)).subtract(bl.multiply(7.5)).add(1)).rename('EVI')
            v = ndvi.addBands(evi).updateMask(limpo.And(soja))
            return (v.reduceRegions(collection=MUN, reducer=RED, scale=30, tileScale=8)
                    .map(lambda f: f.set({'data': img.date().format('YYYY-MM-dd'), 'sensor': sat})))

        partes.append(baixa(col.map(red).flatten(),
                            ['cod_ibge', 'data', 'sensor', 'NDVI_mean', 'EVI_mean', 'NDVI_count']))
    tot = soja.selfMask().reduceRegions(collection=MUN, reducer=ee.Reducer.count(), scale=30, tileScale=8)
    tot = baixa(tot, ['cod_ibge', 'count']).rename(columns={'count': 'n_soja'})
    df = pd.concat(partes, ignore_index=True)
    return df.merge(tot, on='cod_ibge', how='left').assign(colheita=H)


PRODUTOS = {'clima': clima, 'modis': modis, 'landsat': landsat}

if __name__ == '__main__':
    h0, h1 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2001, 2025)
    for H in range(h0, h1 + 1):
        for nome, fun in PRODUTOS.items():
            arq = OUT / f'{nome}_{H}.csv'
            if arq.exists():
                continue
            try:
                df = fun(H)
                df.to_csv(arq, index=False)
                print(H, nome, df.shape, flush=True)
            except Exception as e:
                print(H, nome, 'ERRO', str(e)[:200], flush=True)
