"""Base principal município x safra a partir da extração do GEE (gee_extracao.py) e do IBGE (00_ibge.py).

Safra = ano de colheita H (out/H-1 a mar/H). Decêndios de calendário 1-18 a partir de 1/out.
Fases: F1 out (dec 1-3), F2 nov-dez (4-9), F3 jan (10-12), F4 fev-mar (13-18).
Índices de vegetação: observações com pelo menos 30% dos pixels de soja válidos; NDVI e EVI em [0, 1].
Duas variantes de índices (A3): só MODIS (Terra+Aqua) e fusão MODIS + Landsat, com o Landsat calibrado
para a escala do MODIS por regressão linear nos decêndios com observação dos dois.
Saídas: resultados/base_safra.csv (só MODIS, principal), resultados/base_safra_fusao.csv,
        resultados/calibracao_landsat_modis.csv
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
GEE, RES = RAIZ / 'dados/gee', RAIZ / 'resultados'
FASE = {**{d: 1 for d in range(1, 4)}, **{d: 2 for d in range(4, 10)},
        **{d: 3 for d in range(10, 13)}, **{d: 4 for d in range(13, 19)}}
FRAC_MIN = 0.3


def le(prod):
    arqs = sorted(glob.glob(str(GEE / f'{prod}_*.csv')))
    return pd.concat([pd.read_csv(a) for a in arqs], ignore_index=True)


def decendio(datas, H):
    """Decêndio de calendário (1-18) contado a partir de 1/out/H-1; 0 fora da janela."""
    d = pd.to_datetime(datas).reset_index(drop=True)
    mes_idx = (d.dt.year - (H - 1)) * 12 + d.dt.month - 10          # out=0 ... mar=5
    sub = np.where(d.dt.day <= 10, 0, np.where(d.dt.day <= 20, 1, 2))
    k = mes_idx * 3 + sub + 1
    return np.where((mes_idx >= 0) & (mes_idx <= 5), k, 0)


# ---------------------------------------------------------------- clima por fase
c = le('clima')
for v in ['Tmed', 'Tmax', 'Tmin', 'TmaxAbs', 'Td', 'Tsolo']:
    c[v] = c[v] - 273.15
c['ETp'] = -c['ETp'] * 1000
c['ET'] = -c['ET'] * 1000
c['Rg'] = c['Rg'] / 1e6
c['Rn'] = c['Rn'] / 1e6
c['BH'] = c['P'] - c['ETp']
c['fase'] = c['dec'].map(FASE)
AGG = {'P': 'sum', 'dSeco': 'sum', 'ETp': 'sum', 'ET': 'sum', 'BH': 'sum', 'dQuente': 'sum',
       'Tmed': 'mean', 'Tmax': 'mean', 'Tmin': 'mean', 'TmaxAbs': 'max', 'Td': 'mean', 'Tsolo': 'mean',
       'Usolo1': 'mean', 'Usolo2': 'mean', 'Rg': 'sum', 'Rn': 'sum', 'IAF': 'mean'}
Xc = c.groupby(['cod_ibge', 'colheita', 'fase']).agg(AGG).unstack('fase')
Xc.columns = [f'{v}_F{f}' for v, f in Xc.columns]


# ---------------------------------------------------------------- índices de vegetação por decêndio
def veg(prod):
    d = le(prod).dropna(subset=['NDVI_mean'])
    d = d[(d['NDVI_count'] / d['n_soja'] >= FRAC_MIN)]
    for v in ['NDVI_mean', 'EVI_mean']:
        d.loc[(d[v] < 0) | (d[v] > 1), v] = np.nan
    d['dec'] = decendio(d['data'], d['colheita'].values)
    d = d[d['dec'] > 0]
    return (d.groupby(['cod_ibge', 'colheita', 'dec'])[['NDVI_mean', 'EVI_mean']].mean()
            .rename(columns={'NDVI_mean': 'NDVI', 'EVI_mean': 'EVI'}))


mod, ls = veg('modis'), veg('landsat')

# calibração Landsat -> MODIS nos decêndios comuns
par = mod.join(ls, lsuffix='_mod', rsuffix='_ls', how='inner').dropna()
cal = {}
for v in ['NDVI', 'EVI']:
    b, a = np.polyfit(par[f'{v}_ls'], par[f'{v}_mod'], 1)
    r = np.corrcoef(par[f'{v}_ls'], par[f'{v}_mod'])[0, 1]
    cal[v] = (a, b)
    print(f'{v}: MODIS = {a:.3f} + {b:.3f} x Landsat (r = {r:.2f}, n = {len(par)})')
pd.DataFrame([{'indice': v, 'intercepto': a, 'inclinacao': b} for v, (a, b) in cal.items()]) \
    .to_csv(RES / 'calibracao_landsat_modis.csv', index=False)
ls_cal = ls.copy()
for v, (a, b) in cal.items():
    ls_cal[v] = a + b * ls_cal[v]
fusao = pd.concat([mod, ls_cal]).groupby(level=[0, 1, 2]).mean()


def indices_por_fase(d):
    d = d.reset_index()
    d['fase'] = d['dec'].map(FASE)
    a = d.groupby(['cod_ibge', 'colheita', 'fase']).agg(NDVImax=('NDVI', 'max'), NDVImed=('NDVI', 'mean'),
                                                        EVImax=('EVI', 'max'), EVImed=('EVI', 'mean'))
    a = a.unstack('fase')
    a.columns = [f'{v}_F{f}' for v, f in a.columns]
    s = d.groupby(['cod_ibge', 'colheita']).agg(NDVImax_safra=('NDVI', 'max'), EVImax_safra=('EVI', 'max'),
                                               n_obs_veg=('NDVI', 'count'))
    return a.join(s)


# ---------------------------------------------------------------- alvo, área de soja e estáticos
ibge = pd.read_csv(RAIZ / 'dados/ibge_1612_soja_pr.csv').rename(columns={'ano_colheita': 'colheita',
                                                                         'prod_kg_ha': 'prod'})
area = le('landsat').groupby(['cod_ibge', 'colheita'])['n_soja'].first().mul(0.09).rename('area_soja_ha')
mun = pd.read_csv(RAIZ / 'dados/municipios_pr.csv')[['cod_ibge', 'municipio', 'microrregiao', 'latitude',
                                                       'longitude', 'altitude_m']]
solo = pd.read_csv(RES / 'base_tcc_corrigida.csv').groupby('cod_ibge')[['argila', 'areia']].first()
nomes = pd.read_csv(RES / 'base_tcc_corrigida.csv').groupby('cod_ibge')['cidade'].first()

for nome, v in [('base_safra', mod), ('base_safra_fusao', fusao)]:
    b = Xc.join(indices_por_fase(v)).join(area).reset_index()
    b = b.merge(ibge, on=['cod_ibge', 'colheita']).merge(mun, on='cod_ibge').merge(solo, on='cod_ibge')
    b['cidade'] = b['cod_ibge'].map(nomes)
    b = b.rename(columns={'colheita': 'safra'})
    b.to_csv(RES / f'{nome}.csv', index=False)
    print(nome, b.shape, 'safras', b['safra'].min(), '-', b['safra'].max(),
          '| NaN em NDVI_F3:', int(b['NDVImax_F3'].isna().sum()))
