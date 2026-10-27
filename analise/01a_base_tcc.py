"""Base cidade x safra a partir dos 49 CSVs do TCC (usada só para comparar com a base do GEE).

Correções em relação aos notebooks do TCC:
  - agrega por ano_safra (e não por ano civil, que mistura duas safras);
  - clima deduplicado por decêndio (as linhas de MODIS/Landsat repetiam o clima)
    e calculado com todos os decêndios, inclusive os de NDVI interpolado;
  - NDVI/EVI calculados só com observações reais de sensor;
  - unidades: K -> °C, J/m² -> MJ/m², m -> mm.
Saídas: resultados/base_tcc_desalinhada.csv e resultados/base_tcc_corrigida.csv
"""
import glob
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
MUNIC = RAIZ / 'dados/municipios_pr.csv'
(RAIZ / 'resultados').mkdir(exist_ok=True)


def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode()
    return ''.join(ch for ch in s.upper() if ch.isalpha())


df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RAIZ / 'dados/tcc/*.csv')))], ignore_index=True)
df['cidade'] = df['cidade'].str.replace(r'PR\d{7}$', '', regex=True)
df = df.rename(columns={'ano_safra': 'safra', 'ordem_decendio_safra': 'dec'})
CHAVE = ['cidade', 'safra']

# ---------- clima: um registro por decêndio ----------
clima = df.sort_values('data').drop_duplicates(CHAVE + ['dec']).copy()
for c in ['Temp_PontoOrvalho_C', 'Temp_Solo_0a7cm_C', 'Temp_Max_C', 'Temp_Media_C', 'Temp_Min_C']:
    clima[c] = clima[c] - 273.15
for c in ['Radiacao_Solar_Liquida_Jm2', 'Radiacao_Solar_Global_Jm2']:
    clima[c] = clima[c] / 1e6
clima['ETp_mm'] = -clima['Evapo_Potencial_m'] * 1000
clima['Evap_dossel_mm'] = -clima['Evaporation_From_Top_Of_Canopy_Sum'] * 1000
clima['BH_mm'] = clima['Precipitacao_mm'] - clima['ETp_mm']
clima['dec_seco'] = (clima['Precipitacao_mm'] < 20).astype(float)
clima['dec_quente'] = (clima['Temp_Max_C'] > 32).astype(float)
clima.loc[clima['Temp_Max_C'].isna(), 'dec_quente'] = np.nan

AGG_CLIMA = {
    'Precipitacao_mm': 'sum', 'ETp_mm': 'sum', 'BH_mm': 'sum', 'dec_seco': 'sum', 'dec_quente': 'sum',
    'Temp_Media_C': 'mean', 'Temp_Max_C': 'max', 'Temp_Min_C': 'min', 'Temp_PontoOrvalho_C': 'mean',
    'Temp_Solo_0a7cm_C': 'mean', 'Radiacao_Solar_Global_Jm2': 'mean', 'Radiacao_Solar_Liquida_Jm2': 'mean',
    'Evap_dossel_mm': 'sum', 'IAF_VegBaixa_fracao': 'mean',
}
NOMES = {'Precipitacao_mm': 'P', 'ETp_mm': 'ETp', 'BH_mm': 'BH', 'dec_seco': 'nSeco', 'dec_quente': 'nQuente',
         'Temp_Media_C': 'Tmed', 'Temp_Max_C': 'Tmax', 'Temp_Min_C': 'Tmin', 'Temp_PontoOrvalho_C': 'Td',
         'Temp_Solo_0a7cm_C': 'Tsolo', 'Radiacao_Solar_Global_Jm2': 'Rg', 'Radiacao_Solar_Liquida_Jm2': 'Rn',
         'Evap_dossel_mm': 'Edossel', 'IAF_VegBaixa_fracao': 'IAF'}


def por_fase(d, agg, nomes):
    a = d.groupby(CHAVE + ['Fase']).agg(agg)
    # a soma de um grupo só com NaN vira 0; devolve NaN onde não há dado
    n = d.groupby(CHAVE + ['Fase'])[list(agg)].count()
    a = a.where(n > 0)
    a = a.rename(columns=nomes).unstack('Fase')
    a.columns = [f'{v}_F{f}' for v, f in a.columns]
    return a


X_clima = por_fase(clima, AGG_CLIMA, NOMES)

# ---------- índices de vegetação: só observações reais ----------
veg = df[df['sensor'] != 'Interpolado'].copy()
# controle de qualidade: o Landsat traz NDVI até -2 e EVI até -2e9 (denominador ~0, nuvem)
for v in ['NDVI', 'EVI']:
    fora = (veg[v] < 0) | (veg[v] > 1)
    print(f'{v}: {fora.sum()} observações fora de [0, 1] descartadas', veg.loc[fora, 'sensor'].value_counts().to_dict())
    veg.loc[fora, v] = np.nan
X_veg = por_fase(veg, {'NDVI': 'max', 'EVI': 'max'}, {'NDVI': 'NDVImax', 'EVI': 'EVImax'})
X_veg = X_veg.join(por_fase(veg, {'NDVI': 'mean', 'EVI': 'mean'}, {'NDVI': 'NDVImed', 'EVI': 'EVImed'}))
X_veg['NDVImax_safra'] = veg.groupby(CHAVE)['NDVI'].max()
X_veg['n_obs_sensor'] = veg.groupby(CHAVE).size()

# ---------- alvo e atributos estáticos ----------
base = df.groupby(CHAVE).agg(prod=('Produtividade_kg_ha', 'first'),
                             argila=('Argila_pct', 'first'), areia=('Areia_pct', 'first'))
base = base.join(X_clima).join(X_veg).reset_index()

mun = pd.read_csv(MUNIC)
mun['chave'] = mun['municipio'].map(norm)
base['chave'] = base['cidade'].map(norm).replace({'SERRANAPOLISDOIGUACU': 'SERRANOPOLISDOIGUACU'})
base = base.merge(mun[['chave', 'cod_ibge', 'municipio', 'latitude', 'longitude', 'altitude_m', 'microrregiao']],
                  on='chave', how='left').drop(columns='chave')
faltou = base.loc[base['cod_ibge'].isna(), 'cidade'].unique()
assert len(faltou) == 0, f'municípios sem código IBGE: {faltou}'

# Alinhamento: nos CSVs, ano_safra = s reúne clima/NDVI de out/s a mar/s+1, mas a produtividade
# atribuída é a do rótulo s do IBGE (colheita de s, i.e. safra s-1/s). Corrige-se indexando pela
# colheita: safra = s+1, produtividade = rótulo s+1. A base original fica salva para comparação.
# A produtividade vem da tabela IBGE 1612 baixada por 00_ibge.py (idêntica à dos CSVs no rótulo s).
base.to_csv(RAIZ / 'resultados/base_tcc_desalinhada.csv', index=False)
ibge = pd.read_csv(RAIZ / 'dados/ibge_1612_soja_pr.csv').set_index(['cod_ibge', 'ano_colheita'])['prod_kg_ha']
base['safra'] = base['safra'] + 1
base['prod'] = [ibge.get((int(c), s), np.nan) for c, s in zip(base['cod_ibge'], base['safra'])]
base = base.dropna(subset=['prod'])
base.to_csv(RAIZ / 'resultados/base_tcc_corrigida.csv', index=False)
print(base.shape, base['cidade'].nunique(), 'municípios, colheitas', base['safra'].min(), '-', base['safra'].max())
print(base['microrregiao'].value_counts().to_dict())
print(base[['prod', 'P_F3', 'Tmed_F3', 'NDVImax_safra']].describe().round(2).T)
