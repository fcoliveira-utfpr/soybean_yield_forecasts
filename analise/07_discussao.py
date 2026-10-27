"""Análises de apoio à discussão.

  1. ENSO: ONI (NOAA CPC) x anomalia regional; ganho do ONI de ago-out como preditor antecipado (LOSO, Ridge)
  2. Tamanho efetivo da amostra pelo efeito de delineamento, a partir dos componentes de variância
  3. Métricas por escala (município, microrregião, região): R² e erro relativo
Saídas: dados/oni.csv, resultados/enso.csv, resultados/discussao_resumo.csv
"""
import sys
import urllib.request
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent))
val = import_module('02_validacao')
RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / 'resultados'
saida = {}

# ---------------------------------------------------------------- 1. ENSO
arq = RAIZ / 'dados/oni.csv'
if not arq.exists():
    txt = urllib.request.urlopen('https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt', timeout=60).read().decode()
    linhas = [l.split() for l in txt.splitlines()[1:] if l.strip()]
    pd.DataFrame(linhas, columns=['seas', 'ano', 'total', 'oni']).to_csv(arq, index=False)
oni = pd.read_csv(arq)
# safra H: ASO (ago-out) e SON do ano H-1 estão disponíveis no plantio; NDJ (nov-jan) cobre a fase F2
pv = oni.pivot_table(index='ano', columns='seas', values='oni')
enso = pd.DataFrame({'safra': pv.index + 1, 'ONI_ASO': pv['ASO'].values, 'ONI_SON': pv['SON'].values,
                     'ONI_NDJ': pv['NDJ'].values})
o = pd.read_csv(RES / 'oof_safra_safra_fora_Ridge.csv')
reg = o.groupby('safra').agg(prod=('prod', 'mean'), tend=('tend', 'mean'), pred=('pred', 'mean'))
reg['anom_rel'] = 100 * (reg['prod'] / reg['tend'] - 1)
reg = reg.reset_index().merge(enso, on='safra')
for c in ['ONI_ASO', 'ONI_SON', 'ONI_NDJ']:
    saida[f'corr_anom_{c}'] = reg['anom_rel'].corr(reg[c])
reg['fase_enso'] = np.select([reg['ONI_SON'] <= -0.5, reg['ONI_SON'] >= 0.5], ['La Niña', 'El Niño'], 'Neutral')
g = reg.groupby('fase_enso')['anom_rel'].agg(['count', 'mean', 'min'])
for f in g.index:
    saida[f'n_{f}'] = int(g.loc[f, 'count']); saida[f'anom_media_{f}'] = g.loc[f, 'mean']
saida['quebras_em_la_nina'] = ', '.join(str(s) for s in reg.loc[(reg['anom_rel'] < -15) & (reg['fase_enso'] == 'La Niña'), 'safra'])
saida['quebras_total'] = ', '.join(f"{s} ({f})" for s, f in reg.loc[reg['anom_rel'] < -15, ['safra', 'fase_enso']].values)
reg.to_csv(RES / 'enso.csv', index=False)

# ganho do ONI como preditor antecipado (fase F1 + ONI de ago-out) e na safra completa
b = val.base.merge(enso[['safra', 'ONI_ASO']], on='safra')
ESTAT = ['argila', 'areia', 'latitude', 'longitude', 'altitude_m', 'area_soja_ha']
f1 = ESTAT + [c for c in val.FEATS if c.endswith('_F1')]
for nome, fs in [('F1', f1), ('F1+ONI', f1 + ['ONI_ASO']), ('F1-F4', val.FEATS), ('F1-F4+ONI', val.FEATS + ['ONI_ASO'])]:
    oo = val.oof('safra_fora', 'Ridge', d=b, feats=fs)
    saida[f'skill_{nome}'] = val.metricas(oo)['skill_tend']
    print(nome, round(saida[f'skill_{nome}'], 3), flush=True)

# ---------------------------------------------------------------- 2. tamanho efetivo da amostra
comp = pd.read_csv(RES / 'componentes_variancia.csv', index_col=0)
n_mun, n_reg = val.base['cod_ibge'].nunique(), len(val.base)
for lin in comp.index:
    icc = comp.loc[lin, 'frac_safra']
    de = 1 + (n_mun - 1) * icc
    saida[f'icc_{lin}'] = icc; saida[f'efeito_delineamento_{lin}'] = de; saida[f'n_efetivo_{lin}'] = n_reg / de

# ---------------------------------------------------------------- 3. métricas por escala
saida['rRMSE_municipal'] = 100 * np.sqrt(np.mean((o['prod'] - o['pred']) ** 2)) / o['prod'].mean()
m = o.groupby(['microrregiao', 'safra'])[['prod', 'pred']].mean()
saida['R2_microrregiao'] = r2_score(m['prod'], m['pred'])
saida['rRMSE_microrregiao'] = 100 * np.sqrt(np.mean((m['prod'] - m['pred']) ** 2)) / m['prod'].mean()
r = o.groupby('safra')[['prod', 'pred']].mean()
saida['R2_regional'] = r2_score(r['prod'], r['pred'])
saida['rRMSE_regional'] = 100 * np.sqrt(np.mean((r['prod'] - r['pred']) ** 2)) / r['prod'].mean()

pd.Series(saida).to_csv(RES / 'discussao_resumo.csv', header=['valor'])
print(pd.Series(saida).to_string())
print(g.round(1))
