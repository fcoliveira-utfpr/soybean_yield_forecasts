"""Baixa a produtividade municipal de soja (IBGE/PAM, tabela 1612, variável 112) do Paraná
e confere o rótulo de ano usado nos CSVs do TCC.

Saída: dados/ibge_1612_soja_pr.csv (cod_ibge, ano_colheita, prod_kg_ha)
"""
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
URL = ('https://apisidra.ibge.gov.br/values/t/1612/n6/in%20n3%2041/v/112/p/2000-2025/c81/2713?formato=json')

with urllib.request.urlopen(URL, timeout=120) as r:
    linhas = json.load(r)[1:]
ibge = pd.DataFrame({'cod_ibge': [int(l['D1C']) for l in linhas],
                     'ano_colheita': [int(l['D3C']) for l in linhas],
                     'prod_kg_ha': pd.to_numeric([l['V'] for l in linhas], errors='coerce')})
ibge = ibge.dropna()
ibge.to_csv(RAIZ / 'dados/ibge_1612_soja_pr.csv', index=False)
print(ibge.shape, 'anos', ibge['ano_colheita'].min(), '-', ibge['ano_colheita'].max())

# conferência: produtividade dos CSVs do TCC (rótulo ano_safra) x IBGE no mesmo ano e no ano seguinte
des = pd.read_csv(RAIZ / 'resultados/base_tcc_desalinhada.csv')[['cod_ibge', 'safra', 'prod']]
for desloc in [0, 1]:
    m = des.assign(ano=des['safra'] + desloc).merge(ibge, left_on=['cod_ibge', 'ano'],
                                                    right_on=['cod_ibge', 'ano_colheita'])
    iguais = np.isclose(m['prod'], m['prod_kg_ha'], atol=1).mean()
    print(f'CSV ano_safra=s x IBGE ano s+{desloc}: {len(m)} pares, {100 * iguais:.1f}% idênticos')
