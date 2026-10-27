"""Análises sobre as predições fora-da-amostra (deixa uma safra fora) do melhor modelo.

  - série regional observada x tendência x predita, com intervalo por blocos de safra
  - decomposição do erro (efeito comum da safra x idiossincrático) e Moran's I por safra
  - métricas por município e mapas
  - intervalos conformais: ingênuo (i.i.d.) x por blocos de safra; cobertura
  - risco de quebra (produtividade < 80% da tendência): Brier e AUC
  - antecedência: desempenho usando só as fases 1, 1-2, 1-3, 1-4
  - SHAP agregado por fase e por família de variáveis
Saídas em resultados/ e figuras/.
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from sklearn.metrics import r2_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
val = import_module('02_validacao')

warnings.filterwarnings('ignore')
RAIZ = Path(__file__).resolve().parents[1]
RES, FIG = RAIZ / 'resultados', RAIZ / 'figuras'
FIG.mkdir(exist_ok=True)
GEO = RAIZ / 'dados/malha_municipios_pr.geojson'

VERDE, LARANJA, CINZA, TINTA = '#1b7837', '#eb6834', '#8a8984', '#52514e'
DIVERG = LinearSegmentedColormap.from_list('div', ['#8c510a', '#d8b365', '#f0efec', '#5aae61', '#00441b'])
SEQ = LinearSegmentedColormap.from_list('seq', ['#edf8e9', '#bae4b3', '#74c476', '#31a354', '#00441b'])
NOMES_EN = {'nSeco': 'Dry dekads', 'nQuente': 'Hot dekads', 'BH': 'Water balance', 'P': 'Precipitation',
            'ETp': 'Potential ET', 'Td': 'Dew point', 'Tsolo': 'Soil temperature', 'Tmed': 'Mean temperature',
            'Tmax': 'Max temperature', 'Tmin': 'Min temperature', 'Rg': 'Global radiation', 'Rn': 'Net radiation',
            'Edossel': 'Canopy evaporation', 'IAF': 'LAI (low vegetation)', 'NDVImax': 'NDVI max',
            'EVImax': 'EVI max', 'NDVImed': 'NDVI mean', 'EVImed': 'EVI mean', 'NDVImax_safra': 'Seasonal NDVI max',
            'n_obs_veg': 'Vegetation observations', 'EVImax_safra': 'Seasonal EVI max',
            'area_soja_ha': 'Soybean area', 'TmaxAbs': 'Absolute max temperature', 'dQuente': 'Hot days',
            'dSeco': 'Dry days', 'Usolo1': 'Soil water 0–7 cm', 'Usolo2': 'Soil water 7–28 cm', 'ET': 'Evapotranspiration', 'argila': 'Clay', 'areia': 'Sand', 'altitude_m': 'Altitude',
            'latitude': 'Latitude', 'longitude': 'Longitude'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': CINZA, 'axes.labelcolor': TINTA, 'xtick.color': TINTA,
                     'ytick.color': TINTA, 'axes.grid': True, 'axes.axisbelow': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})

def salva(fig, nome):
    """PNG para conferência e cópia para o manuscrito (artigo/figuras)."""
    fig.savefig(FIG / f'{nome}.png')
    fig.savefig(RAIZ / 'artigo/figuras' / f'{nome}.png')
    plt.close(fig)


resumo = pd.read_csv(RES / 'validacao_resumo.csv')
r = resumo[(resumo['base'] == 'safra') & (resumo['esquema'] == 'safra_fora') & (resumo['modelo'] != 'Tendência')]
MELHOR = r.sort_values('RMSE').iloc[0]['modelo']
print('melhor modelo (safra fora):', MELHOR)
o = pd.read_csv(RES / f'oof_safra_safra_fora_{MELHOR}.csv')
o['res'] = o['prod'] - o['pred']
o['anom'] = o['prod'] - o['tend']
o['anom_pred'] = o['pred'] - o['tend']
base = val.base
saida = {}

# ---------------------------------------------------------------- malha
geo = json.load(open(GEO))
POLIS = {}
for f in geo['features']:
    cod = int(f['properties']['codarea'])
    g = f['geometry']
    aneis = [p[0] for p in g['coordinates']] if g['type'] == 'MultiPolygon' else [g['coordinates'][0]]
    POLIS[cod] = [np.array(a) for a in aneis]
COD_OESTE = set(base['cod_ibge'].astype(int))
xy = np.vstack([a for c in COD_OESTE for a in POLIS[c]])
XL = (xy[:, 0].min() - 0.15, xy[:, 0].max() + 0.15)
YL = (xy[:, 1].min() - 0.15, xy[:, 1].max() + 0.15)


def mapa(ax, valores, cmap, vmin, vmax, titulo, rotulo):
    for cod, aneis in POLIS.items():
        for a in aneis:
            if cod in valores.index and not np.isnan(valores[cod]):
                cor = cmap((valores[cod] - vmin) / (vmax - vmin))
                ax.add_patch(Polygon(a, closed=True, facecolor=cor, edgecolor='white', linewidth=0.4))
            else:
                ax.add_patch(Polygon(a, closed=True, facecolor='#f4f3f0', edgecolor='#d8d7d2', linewidth=0.3))
    ax.set_xlim(XL); ax.set_ylim(YL)
    ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(YL))))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(titulo, fontsize=9, color=TINTA, loc='left')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    cb = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label(rotulo, color=TINTA); cb.outline.set_visible(False)


# ---------------------------------------------------------------- 1. área de estudo
est = base.groupby('cod_ibge')['prod'].agg(['mean', 'std'])
est['cv'] = 100 * est['std'] / est['mean']
inc = base.groupby('cod_ibge').apply(lambda g: np.polyfit(g['safra'], g['prod'], 1)[0])
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
mapa(ax[0], est['mean'], SEQ, est['mean'].min(), est['mean'].max(), '(a) Mean yield 2001–2023', 'kg ha⁻¹')
mapa(ax[1], inc, DIVERG, -np.abs(inc).max(), np.abs(inc).max(), '(b) Linear trend', 'kg ha⁻¹ yr⁻¹')
mapa(ax[2], est['cv'], SEQ, est['cv'].min(), est['cv'].max(), '(c) Coefficient of variation', '%')
salva(fig, 'fig01_area_estudo')

# ---------------------------------------------------------------- 2. comparação de validações
ordem = [('tese', 'aleatorio', 'Earlier study\nreplica\nrandom'),
         ('tcc_desalinhada', 'aleatorio', 'Earlier data\nmisaligned\nrandom'),
         ('tcc_desalinhada', 'safra_fora', 'Earlier data\nmisaligned\nseason out'),
         ('tcc_corrigida', 'safra_fora', 'Earlier data\naligned\nseason out'),
         ('safra', 'aleatorio', 'New data\n\nrandom'),
         ('safra', 'cidade_fora', 'New data\n\nmunicipality out'),
         ('safra', 'safra_fora', 'New data\n\nseason out'),
         ('safra', 'futuro', 'New data\n\nrolling origin')]
fig, ax = plt.subplots(figsize=(10, 3.6))
ja = set()
for i, (b, e, rot) in enumerate(ordem):
    for j, (mod, cor) in enumerate([('Tendência', CINZA), ('LGB', '#74c476'), ('Ridge', VERDE)]):
        lin = resumo[(resumo['base'] == b) & (resumo['esquema'] == e) & (resumo['modelo'] == mod)]
        if lin.empty:
            continue
        y = lin['R2'].iloc[0]
        dp = lin['R2_dp'].iloc[0] if 'R2_dp' in lin and not np.isnan(lin['R2_dp'].iloc[0]) else 0
        ax.errorbar(i + (j - 1) * 0.22, y, yerr=dp, fmt='o', ms=6, color=cor, capsize=2, lw=1.2,
                    label={'Tendência': 'Trend only', 'LGB': 'LightGBM', 'Ridge': 'Ridge'}[mod]
                    if mod not in ja else None)
        ja.add(mod)
ax.axhline(0, color=CINZA, lw=0.8)
ax.set_xticks(range(len(ordem))); ax.set_xticklabels([o_[2] for o_ in ordem], fontsize=7.5)
ax.set_ylabel('Test R² of yield'); ax.legend(frameon=False, loc='lower left')
salva(fig, 'fig02_validacoes')

# ---------------------------------------------------------------- 3. erro: efeito da safra x idiossincrático
ef_safra = o.groupby('safra')['res'].transform('mean')
frac_safra = np.var(ef_safra) / np.var(o['res'])
saida['fracao_var_residuo_comum_safra'] = frac_safra
anom_reg = o.groupby('safra')[['anom', 'anom_pred']].mean()
saida['R2_anomalia_regional'] = r2_score(anom_reg['anom'], anom_reg['anom_pred'])
saida['corr_anomalia_regional'] = anom_reg.corr().iloc[0, 1]
o['anom_intra'] = o['anom'] - o.groupby('safra')['anom'].transform('mean')
o['anom_pred_intra'] = o['anom_pred'] - o.groupby('safra')['anom_pred'].transform('mean')
saida['R2_anomalia_intra_safra'] = r2_score(o['anom_intra'], o['anom_pred_intra'])

# Moran's I dos resíduos por safra (k=6 vizinhos, linha-padronizada, 999 permutações)
coord = base.drop_duplicates('cod_ibge').set_index('cod_ibge')[['latitude', 'longitude']]
coord['x'] = coord['longitude'] * np.cos(np.deg2rad(coord['latitude'].mean()))
cods = coord.index.values
D = np.hypot(coord['x'].values[:, None] - coord['x'].values, coord['latitude'].values[:, None] - coord['latitude'].values)
W = np.zeros_like(D)
for i in range(len(D)):
    W[i, np.argsort(D[i])[1:7]] = 1 / 6
rng = np.random.default_rng(0)


def moran(z):
    z = z - z.mean()
    return (z @ W @ z) / (z @ z)


mor = []
for s, g in o.groupby('safra'):
    z = g.set_index('cod_ibge')['res'].reindex(cods).values
    I = moran(z)
    perm = np.array([moran(rng.permutation(z)) for _ in range(999)])
    mor.append({'safra': s, 'I': I, 'p': (np.sum(perm >= I) + 1) / 1000})
mor = pd.DataFrame(mor)
mor.to_csv(RES / 'moran_residuos.csv', index=False)
saida['moran_mediana'] = mor['I'].median()
saida['moran_n_signif_5pct'] = int((mor['p'] < 0.05).sum())

# ---------------------------------------------------------------- 4. conformal ingênuo x blocos
ALFA = 0.10
o['lo_iid'] = np.nan; o['hi_iid'] = np.nan
reg = o.groupby('safra').agg(prod=('prod', 'mean'), pred=('pred', 'mean'), tend=('tend', 'mean'))
reg['res'] = reg['prod'] - reg['pred']
n_mun = o.groupby('safra').size()
for s in reg.index:
    outros = o[o['safra'] != s]
    q = np.quantile(np.abs(outros['res']), 1 - ALFA)
    m = o['safra'] == s
    o.loc[m, 'lo_iid'] = o.loc[m, 'pred'] - q
    o.loc[m, 'hi_iid'] = o.loc[m, 'pred'] + q
    # regional: ingênuo (resíduos independentes => erro da média cai com 1/sqrt(n))
    reg.loc[s, 'q_iid'] = q / np.sqrt(n_mun[s])
    # por blocos: quantil do erro da média regional nas outras safras
    reg.loc[s, 'q_bloco'] = np.quantile(np.abs(reg.drop(s)['res']), 1 - ALFA)
o['cobre_iid'] = (o['prod'] >= o['lo_iid']) & (o['prod'] <= o['hi_iid'])
cob_safra = o.groupby('safra')['cobre_iid'].mean()
saida['cobertura_municipal_90'] = o['cobre_iid'].mean()
saida['cobertura_municipal_min_safra'] = cob_safra.min()
saida['safra_pior_cobertura'] = int(cob_safra.idxmin())
saida['cobertura_regional_iid'] = (np.abs(reg['res']) <= reg['q_iid']).mean()
saida['cobertura_regional_bloco'] = (np.abs(reg['res']) <= reg['q_bloco']).mean()
saida['largura_regional_iid'] = reg['q_iid'].mean() * 2
saida['largura_regional_bloco'] = reg['q_bloco'].mean() * 2
reg.to_csv(RES / 'serie_regional.csv')
cob_safra.rename('cobertura').to_csv(RES / 'cobertura_por_safra.csv')

fig, ax = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
ax[0].fill_between(reg.index, reg['pred'] - reg['q_bloco'], reg['pred'] + reg['q_bloco'], color=VERDE, alpha=0.15,
                   lw=0, label='90% interval, season blocks')
ax[0].fill_between(reg.index, reg['pred'] - reg['q_iid'], reg['pred'] + reg['q_iid'], color=VERDE, alpha=0.35,
                   lw=0, label='90% interval, naive (i.i.d.)')
ax[0].plot(reg.index, reg['tend'], color=CINZA, lw=1.5, ls='--', label='Trend only')
ax[0].plot(reg.index, reg['pred'], color=VERDE, lw=2, marker='o', ms=4, label=f'{MELHOR} (season out)')
ax[0].plot(reg.index, reg['prod'], color=TINTA, lw=2, marker='s', ms=4, label='Observed (IBGE)')
ax[0].set_ylabel('Regional mean\nyield (kg ha⁻¹)')
ax[0].legend(frameon=False, ncol=2, fontsize=7.5, loc='lower left')
ax[1].bar(cob_safra.index, 100 * cob_safra.values, color=VERDE, width=0.7)
ax[1].axhline(90, color=LARANJA, lw=1.2, ls='--')
ax[1].set_ylabel('Municipal\ncoverage (%)'); ax[1].set_xlabel('Harvest year')
salva(fig, 'fig03_serie_regional_incerteza')

# ---------------------------------------------------------------- 5. risco de quebra
LIM = 0.8
o['quebra'] = (o['prod'] < LIM * o['tend']).astype(int)
p_quebra, p_clim = [], []
for s, g in o.groupby('safra'):
    outros = o[o['safra'] != s]
    rr = outros['res'].values
    p_quebra.append(pd.Series([(pr + rr < LIM * t).mean() for pr, t in zip(g['pred'], g['tend'])], index=g.index))
    p_clim.append(pd.Series(outros['quebra'].mean(), index=g.index))
o['p_quebra'] = pd.concat(p_quebra); o['p_clim'] = pd.concat(p_clim)
bs = np.mean((o['p_quebra'] - o['quebra']) ** 2)
bs_c = np.mean((o['p_clim'] - o['quebra']) ** 2)
saida.update(freq_quebra=o['quebra'].mean(), brier=bs, brier_clim=bs_c, brier_skill=1 - bs / bs_c,
             auc_quebra=roc_auc_score(o['quebra'], o['p_quebra']))
safras_q = o.groupby('safra')['quebra'].mean()
saida['safras_com_quebra_generalizada'] = safras_q[safras_q > 0.5].index.tolist()

# ---------------------------------------------------------------- 6. métricas e mapas por município
pm = o.groupby('cod_ibge').apply(lambda g: pd.Series({
    'municipio': g['municipio'].iloc[0], 'microrregiao': g['microrregiao'].iloc[0],
    'MAE': np.mean(np.abs(g['res'])), 'RMSE': np.sqrt(np.mean(g['res'] ** 2)),
    'skill_tend': 1 - np.mean(g['res'] ** 2) / np.mean(g['anom'] ** 2),
    'corr_anom': np.corrcoef(g['anom'], g['anom_pred'])[0, 1], 'auc_quebra':
    roc_auc_score(g['quebra'], g['p_quebra']) if g['quebra'].nunique() == 2 else np.nan}))
pm.to_csv(RES / 'metricas_municipio.csv')
saida['skill_mun_mediana'] = pm['skill_tend'].median()
saida['n_mun_skill_pos'] = int((pm['skill_tend'] > 0).sum())
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
mapa(ax[0], pm['MAE'], SEQ, pm['MAE'].min(), pm['MAE'].max(), '(a) MAE', 'kg ha⁻¹')
lim = np.nanmax(np.abs(pm['skill_tend']))
mapa(ax[1], pm['skill_tend'], DIVERG, -lim, lim, '(b) Skill over the trend', '1 − MSE/MSE$_{tend}$')
mapa(ax[2], pm['corr_anom'], DIVERG, -1, 1, '(c) Correlation of anomalies', 'r')
salva(fig, 'fig04_mapas_desempenho')

# mapas de anomalia relativa observada x predita nas safras de quebra
o['anom_rel'] = 100 * o['anom'] / o['tend']
o['anom_pred_rel'] = 100 * o['anom_pred'] / o['tend']
secas = saida['safras_com_quebra_generalizada']
fig, ax = plt.subplots(2, len(secas), figsize=(3 * len(secas), 5.2))
for j, s in enumerate(secas):
    g = o[o['safra'] == s].set_index('cod_ibge')
    mapa(ax[0, j], g['anom_rel'], DIVERG, -80, 80, f'{s} observed', '% of trend')
    mapa(ax[1, j], g['anom_pred_rel'], DIVERG, -80, 80, f'{s} predicted', '% of trend')
salva(fig, 'fig05_mapas_quebras')

# ---------------------------------------------------------------- 7. antecedência (fases disponíveis)
ESTAT = ['argila', 'areia', 'latitude', 'longitude', 'altitude_m', 'area_soja_ha']
if '--rapido' in sys.argv:                 # só refaz figuras: reaproveita a antecedência já calculada
    ante = pd.read_csv(RES / 'antecedencia.csv')
else:
    ante = []
    for k in [1, 2, 3, 4]:
        arq = RES / f'oof_antecedencia_F{k}.csv'      # retomada: fase já calculada não é refeita
        if arq.exists():
            oo = pd.read_csv(arq)
        else:
            fs = ESTAT + [c for c in val.FEATS if any(c.endswith(f'_F{f}') for f in range(1, k + 1))]
            if k == 4:
                fs += ['NDVImax_safra', 'EVImax_safra', 'n_obs_veg']
            oo = val.oof('safra_fora', MELHOR, feats=fs)
            oo.to_csv(arq, index=False)
        ante.append({'fases': f'F1–F{k}', **val.metricas(oo)})
    ante = pd.DataFrame(ante)
    ante.to_csv(RES / 'antecedencia.csv', index=False)
fig, ax = plt.subplots(figsize=(4.5, 3))
ax.plot(ante['fases'], ante['skill_tend'], color=VERDE, marker='o', lw=2, ms=6)
ax.axhline(0, color=CINZA, lw=0.8)
ax.set_ylabel('Skill over the trend'); ax.set_xlabel('Phenological phases available')
salva(fig, 'fig06_antecedencia')

# ---------------------------------------------------------------- 8. SHAP por fase e família
d = base.copy()
d['tend'] = val.tendencia(d, d)
m = val.ajusta(MELHOR, d[val.FEATS], d['prod'] - d['tend'], d['safra'])
est_ = m[-1] if hasattr(m, 'steps') else m
Xs = d[val.FEATS] if not hasattr(m, 'steps') else pd.DataFrame(m[:-1].transform(d[val.FEATS]), columns=val.FEATS)
if MELHOR == 'Ridge':                          # explicador linear sobre os preditores padronizados
    sv = np.abs(shap.LinearExplainer(est_, Xs).shap_values(Xs)).mean(0)
else:
    sv = np.abs(shap.TreeExplainer(est_).shap_values(Xs)).mean(0)
imp = pd.DataFrame({'var': val.FEATS, 'shap': sv})
imp['fase'] = imp['var'].str.extract(r'_F(\d)$')[0].fillna('static')
imp['familia'] = imp['var'].str.replace(r'_F\d$', '', regex=True)
imp.sort_values('shap', ascending=False).to_csv(RES / 'shap_variaveis.csv', index=False)
pf = imp.groupby('fase')['shap'].sum()
pf.index = [f'F{i}' if i.isdigit() else 'Season/\nstatic' for i in pf.index]
fam = imp.groupby('familia')['shap'].sum().sort_values().tail(12)
fam.index = fam.index.map(lambda v: NOMES_EN.get(v, v))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={'wspace': 0.65})
ax[0].bar(pf.index, pf.values, color=VERDE, width=0.6)
ax[0].set_ylabel('Σ |SHAP| (kg ha⁻¹)'); ax[0].set_title('(a) By phase', loc='left', fontsize=9)
ax[1].barh(fam.index, fam.values, color=VERDE, height=0.6)
ax[1].set_xlabel('Σ |SHAP| (kg ha⁻¹)'); ax[1].set_title('(b) By variable', loc='left', fontsize=9)
salva(fig, 'fig07_shap')

pd.Series(saida).to_csv(RES / 'resumo_analises.csv', header=['valor'])
print(pd.Series(saida).to_string())
print(ante.round(3).to_string())
print(pm.sort_values('skill_tend').round(2).to_string())
print(mor.round(3).to_string())
