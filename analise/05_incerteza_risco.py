"""Estrutura estocástica do erro, intervalos de previsão e risco de quebra (frentes C1-C3 do planejamento).

C1  modelo misto com efeitos cruzados de safra e município sobre os resíduos fora-da-amostra (LOSO) e
    sobre as anomalias observadas; variograma dos resíduos dentro da safra; intervalos gaussianos a partir
    dos componentes de variância estimados só nas safras de treino.
C2  regressão quantílica conformalizada (CQR, LightGBM quantílico) calibrada nas outras safras.
C3  probabilidade de quebra (produtividade < 70, 80 e 90% da tendência) e diagrama de confiabilidade.
Saídas: resultados/componentes_variancia.csv, variograma.csv, intervalos_cobertura.csv,
        intervalos_por_safra.csv, risco_limiares.csv, confiabilidade.csv; figuras fig08-fig09.
"""
import sys
import warnings
from importlib import import_module
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
val = import_module('02_validacao')
warnings.filterwarnings('ignore')
RAIZ = Path(__file__).resolve().parents[1]
RES, FIG = RAIZ / 'resultados', RAIZ / 'figuras'
VERDE, VERDE_CL, MARROM, LARANJA, CINZA, TINTA = '#1b7837', '#74c476', '#8c510a', '#eb6834', '#8a8984', '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'axes.axisbelow': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6,
                     'axes.edgecolor': CINZA, 'axes.labelcolor': TINTA, 'xtick.color': TINTA,
                     'ytick.color': TINTA, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
ALFA, Z = 0.10, norm.ppf(0.95)


def salva(fig, nome):
    fig.savefig(FIG / f'{nome}.png')
    fig.savefig(RAIZ / 'artigo/figuras' / f'{nome}.png')
    plt.close(fig)


resumo = pd.read_csv(RES / 'validacao_resumo.csv')
r = resumo[(resumo['base'] == 'safra') & (resumo['esquema'] == 'safra_fora') & (resumo['modelo'] != 'Tendência')]
MELHOR = r.sort_values('RMSE').iloc[0]['modelo']
o = pd.read_csv(RES / f'oof_safra_safra_fora_{MELHOR}.csv')
o['res'] = o['prod'] - o['pred']
o['anom'] = o['prod'] - o['tend']
SAFRAS = sorted(o['safra'].unique())
saida = {'modelo': MELHOR}


# ---------------------------------------------------------------- C1a componentes de variância
def componentes(d, col):
    m = smf.mixedlm(f'{col} ~ 1', d.assign(g=1), groups='g',
                    vc_formula={'safra': '0 + C(safra)', 'mun': '0 + C(cod_ibge)'}).fit(reml=True)
    vc = dict(zip(m.model.exog_vc.names, m.vcomp))
    return {'safra': vc['safra'], 'municipio': vc['mun'], 'residual': m.scale}


comp = pd.DataFrame({'anomalia_observada': componentes(o, 'anom'), 'residuo_modelo': componentes(o, 'res')}).T
comp['total'] = comp.sum(axis=1)
for c in ['safra', 'municipio', 'residual']:
    comp[f'frac_{c}'] = comp[c] / comp['total']
comp.to_csv(RES / 'componentes_variancia.csv')
print(comp.round(3).to_string())

# ---------------------------------------------------------------- C1b variograma dentro da safra
coord = val.base.drop_duplicates('cod_ibge').set_index('cod_ibge')[['latitude', 'longitude']]
xk = coord['longitude'].values * 111.32 * np.cos(np.deg2rad(coord['latitude'].mean()))
yk = coord['latitude'].values * 110.57
D = np.hypot(xk[:, None] - xk, yk[:, None] - yk)
iu = np.triu_indices(len(coord), 1)
dist = D[iu]
semi = []
for s, g in o.groupby('safra'):
    z = g.set_index('cod_ibge')['res'].reindex(coord.index).values
    z = z - np.nanmean(z)                           # remove o efeito comum da safra
    semi.append(0.5 * (z[:, None] - z)[iu] ** 2)
semi = np.nanmean(np.vstack(semi), axis=0)
bins = np.arange(0, dist.max() + 15, 15)
cls = np.digitize(dist, bins)
vg = pd.DataFrame({'h': dist, 'g': semi, 'c': cls}).groupby('c').agg(h=('h', 'mean'), gamma=('g', 'mean'),
                                                                      n=('g', 'size'))
vg = vg[vg['n'] >= 20]


def expo(h, c0, c, a):
    return c0 + c * (1 - np.exp(-h / a))


(c0, c1, a), _ = curve_fit(expo, vg['h'], vg['gamma'], p0=[vg['gamma'].min(), vg['gamma'].max(), 30],
                           bounds=([0, 0, 1], [np.inf, np.inf, 500]))
vg.to_csv(RES / 'variograma.csv', index=False)
saida.update(variograma_pepita=c0, variograma_patamar=c0 + c1, variograma_alcance_pratico_km=3 * a,
             variograma_frac_pepita=c0 / (c0 + c1))

# ---------------------------------------------------------------- C1c / C2 intervalos
X, F = val.base, val.FEATS
o = o.merge(X[['cod_ibge', 'safra'] + F], on=['cod_ibge', 'safra'], how='left')
lin, reg_lin = [], []
for s in SAFRAS:
    te, outros = o[o['safra'] == s], o[o['safra'] != s]
    n = len(te)
    # (i) conformal i.i.d.: |resíduo| das outras safras
    q_iid = np.quantile(np.abs(outros['res']), 1 - ALFA)
    # (ii) modelo misto: componentes estimados sem a safra de teste
    vc = componentes(outros, 'res')
    sd_mun = np.sqrt(vc['safra'] + vc['municipio'] + vc['residual'])
    sd_reg = np.sqrt(vc['safra'] + (vc['municipio'] + vc['residual']) / n)
    # (iii) CQR: quantis da anomalia com LightGBM treinado nas outras safras
    tr = X[X['safra'] != s].copy()
    tr['tend'] = val.tendencia(tr, tr)
    qs = {}
    for nome, al in [('lo', ALFA / 2), ('hi', 1 - ALFA / 2)]:
        m = lgb.LGBMRegressor(objective='quantile', alpha=al, n_estimators=400, learning_rate=0.03,
                              num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.6, verbose=-1, n_jobs=4)
        m.fit(tr[F], tr['prod'] - tr['tend'])
        qs[nome] = m.predict(te[F])
    lin.append(pd.DataFrame({'safra': s, 'cod_ibge': te['cod_ibge'].values, 'prod': te['prod'].values,
                             'pred': te['pred'].values, 'tend': te['tend'].values,
                             'q_iid': q_iid, 'sd_misto': sd_mun,
                             'cqr_lo': te['tend'].values + qs['lo'], 'cqr_hi': te['tend'].values + qs['hi']}))
    reg_lin.append({'safra': s, 'prod': te['prod'].mean(), 'pred': te['pred'].mean(), 'n': n,
                    'q_iid_reg': q_iid / np.sqrt(n), 'sd_misto_reg': sd_reg})
iv = pd.concat(lin, ignore_index=True)
# calibração conformal do CQR com os escores das outras safras (cross-conformal por safra)
iv['E'] = np.maximum(iv['cqr_lo'] - iv['prod'], iv['prod'] - iv['cqr_hi'])
iv['Q_cqr'] = [np.quantile(iv.loc[iv['safra'] != s, 'E'], min(1, (1 - ALFA) * (1 + 1 / (iv['safra'] != s).sum())))
               for s in iv['safra']]
MET = {
    'Conformal i.i.d.': (iv['pred'] - iv['q_iid'], iv['pred'] + iv['q_iid']),
    'Mixed model': (iv['pred'] - Z * iv['sd_misto'], iv['pred'] + Z * iv['sd_misto']),
    'CQR': (iv['cqr_lo'] - iv['Q_cqr'], iv['cqr_hi'] + iv['Q_cqr']),
}
cob, por_safra = [], []
for nome, (lo, hi) in MET.items():
    dentro = (iv['prod'] >= lo) & (iv['prod'] <= hi)
    cs = dentro.groupby(iv['safra']).mean()
    cob.append({'nivel': 'municipal', 'metodo': nome, 'cobertura': dentro.mean(), 'largura': (hi - lo).mean(),
                'cobertura_min_safra': cs.min(), 'safra_min': int(cs.idxmin()),
                'safras_abaixo_80': int((cs < 0.8).sum())})
    por_safra.append(cs.rename(nome))
reg = pd.DataFrame(reg_lin).set_index('safra')
reg['res'] = reg['prod'] - reg['pred']
reg['q_bloco'] = [np.quantile(np.abs(reg.drop(s)['res']), 1 - ALFA) for s in reg.index]
for nome, w in [('Conformal i.i.d.', reg['q_iid_reg']), ('Mixed model', Z * reg['sd_misto_reg']),
                ('Season-block conformal', reg['q_bloco'])]:
    cob.append({'nivel': 'regional', 'metodo': nome, 'cobertura': (np.abs(reg['res']) <= w).mean(),
                'largura': 2 * w.mean()})
cob = pd.DataFrame(cob)
cob.to_csv(RES / 'intervalos_cobertura.csv', index=False)
pd.concat(por_safra, axis=1).to_csv(RES / 'intervalos_por_safra.csv')
print(cob.round(3).to_string())

# figura: cobertura municipal por safra e componentes de variância
ps = pd.concat(por_safra, axis=1)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={'width_ratios': [2.6, 1], 'wspace': 0.3})
w = 0.27
for j, (nome, cor) in enumerate(zip(ps.columns, [CINZA, VERDE_CL, VERDE])):
    ax[0].bar(ps.index + (j - 1) * w, 100 * ps[nome], width=w, color=cor, label=nome)
ax[0].axhline(90, color=LARANJA, ls='--', lw=1.2, label='Nominal (90%)')
ax[0].set_ylabel('Municipal coverage (%)'); ax[0].set_xlabel('Harvest year')
ax[0].legend(frameon=False, ncol=4, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.2))
ax[0].set_title('(a) Coverage of 90% intervals by season', loc='left', fontsize=9)
lab = {'anomalia_observada': 'Observed\nanomaly', 'residuo_modelo': f'{MELHOR}\nresidual'}
base_ = np.zeros(2)
for c, cor, nm in [('safra', VERDE, 'Season'), ('municipio', VERDE_CL, 'Municipality'),
                   ('residual', '#d9d8d2', 'Residual')]:
    v = comp[c].values / 1e3
    ax[1].bar([lab[i] for i in comp.index], v, bottom=base_, color=cor, width=0.55, label=nm,
              edgecolor='white', linewidth=1)
    base_ += v
ax[1].set_ylabel('Variance (10³ kg² ha⁻²)'); ax[1].legend(frameon=False, fontsize=7.5)
ax[1].set_title('(b) Variance components', loc='left', fontsize=9)
salva(fig, 'fig08_intervalos_variancia')

# ---------------------------------------------------------------- C3 risco de quebra
ris, conf = [], []
for lim in [0.7, 0.8, 0.9]:
    ev = (o['prod'] < lim * o['tend']).astype(int)
    p, pc = np.zeros(len(o)), np.zeros(len(o))
    for s in SAFRAS:
        m, rr = (o['safra'] == s).values, o.loc[o['safra'] != s, 'res'].values
        p[m] = [(pr + rr < lim * t).mean() for pr, t in zip(o.loc[m, 'pred'], o.loc[m, 'tend'])]
        pc[m] = ev[~m].mean()
    bs, bc = np.mean((p - ev) ** 2), np.mean((pc - ev) ** 2)
    ris.append({'limiar': lim, 'freq': ev.mean(), 'brier': bs, 'brier_clim': bc, 'bss': 1 - bs / bc,
                'auc': roc_auc_score(ev, p)})
    cl = pd.cut(p, np.linspace(0, 1, 6), include_lowest=True)
    conf.append(pd.DataFrame({'limiar': lim, 'p': p, 'ev': ev}).groupby(cl, observed=True)
                .agg(p_media=('p', 'mean'), freq_obs=('ev', 'mean'), n=('ev', 'size')).reset_index(drop=True)
                .assign(limiar=lim))
ris = pd.DataFrame(ris); conf = pd.concat(conf)
ris.to_csv(RES / 'risco_limiares.csv', index=False); conf.to_csv(RES / 'confiabilidade.csv', index=False)
print(ris.round(3).to_string())

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={'wspace': 0.35})
for lim, cor in zip([0.7, 0.8, 0.9], [VERDE_CL, VERDE, MARROM]):
    c = conf[conf['limiar'] == lim]
    ax[0].plot(c['p_media'], c['freq_obs'], marker='o', color=cor, lw=2, ms=6, label=f'< {int(lim * 100)}% of trend')
ax[0].plot([0, 1], [0, 1], color=CINZA, ls='--', lw=1)
ax[0].set_xlabel('Forecast probability'); ax[0].set_ylabel('Observed frequency')
ax[0].set_title('(a) Reliability of shortfall forecasts', loc='left', fontsize=9)
ax[0].legend(frameon=False, fontsize=7.5)
x = np.arange(3)
ax[1].bar(x - 0.18, ris['bss'], width=0.36, color=VERDE, label='Brier skill score')
ax[1].bar(x + 0.18, ris['auc'], width=0.36, color=VERDE_CL, label='AUC')
ax[1].set_xticks(x); ax[1].set_xticklabels([f'< {int(l * 100)}%' for l in ris['limiar']])
ax[1].set_xlabel('Shortfall threshold (% of trend)'); ax[1].set_ylim(0, 1.18)
ax[1].set_title('(b) Skill by threshold', loc='left', fontsize=9)
ax[1].legend(frameon=False, fontsize=7.5, ncol=2, loc='upper center')
salva(fig, 'fig09_risco')

pd.Series(saida).to_csv(RES / 'resumo_incerteza.csv', header=['valor'])
print(pd.Series(saida).to_string())
