"""Esquemas de validação, modelos com hiperparâmetros aninhados, forma da tendência e testes por blocos de safra.

Esquemas:
  aleatorio    : split 70/30 (20 sementes), tendência com todas as safras (como no TCC); hiperparâmetros fixos
  cidade_fora  : deixa um município fora; tendência = média regional da safra nos municípios de treino
  safra_fora   : deixa uma safra inteira fora (LOSO); tendência só com as safras de treino  <- principal
  futuro       : origem móvel, treina só com safras anteriores (a partir de 2008)
Nos esquemas não aleatórios os hiperparâmetros são escolhidos em cada dobra por busca em grade com
validação interna agrupada por safra (GroupKFold, 5 grupos).
Bases: tese (réplica da engenharia do TCC), tcc_desalinhada, tcc_corrigida, safra (GEE, só MODIS), safra_fusao.
Saídas: resultados/validacao_resumo.csv, resultados/oof_<base>_<esquema>_<modelo>.csv,
        resultados/tendencia_formas.csv, resultados/bootstrap_diferencas.csv
"""
import glob
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / 'resultados'
ID = ['cidade', 'safra', 'cod_ibge', 'municipio', 'microrregiao', 'prod']


def carrega(nome):
    d = pd.read_csv(RES / f'base_{nome}.csv')
    return d, [c for c in d.columns if c not in ID]


base, FEATS = carrega('safra')

# ---------------------------------------------------------------- modelos e grades
FIXOS = {
    'LGB': lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=15, min_child_samples=20,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.6, verbose=-1, n_jobs=4),
    'XGB': lambda: xgb.XGBRegressor(n_estimators=500, learning_rate=0.02, max_depth=3, subsample=0.8,
                                    colsample_bytree=0.6, n_jobs=4),
    'RF': lambda: make_pipeline(SimpleImputer(), RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                                                       n_jobs=4, random_state=0)),
    'Ridge': lambda: make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10)),
}
GRADES = {
    'LGB': {'num_leaves': [7, 15, 31], 'learning_rate': [0.01, 0.03], 'min_child_samples': [10, 30]},
    'XGB': {'max_depth': [2, 3, 5], 'learning_rate': [0.01, 0.03]},
    'RF': {'randomforestregressor__min_samples_leaf': [1, 5, 15],
           'randomforestregressor__max_features': [0.33, 1.0]},
    'Ridge': {'ridge__alpha': list(np.logspace(-1, 4, 11))},
}
ESCOLHAS = []


def ajusta(nome, X, y, grupos, aninhado=True):
    if not aninhado:
        return FIXOS[nome]().fit(X, y)
    gs = GridSearchCV(FIXOS[nome](), GRADES[nome], cv=GroupKFold(n_splits=5),
                      scoring='neg_mean_squared_error', n_jobs=1, refit=True)
    gs.fit(X, y, groups=grupos)
    ESCOLHAS.append({'modelo': nome, **gs.best_params_})
    return gs.best_estimator_


# ---------------------------------------------------------------- tendência
def tendencia(fit, apl, forma='linear'):
    """Tendência por município ajustada em `fit` e avaliada em `apl`."""
    if forma == 'regional':                 # inclinação comum, intercepto municipal
        mu = fit.groupby('cidade')['prod'].mean()
        t0 = fit['safra'].mean()
        b = np.polyfit(fit['safra'] - t0, fit['prod'] - fit['cidade'].map(mu), 1)[0]
        return apl['cidade'].map(mu) + b * (apl['safra'] - t0)
    grau = {'linear': 1, 'quadratica': 2}[forma]
    out = pd.Series(np.nan, index=apl.index)
    for c, g in apl.groupby('cidade'):
        f = fit[fit['cidade'] == c]
        coef = np.polyfit(f['safra'] - 2010, f['prod'], grau)
        out[g.index] = np.polyval(coef, g['safra'] - 2010)
    return out


def metricas(d):
    y, p, t = d['prod'], d['pred'], d['tend']
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p), RMSE=np.sqrt(np.mean((y - p) ** 2)),
                R2_anom=r2_score(y - t, p - t), skill_tend=1 - np.mean((y - p) ** 2) / np.mean((y - t) ** 2))


# ---------------------------------------------------------------- esquemas
def oof(esquema, nome, d=None, feats=None, forma='linear', aninhado=True):
    d = base if d is None else d
    feats = FEATS if feats is None else feats
    if esquema == 'safra_fora':
        grupos = [(d['safra'] != s, d['safra'] == s) for s in sorted(d['safra'].unique())]
    elif esquema == 'futuro':
        grupos = [(d['safra'] < s, d['safra'] == s) for s in range(2008, d['safra'].max() + 1)]
    elif esquema == 'cidade_fora':
        grupos = [(d['cidade'] != c, d['cidade'] == c) for c in sorted(d['cidade'].unique())]
    partes = []
    for mtr, mte in grupos:
        tr, te = d[mtr].copy(), d[mte].copy()
        if esquema == 'cidade_fora':
            reg = tr.groupby('safra')['prod'].mean()
            tr['tend'], te['tend'] = tr['safra'].map(reg), te['safra'].map(reg)
        else:
            tr['tend'], te['tend'] = tendencia(tr, tr, forma), tendencia(tr, te, forma)
        if nome == 'Tendência':
            te['pred'] = te['tend']
        else:
            m = ajusta(nome, tr[feats], tr['prod'] - tr['tend'], tr['safra'], aninhado)
            te['pred'] = te['tend'] + m.predict(te[feats])
        partes.append(te[[c for c in ID if c in te] + ['tend', 'pred']])
    return pd.concat(partes)


def aleatorio(nome, d, feats, sementes=20):
    d = d.copy()
    d['tend'] = tendencia(d, d)
    linhas = []
    for s in range(sementes):
        tr, te = train_test_split(d, test_size=0.3, random_state=s)
        te = te.copy()
        te['pred'] = te['tend'] if nome == 'Tendência' else \
            te['tend'] + FIXOS[nome]().fit(tr[feats], tr['prod'] - tr['tend']).predict(te[feats])
        linhas.append(metricas(te))
    r = pd.DataFrame(linhas).agg(['mean', 'std'])
    return {**r.loc['mean'].to_dict(), **{f'{k}_dp': v for k, v in r.loc['std'].items()}}


def base_tese():
    """Réplica da engenharia de atributos do TCC (ano civil, só linhas reais, unidades originais)."""
    df = pd.concat([pd.read_csv(f) for f in glob.glob(str(RAIZ / 'dados/tcc/*.csv'))], ignore_index=True)
    df['cidade'] = df['cidade'].str.replace(r'PR\d{7}$', '', regex=True)
    df = df[df['sensor'] != 'Interpolado']
    agg = {'NDVI': ['mean', 'max', 'std'], 'EVI': ['mean', 'max', 'std'], 'Precipitacao_mm': ['sum', 'mean'],
           'Radiacao_Solar_Global_Jm2': ['mean'], 'Radiacao_Solar_Liquida_Jm2': ['mean'],
           'Temp_Max_C': ['mean', 'std'], 'Temp_Media_C': ['mean', 'std'], 'Temp_Min_C': ['mean', 'std'],
           'Temp_PontoOrvalho_C': ['mean', 'std'], 'Evaporation_From_Top_Of_Canopy_Sum': ['mean', 'std'],
           'Evapo_Potencial_m': ['mean', 'std'], 'Temp_Solo_0a7cm_C': ['mean', 'std'],
           'Pressao_Superficie_Pa': ['mean', 'std'], 'IAF_VegBaixa_fracao': ['mean', 'std'], 'dia_safra': ['max']}
    a = df.groupby(['ano', 'cidade', 'Fase']).agg(agg)
    a.columns = [f'{c}_{s}' for c, s in a.columns]
    a = a.unstack('Fase')
    a.columns = [f'{c}_{f}' for c, f in a.columns]
    y = df.groupby(['ano', 'cidade'])['Produtividade_kg_ha'].first().rename('prod')
    d = a.join(y).reset_index().rename(columns={'ano': 'safra'}).fillna(0)
    return d, [c for c in d.columns if c not in ('safra', 'cidade', 'prod')]


def bootstrap_dif(oa, ob, n=2000, seed=0):
    """Diferença de MSE (A - B) e IC 95% reamostrando safras inteiras; p = fração das réplicas >= 0."""
    a = oa.assign(e=(oa['prod'] - oa['pred']) ** 2).groupby('safra')['e'].agg(['sum', 'count'])
    b = ob.assign(e=(ob['prod'] - ob['pred']) ** 2).groupby('safra')['e'].sum()
    s = a.index.values
    rng = np.random.default_rng(seed)
    difs = []
    for _ in range(n):
        k = rng.choice(s, len(s), replace=True)
        difs.append((a.loc[k, 'sum'].sum() - b.loc[k].sum()) / a.loc[k, 'count'].sum())
    obs = (a['sum'].sum() - b.sum()) / a['count'].sum()
    lo, hi = np.percentile(difs, [2.5, 97.5])
    return obs, lo, hi, float(np.mean(np.array(difs) >= 0))


if __name__ == '__main__':
    import json
    MODELOS = ['Tendência', 'Ridge', 'RF', 'XGB', 'LGB']
    CACHE = RES / 'validacao_cache.jsonl'          # retomada: cada etapa concluída fica gravada aqui
    feito = {}
    if CACHE.exists():
        for l in CACHE.read_text().splitlines():
            e = json.loads(l); feito[(e['base'], e['esquema'], e['modelo'])] = e
    resumo = []

    def guarda(b, esq, nome, met):
        e = {'base': b, 'esquema': esq, 'modelo': nome, **{k: float(v) for k, v in met.items()}}
        resumo.append(e)
        if (b, esq, nome) not in feito:
            with open(CACHE, 'a') as f:
                f.write(json.dumps(e) + '\n')
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()}, flush=True)

    def etapa(b, esq, nome, fun, arq=None):
        """Executa fun() só se a etapa não estiver no cache; fun devolve métricas ou um DataFrame OOF."""
        if (b, esq, nome) in feito:
            e = feito[(b, esq, nome)]
            return guarda(b, esq, nome, {k: v for k, v in e.items() if k not in ('base', 'esquema', 'modelo')})
        r = fun()
        if isinstance(r, pd.DataFrame):
            if arq:
                r.to_csv(RES / arq, index=False)
            r = metricas(r)
        guarda(b, esq, nome, r)

    # 1) demonstração do vazamento (hiperparâmetros fixos, como no TCC)
    dt, ft = base_tese()
    for nome in ['Tendência', 'LGB']:
        etapa('tese', 'aleatorio', nome, lambda: aleatorio(nome, dt, ft))
    for b in ['tcc_desalinhada', 'tcc_corrigida']:
        d, f = carrega(b)
        for nome in ['Tendência', 'LGB']:
            etapa(b, 'aleatorio', nome, lambda: aleatorio(nome, d, f))
            etapa(b, 'safra_fora', nome, lambda: oof('safra_fora', nome, d, f, aninhado=False))

    # 2) base do GEE: todos os esquemas e modelos, hiperparâmetros aninhados
    for nome in MODELOS:
        etapa('safra', 'aleatorio', nome, lambda: aleatorio(nome, base, FEATS))
    for esq in ['safra_fora', 'futuro', 'cidade_fora']:
        for nome in (MODELOS if esq != 'cidade_fora' else ['Tendência', 'LGB']):
            etapa('safra', esq, nome, lambda: oof(esq, nome), f'oof_safra_{esq}_{nome}.csv')
    # sensores: fusão MODIS + Landsat
    df_, ff_ = carrega('safra_fusao')
    etapa('safra_fusao', 'safra_fora', 'LGB', lambda: oof('safra_fora', 'LGB', df_, ff_),
          'oof_safra_fusao_safra_fora_LGB.csv')
    r = pd.DataFrame(resumo)
    r.to_csv(RES / 'validacao_resumo.csv', index=False)
    if ESCOLHAS:
        pd.DataFrame(ESCOLHAS).to_csv(RES / 'hiperparametros_escolhidos.csv', mode='a', index=False,
                                      header=not (RES / 'hiperparametros_escolhidos.csv').exists())

    # 3) forma da tendência (só tendência e LGB, safra fora)
    melhor = r[(r['base'] == 'safra') & (r['esquema'] == 'safra_fora') & (r['modelo'] != 'Tendência')] \
        .sort_values('RMSE').iloc[0]['modelo']
    for forma in ['linear', 'quadratica', 'regional']:
        for nome in ['Tendência', melhor]:
            etapa('safra', f'tendencia_{forma}', nome, lambda: oof('safra_fora', nome, forma=forma))
    pd.DataFrame([e for e in resumo if e['esquema'].startswith('tendencia_')]) \
        .to_csv(RES / 'tendencia_formas.csv', index=False)

    # 4) diferenças de MSE por bootstrap de safras (safra fora)
    ref = pd.read_csv(RES / f'oof_safra_safra_fora_{melhor}.csv')
    difs = []
    for nome in [m_ for m_ in MODELOS if m_ != melhor]:
        o = pd.read_csv(RES / f'oof_safra_safra_fora_{nome}.csv')
        difs.append(dict(zip(['comparacao', 'dif_MSE', 'IC_inf', 'IC_sup', 'p_boot'],
                             [f'{melhor} - {nome}', *bootstrap_dif(ref, o)])))
    fus = pd.read_csv(RES / 'oof_safra_fusao_safra_fora_LGB.csv')
    difs.append(dict(zip(['comparacao', 'dif_MSE', 'IC_inf', 'IC_sup', 'p_boot'],
                         ['LGB fusão - LGB só MODIS', *bootstrap_dif(fus, pd.read_csv(RES / 'oof_safra_safra_fora_LGB.csv'))])))
    pd.DataFrame(difs).to_csv(RES / 'bootstrap_diferencas.csv', index=False)
    print(r[['base', 'esquema', 'modelo', 'R2', 'MAE', 'RMSE', 'skill_tend']].round(3).to_string())
    print(pd.DataFrame(difs).round(1).to_string())
