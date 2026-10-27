"""Tabelas e figuras do material suplementar, geradas a partir de resultados/.

Saídas: artigo/tabelas_suplementares.tex (incluído por artigo/supplementary.tex) e
        artigo/figuras/figS*_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RES, ART = RAIZ / 'resultados', RAIZ / 'artigo'
VERDE, CINZA, TINTA, LARANJA = '#1b7837', '#8a8984', '#52514e', '#eb6834'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'axes.axisbelow': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6,
                     'axes.edgecolor': CINZA, 'axes.labelcolor': TINTA, 'xtick.color': TINTA,
                     'ytick.color': TINTA, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})


def num(x, d=2):
    if pd.isna(x):
        return ''
    t = f'{x:.{d}f}'
    return t[1:] if t.startswith('-') and float(t) == 0 else t.replace('-', '$-$')


def tabela(df, legenda, rotulo, alinh, longa=False):
    cab = ' & '.join(df.columns) + r' \\'
    linhas = '\n'.join(' & '.join(str(v) for v in r) + r' \\' for r in df.values)
    if longa:
        return (f'\\begingroup\\footnotesize\n\\begin{{longtable}}{{{alinh}}}\n\\caption{{{legenda}}}\\label{{{rotulo}}}\\\\\n\\toprule\n{cab}\n'
                f'\\midrule\n\\endfirsthead\n\\toprule\n{cab}\n\\midrule\n\\endhead\n{linhas}\n\\bottomrule\n\\end{{longtable}}\n\\endgroup\n')
    return (f'\\begin{{table}}[H]\n\\centering\n\\caption{{{legenda}}}\\label{{{rotulo}}}\n\\small\n'
            f'\\begin{{tabular}}{{{alinh}}}\n\\toprule\n{cab}\n\\midrule\n{linhas}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n')


partes = []

# S1 preditores
DESC = {'P': ('Precipitation', 'mm', 'CHIRPS'), 'dSeco': ('Days with $P < 1$ mm', 'days', 'CHIRPS'),
        'ETp': ('Potential evapotranspiration', 'mm', 'ERA5-Land'), 'ET': ('Actual evapotranspiration', 'mm', 'ERA5-Land'),
        'BH': ('Water balance $P - ET_p$ (Eq. 1)', 'mm', 'CHIRPS, ERA5-Land'),
        'dQuente': ('Days with $T_{max} > 32\\,^\\circ$C', 'days', 'ERA5-Land'),
        'Tmed': ('Mean air temperature', '$^\\circ$C', 'ERA5-Land'), 'Tmax': ('Mean maximum air temperature', '$^\\circ$C', 'ERA5-Land'),
        'Tmin': ('Mean minimum air temperature', '$^\\circ$C', 'ERA5-Land'), 'TmaxAbs': ('Absolute maximum air temperature', '$^\\circ$C', 'ERA5-Land'),
        'Td': ('Mean dew-point temperature', '$^\\circ$C', 'ERA5-Land'), 'Tsolo': ('Mean soil temperature, 0--7 cm', '$^\\circ$C', 'ERA5-Land'),
        'Usolo1': ('Volumetric soil water, 0--7 cm', 'm$^3$ m$^{-3}$', 'ERA5-Land'),
        'Usolo2': ('Volumetric soil water, 7--28 cm', 'm$^3$ m$^{-3}$', 'ERA5-Land'),
        'Rg': ('Global solar radiation', 'MJ m$^{-2}$', 'ERA5-Land'), 'Rn': ('Net solar radiation', 'MJ m$^{-2}$', 'ERA5-Land'),
        'IAF': ('Leaf area index of low vegetation', '--', 'ERA5-Land'),
        'NDVImax': ('Maximum NDVI', '--', 'MOD13Q1, MYD13Q1'), 'NDVImed': ('Mean NDVI', '--', 'MOD13Q1, MYD13Q1'),
        'EVImax': ('Maximum EVI', '--', 'MOD13Q1, MYD13Q1'), 'EVImed': ('Mean EVI', '--', 'MOD13Q1, MYD13Q1')}
s1 = pd.DataFrame([(v[0], v[1], v[2], 'F1--F4') for v in DESC.values()],
                  columns=['Predictor', 'Unit', 'Source', 'Phases'])
extra = pd.DataFrame([('Seasonal maximum NDVI and EVI', '--', 'MOD13Q1, MYD13Q1', 'season'),
                      ('Number of valid vegetation observations', '--', 'MOD13Q1, MYD13Q1', 'season'),
                      ('Soybean area', 'ha', 'MapBiomas Collection 10', 'season'),
                      ('Clay and sand content', '\\%', 'Tluszcz (2025)', 'static'),
                      ('Latitude, longitude, altitude', '$^\\circ$, m', 'IBGE', 'static')], columns=s1.columns)
partes.append(('Predictors', tabela(pd.concat([s1, extra]), 'Predictors of the municipality--season dataset (93 in total). Phase predictors were computed for each of the four phases.', 'tab:S_pred', '@{}llll@{}')))

# S2 validação completa
v = pd.read_csv(RES / 'validacao_resumo.csv')
v = v[~v['esquema'].str.startswith('tendencia_')]
NB = {'tese': 'Earlier (replica)', 'tcc_desalinhada': 'Earlier, misaligned',
      'tcc_corrigida': 'Earlier, aligned', 'safra': 'New', 'safra_fusao': 'New, + Landsat'}
NE = {'aleatorio': 'Random', 'safra_fora': 'LOSO', 'futuro': 'Rolling', 'cidade_fora': 'Mun. out'}
NM = {'Tendência': 'Trend', 'Ridge': 'Ridge', 'RF': 'Random forest', 'XGB': 'XGBoost', 'LGB': 'LightGBM'}
s2 = pd.DataFrame({'Dataset': v['base'].map(NB), 'Scheme': v['esquema'].map(NE), 'Model': v['modelo'].map(NM),
                   '$R^2$': v['R2'].map(num), 'MAE': v['MAE'].map(lambda x: num(x, 0)),
                   'RMSE': v['RMSE'].map(lambda x: num(x, 0)), '$R^2_{anom}$': v['R2_anom'].map(num),
                   'Skill': v['skill_tend'].map(num)})
partes.append(('Validation results', tabela(s2, 'Test performance of all datasets, validation schemes and models (MAE and RMSE in kg ha$^{-1}$).', 'tab:S_cv', '@{}lllrrrrr@{}', longa=True)))

# S3 bootstrap e formas da tendência
b = pd.read_csv(RES / 'bootstrap_diferencas.csv')
s3 = pd.DataFrame({'Comparison (A $-$ B)': b['comparacao'].str.replace('Tendência', 'trend').str.replace('fusão', 'fusion').str.replace('só MODIS', 'MODIS only'),
                   '$\\Delta$MSE ($10^3$ kg$^2$ ha$^{-2}$)': (b['dif_MSE'] / 1e3).map(lambda x: num(x, 1)),
                   '95\\% interval': [f'[{num(l / 1e3, 1)}; {num(h / 1e3, 1)}]' for l, h in zip(b['IC_inf'], b['IC_sup'])],
                   'Replicates $\\ge 0$': b['p_boot'].map(lambda x: num(x, 3))})
t = pd.read_csv(RES / 'tendencia_formas.csv')
s3b = pd.DataFrame({'Trend': t['esquema'].str.replace('tendencia_', '').map({'linear': 'Linear, municipal', 'quadratica': 'Quadratic, municipal', 'regional': 'Linear, regional slope'}),
                    'Model': t['modelo'].map(NM), '$R^2$': t['R2'].map(num), 'MAE': t['MAE'].map(lambda x: num(x, 0)),
                    'Skill': t['skill_tend'].map(lambda x: num(x, 3))})
partes.append(('Model and trend comparisons',
               tabela(s3, 'Differences in mean squared error between models under LOSO, with 95\\% intervals from 2000 bootstrap replicates of whole seasons.', 'tab:S_boot', '@{}lrcr@{}')
               + tabela(s3b, 'Effect of the form of the technological trend under LOSO (MAE in kg ha$^{-1}$).', 'tab:S_trend', '@{}llrrr@{}')))

# S4 métricas por município
pm = pd.read_csv(RES / 'metricas_municipio.csv').sort_values(['microrregiao', 'municipio'])
s4 = pd.DataFrame({'Municipality': pm['municipio'], 'Microregion': pm['microrregiao'],
                   'MAE': pm['MAE'].map(lambda x: num(x, 0)), 'RMSE': pm['RMSE'].map(lambda x: num(x, 0)),
                   'Skill': pm['skill_tend'].map(num), '$r$ anomaly': pm['corr_anom'].map(num),
                   'AUC shortfall': pm['auc_quebra'].map(num)})
partes.append(('Performance by municipality', tabela(s4, 'LOSO performance of ridge regression by municipality (MAE and RMSE in kg ha$^{-1}$; AUC for shortfall below 80\\% of the trend, blank when no shortfall occurred).', 'tab:S_mun', '@{}llrrrrr@{}', longa=True)))

# S5 por safra: Moran, cobertura, média regional
mo = pd.read_csv(RES / 'moran_residuos.csv').set_index('safra')
cs = pd.read_csv(RES / 'intervalos_por_safra.csv', index_col=0)
reg = pd.read_csv(RES / 'serie_regional.csv').set_index('safra')
s5 = pd.DataFrame({'Season': mo.index, 'Observed': reg['prod'].map(lambda x: num(x, 0)).values,
                   'Forecast': reg['pred'].map(lambda x: num(x, 0)).values, 'Trend': reg['tend'].map(lambda x: num(x, 0)).values,
                   "Moran's $I$": mo['I'].map(num).values, '$p$': mo['p'].map(lambda x: num(x, 3)).values,
                   'Conformal': (100 * cs['Conformal i.i.d.']).map(lambda x: num(x, 0)).values,
                   'Mixed': (100 * cs['Mixed model']).map(lambda x: num(x, 0)).values,
                   'CQR': (100 * cs['CQR']).map(lambda x: num(x, 0)).values})
partes.append(('Results by season', tabela(s5, "Regional mean yield (kg ha$^{-1}$), Moran's $I$ of the LOSO residuals and coverage (\\%) of the municipal 90\\% intervals by season.", 'tab:S_safra', '@{}rrrrrrrrr@{}', longa=True)))

# S6 risco
r = pd.read_csv(RES / 'risco_limiares.csv')
s6 = pd.DataFrame({'Threshold': (100 * r['limiar']).map(lambda x: f'< {x:.0f}\\%'), 'Frequency': r['freq'].map(lambda x: num(x, 3)),
                   'Brier': r['brier'].map(lambda x: num(x, 3)), 'Brier clim.': r['brier_clim'].map(lambda x: num(x, 3)),
                   'BSS': r['bss'].map(num), 'AUC': r['auc'].map(num)})
c = pd.read_csv(RES / 'confiabilidade.csv')
s6b = pd.DataFrame({'Threshold': (100 * c['limiar']).map(lambda x: f'< {x:.0f}\\%'), 'Mean forecast probability': c['p_media'].map(num),
                    'Observed frequency': c['freq_obs'].map(num), 'Forecasts': c['n']})
partes.append(('Shortfall risk', tabela(s6, 'Skill of shortfall probability forecasts by threshold (\\% of the trend).', 'tab:S_risk', '@{}lrrrrr@{}')
               + tabela(s6b, 'Reliability of shortfall probability forecasts: bins of forecast probability.', 'tab:S_rel', '@{}lrrr@{}')))

# figuras: Moran por safra e variograma
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={'wspace': 0.3})
cor = [VERDE if p < 0.05 else CINZA for p in mo['p']]
ax[0].bar(mo.index, mo['I'], color=cor, width=0.7)
ax[0].set_xlabel('Harvest year'); ax[0].set_ylabel("Moran's $I$ of residuals")
ax[0].set_title('(a) Spatial autocorrelation by season (green: p < 0.05)', loc='left', fontsize=9)
vg = pd.read_csv(RES / 'variograma.csv')
ri = pd.read_csv(RES / 'resumo_incerteza.csv', index_col=0)['valor']
c0, sill, pr = float(ri['variograma_pepita']), float(ri['variograma_patamar']), float(ri['variograma_alcance_pratico_km'])
h = np.linspace(0, vg['h'].max(), 200)
ax[1].plot(vg['h'], vg['gamma'] / 1e3, 'o', color=VERDE, ms=5, label='Empirical')
ax[1].plot(h, (c0 + (sill - c0) * (1 - np.exp(-3 * h / pr))) / 1e3, color=TINTA, lw=1.5, label='Exponential fit')
ax[1].set_xlabel('Distance (km)'); ax[1].set_ylabel('Semivariance (10³ kg² ha⁻²)')
ax[1].set_title('(b) Semivariogram of within-season residuals', loc='left', fontsize=9)
ax[1].legend(frameon=False, fontsize=8)
fig.savefig(ART / 'figuras/figS1_moran_variograma.png'); plt.close(fig)

with open(ART / 'tabelas_suplementares.tex', 'w') as f:
    f.write('%% GERADO por analise/06_suplementar.py a partir de resultados/ -- não editar à mão\n')
    for tit, tex in partes:
        f.write(f'\n\\section{{{tit}}}\n\n{tex}\n')
print('ok', len(partes), 'seções')
