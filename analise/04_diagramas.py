"""Diagramas de método: fluxograma (D1) e esquemas de validação (D2)."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch

RAIZ = Path(__file__).resolve().parents[1]
FIG = RAIZ / 'figuras'
VERDE, VERDE_CL, NEUTRO, TINTA, CINZA = '#1b7837', '#bae4b3', '#f0efec', '#52514e', '#8a8984'
plt.rcParams.update({'font.size': 9, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})


def salva(fig, nome):
    fig.savefig(FIG / f'{nome}.png')
    fig.savefig(RAIZ / 'artigo/figuras' / f'{nome}.png')
    plt.close(fig)


# ---------------------------------------------------------------- D2: esquemas de validação
N_MUN, N_SAF = 49, 25
micro = np.repeat([0, 1, 2], [20, 18, 11])          # Toledo, Cascavel, Foz do Iguaçu
rng = np.random.default_rng(3)
s_teste, m_teste, micro_teste, origem = 14, 30, 1, 15

esquemas = {}
m = (rng.random((N_MUN, N_SAF)) < 0.3).astype(int) + 1
esquemas['(a) Random 70/30'] = m
m = np.ones((N_MUN, N_SAF), int); m[m_teste] = 2
esquemas['(b) Municipality out'] = m
m = np.ones((N_MUN, N_SAF), int); m[:, s_teste] = 2
esquemas['(c) Season out'] = m
m = np.ones((N_MUN, N_SAF), int); m[micro == micro_teste] = 0; m[:, s_teste] = 0
m[np.ix_(micro == micro_teste, [s_teste])] = 2
esquemas['(d) Season and microregion out'] = m
m = np.zeros((N_MUN, N_SAF), int); m[:, :origem] = 1; m[:, origem] = 2
esquemas['(e) Rolling origin'] = m

cmap = ListedColormap([NEUTRO, VERDE_CL, VERDE])
fig, axs = plt.subplots(1, 5, figsize=(12, 3.2), sharey=True, gridspec_kw={'wspace': 0.18})
for ax, (tit, m) in zip(axs, esquemas.items()):
    ax.imshow(m, cmap=cmap, vmin=0, vmax=2, aspect='auto', interpolation='nearest')
    for y in [19.5, 37.5]:
        ax.axhline(y, color='white', lw=1.5)
    ax.set_title(tit, fontsize=8.5, loc='left', color=TINTA)
    ax.set_xticks([0, 24]); ax.set_xticklabels(['2001', '2025'], fontsize=7.5)
    ax.set_xlabel('Season (harvest)', color=TINTA)
    for s in ax.spines.values():
        s.set_visible(False)
axs[0].set_yticks([9.5, 28.5, 43]); axs[0].set_yticklabels(['Toledo', 'Cascavel', 'Foz do\nIguaçu'])
axs[0].set_ylabel('Municipalities by microregion', color=TINTA)
fig.legend(handles=[Patch(color=VERDE_CL, label='Training'), Patch(color=VERDE, label='Test'),
                    Patch(color=NEUTRO, label='Not used')], loc='lower center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.16))
salva(fig, 'fig_esquemas_validacao')

# ---------------------------------------------------------------- D1: fluxograma
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')


def caixa(x, y, w, h, titulo, texto, forte=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                facecolor=VERDE if forte else '#edf8e9', edgecolor=VERDE, lw=1.2))
    cor = 'white' if forte else TINTA
    ax.text(x + w / 2, y + h - 0.18, titulo, ha='center', va='top', fontsize=8.5, weight='bold', color=cor)
    ax.text(x + w / 2, y + h - 0.45, texto, ha='center', va='top', fontsize=7.2, color=cor, linespacing=1.35)


def seta(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=CINZA, lw=1.2))


# dados
caixa(0.1, 4.3, 2.2, 1.55, 'Satellite', 'MODIS MOD13Q1/MYD13Q1\nLandsat 5/7/8/9 (QA mask)\nNDVI, EVI on the\nMapBiomas soybean mask')
caixa(2.5, 4.3, 2.2, 1.55, 'Weather', 'ERA5-Land (temperature,\nradiation, ET, soil water)\nCHIRPS (precipitation)\nby ten-day period')
caixa(4.9, 4.3, 2.2, 1.55, 'Yield', 'IBGE/PAM table 1612\n49 municipalities\nharvests 2001–2025')
caixa(7.3, 4.3, 2.6, 1.55, 'Phenological phases', 'F1 Oct · F2 Nov–Dec\nF3 Jan · F4 Feb–Mar\n(sowing to maturity)')
# base
caixa(2.5, 2.55, 4.6, 1.2, 'Municipality × season dataset',
      'aggregation by phase · alignment by harvest year\nquality control · Google Earth Engine', forte=True)
for x in [1.2, 3.6, 6.0, 8.6]:
    seta(x, 4.3, 4.8, 3.75)
# modelagem
caixa(0.1, 0.2, 2.3, 1.75, 'Trend and models', 'municipal linear trend\n(training seasons only)\nRidge · RF · XGBoost\nLightGBM on the anomaly')
caixa(2.6, 0.2, 2.3, 1.75, 'Validation schemes', 'random · municipality out\nseason out · season and\nmicroregion out\nrolling origin')
caixa(5.1, 0.2, 2.3, 1.75, 'Error structure', 'season component\nMoran\'s I by season\nmixed model')
caixa(7.6, 0.2, 2.3, 1.75, 'Uncertainty and risk', 'conformal intervals\n(i.i.d. × season blocks, CQR)\nshortfall probability\nlead time · SHAP')
for x in [1.25, 3.75, 6.25, 8.75]:
    seta(4.8, 2.55, x, 1.95)
salva(fig, 'fig_fluxograma')
print('ok')
