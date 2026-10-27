"""Recupera do log de 02_validacao.py as etapas já concluídas e grava em resultados/validacao_cache.jsonl.
Uso na retomada após interrupção: .venv/bin/python analise/semeia_cache.py && .venv/bin/python analise/02_validacao.py
"""
import ast
import json
import re
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / 'resultados'
cache = RES / 'validacao_cache.jsonl'
feito = set()
if cache.exists():
    feito = {(e['base'], e['esquema'], e['modelo']) for e in map(json.loads, cache.read_text().splitlines())}
novos = 0
with open(cache, 'a') as f:
    for l in (RES / 'log_02.txt').read_text().splitlines():
        if not l.startswith("{'base'"):
            continue
        e = ast.literal_eval(re.sub(r'np\.float64\(([^)]*)\)', r'\1', l))
        k = (e['base'], e['esquema'], e['modelo'])
        oof = {'safra_fora', 'futuro', 'cidade_fora'}
        # etapas com predições fora-da-amostra só valem se o CSV correspondente existir
        if e['base'] == 'safra' and e['esquema'] in oof and not (RES / f"oof_safra_{e['esquema']}_{e['modelo']}.csv").exists():
            continue
        if k not in feito:
            f.write(json.dumps(e) + '\n'); feito.add(k); novos += 1
print(f'{novos} etapas recuperadas do log; {len(feito)} no cache')
