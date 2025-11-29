#!/usr/bin/env python3
"""Analyse la distribution des tailles de fichiers HTML"""
from pathlib import Path
import statistics

data_dir = Path("/home/rag/data")
html_files = list(data_dir.glob("*.html"))

sizes = []
for file in html_files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            lines = len(f.readlines())
            char_count = file.stat().st_size
            sizes.append({
                'name': file.name,
                'lines': lines,
                'chars': char_count
            })
    except Exception as e:
        print(f"Error reading {file.name}: {e}")

# Trier par nombre de lignes
sizes.sort(key=lambda x: x['lines'])

print("="*80)
print("ANALYSE DES FICHIERS HTML")
print("="*80)

print(f"\nTotal fichiers: {len(sizes)}")
print(f"\nLignes - Min: {sizes[0]['lines']} | Max: {sizes[-1]['lines']} | Moyenne: {statistics.mean([s['lines'] for s in sizes]):.0f}")
print(f"Chars  - Min: {sizes[0]['chars']:,} | Max: {sizes[-1]['chars']:,} | Moyenne: {statistics.mean([s['chars'] for s in sizes]):,.0f}")

print("\n" + "="*80)
print("10 FICHIERS LES PLUS PETITS")
print("="*80)
for s in sizes[:10]:
    print(f"{s['lines']:4d} lignes | {s['chars']:8,} chars | {s['name']}")

print("\n" + "="*80)
print("10 FICHIERS LES PLUS GRANDS")
print("="*80)
for s in sizes[-10:]:
    print(f"{s['lines']:4d} lignes | {s['chars']:8,} chars | {s['name']}")

# Distribution par quartiles
print("\n" + "="*80)
print("DISTRIBUTION PAR QUARTILES (lignes)")
print("="*80)
line_counts = [s['lines'] for s in sizes]
print(f"Q1 (25%):  {statistics.quantiles(line_counts, n=4)[0]:.0f} lignes")
print(f"Q2 (50%):  {statistics.quantiles(line_counts, n=4)[1]:.0f} lignes (médiane)")
print(f"Q3 (75%):  {statistics.quantiles(line_counts, n=4)[2]:.0f} lignes")

# Catégorisation
print("\n" + "="*80)
print("CATÉGORISATION")
print("="*80)
very_short = [s for s in sizes if s['lines'] < 50]
short = [s for s in sizes if 50 <= s['lines'] < 100]
medium = [s for s in sizes if 100 <= s['lines'] < 200]
long_docs = [s for s in sizes if 200 <= s['lines'] < 400]
very_long = [s for s in sizes if s['lines'] >= 400]

print(f"Très courts (< 50 lignes):      {len(very_short):3d} fichiers")
print(f"Courts (50-100 lignes):         {len(short):3d} fichiers")
print(f"Moyens (100-200 lignes):        {len(medium):3d} fichiers")
print(f"Longs (200-400 lignes):         {len(long_docs):3d} fichiers")
print(f"Très longs (≥ 400 lignes):      {len(very_long):3d} fichiers")

print("\n" + "="*80)
print("RECOMMANDATIONS DE CHUNK SIZE")
print("="*80)
print("Très courts:  300-400 chars  (garder contexte complet)")
print("Courts:       400-500 chars  (sections complètes)")
print("Moyens:       500-700 chars  (bon équilibre)")
print("Longs:        600-800 chars  (précision)")
print("Très longs:   700-1000 chars (éviter fragmentation)")
