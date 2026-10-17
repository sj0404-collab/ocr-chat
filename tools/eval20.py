#!/usr/bin/env python3
"""Оценка по 20 скринам testset20: GT-реплика «найдена», если её нормализованная форма
(нижний регистр, ё=е, только [0-9a-zа-я], пробелы стёрты) содержится в склейке OCR-строк.
Частичный скор = 1 - CER против лучшего окна той же длины в склейке.
Использование: python3 tools/eval20.py MODEL.onnx [--quick] [--fail]
  --quick: 6 самых сложных скринов (для цикла внутри обучения)
  --fail : подробно по несовпадениям
Финальная строка лога: "SCORE=<binary> SOFT=<partial> HIT=<m>/<n>" — парсится обучением."""
import sys, json, subprocess, re, unicodedata, os

QUICK = ['s01', 's05', 's06', 's15', 's16', 's18']

def norm(s):
    s = unicodedata.normalize('NFKC', s).lower().replace('ё', 'е')
    return re.sub(r'[^0-9a-zа-я]', '', s)

def lev(a, b):
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def window_sim(g, blob):
    """max over windows of len(g) chars in blob of (1 - lev/len(g))"""
    if g in blob: return 1.0, g
    n, best, bo = len(g), 0.0, ''
    if n > len(blob): return 0.0, ''
    for i in range(0, len(blob) - n + 1):
        w = blob[i:i + n]
        d = lev(g, w)
        sim = 1 - d / n
        if sim > best: best, bo = sim, w
        if best >= 0.999: break
    return best, bo

def run_ocr(port, model, img):
    out = subprocess.run([sys.executable, port, model, img],
                         capture_output=True, text=True, timeout=900).stdout
    return re.findall(r'«(.*?)»', out)

def main():
    model = sys.argv[1]
    quick = '--quick' in sys.argv
    failv = '--fail' in sys.argv
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    port = os.path.join(root, 'tools', 'ocr_engine_port.py')
    gt = json.load(open(os.path.join(root, 'testset20', 'gt20.json')))
    keys = sorted(gt) if not quick else QUICK
    hit_n = soft = 0
    tot = 0
    for k in keys:
        img = None
        for ext in ('jpg', 'png'):
            p = f'{root}/testset20/{k}.{ext}'
            if os.path.exists(p): img = p
        lines = run_ocr(port, model, img)
        blob = norm(' '.join(lines))
        shits = 0
        fails = []
        for g in gt[k]:
            gn = norm(g)
            sim, bo = window_sim(gn, blob)
            tot += 1
            if sim >= 0.999: hit_n += 1; shits += 1
            soft += sim
            if sim < 0.999: fails.append((g, sim, bo))
        tag = '  '.join(fails and (f'{g}~~{sim:.2f}→{bo}' for g, sim, bo in fails[:3])
                            if failv else ([f'fails={len(fails)}/{len(gt[k])}'] if fails else [])) or 'clean'
        print(f'{k}: {shits}/{len(gt[k])}  {tag}', flush=True)
    print(f'SCORE={hit_n / max(1, tot):.3f} SOFT={soft / max(1, tot):.3f} HIT={hit_n}/{tot}', flush=True)

if __name__ == '__main__':
    main()
