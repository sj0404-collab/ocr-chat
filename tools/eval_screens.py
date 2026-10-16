#!/usr/bin/env python3
"""Оценка OCR по фикстурам: sim(GT-строка, лучшая OCR-строка) = 1 - CER. Итог = среднее."""
import sys, json, subprocess, re

def lev(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]; dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[len(b)]

def norm(s):
    return re.sub(r'\s+', ' ', s).strip().upper()

def run_ocr(model, imgpath):
    out = subprocess.run([sys.executable, 'ocr_engine_test.py', model, imgpath],
                         capture_output=True, text=True, timeout=600).stdout
    lines = re.findall(r'«(.*?)»', out)
    dark = 'darkPage=True' in out
    return lines, dark

def main(model, fixtures='gt_fixtures.json', updir='uploads'):
    gt = json.load(open(fixtures))
    grand, gn = 0.0, 0
    for fname, gt_lines in gt.items():
        ocr_lines, dark = run_ocr(model, f'{updir}/{fname}')
        ocr_n = [norm(x) for x in ocr_lines]
        print(f"\n### {fname} (dark={dark}) — GT {len(gt_lines)} строк")
        file_sim = 0.0
        for g in gt_lines:
            gn += 1
            g_n = norm(g)
            best, best_o = 0.0, ''
            pool = ocr_n + [ocr_n[i] + ' ' + ocr_n[i+1] for i in range(len(ocr_n)-1)] \
                    + [(ocr_n[i] + ' ' + ocr_n[i+1]).replace('- ', '') for i in range(len(ocr_n)-1)] \
                    + [o.replace('- ', '') for o in ocr_n]
            for o in pool:
                sim = 1 - lev(g_n, o) / max(1, len(g_n))
                if sim > best: best, best_o = sim, o
            # GT, покрытый конкатенацией соседних OCR-строк (разбиение), тоже засчитываем
            blob = norm(' '.join(ocr_lines))
            cover = 1 - lev(g_n, max(g_n, g_n) and (g_n if g_n in blob else g_n[:1])) if False else best
            file_sim += best; grand += best
            mark = 'OK ' if best >= 0.999 else ('~~ ' if best >= 0.8 else 'XX ')
            print(f"  [{mark}{best:.2f}] GT «{g}»  →  «{best_o}»")
        print(f"  == файл: {file_sim/len(gt_lines):.3f}")
    print(f"\n######## ИТОГО по {gn} GT-строкам: {grand/gn:.3f} (1.000 = идеал) ########")

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/tmp/r15.onnx')
