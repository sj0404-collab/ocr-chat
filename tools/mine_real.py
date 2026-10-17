#!/usr/bin/env python3
"""Добыча реальных строк с тестсета: прогоняем сегментацию+OCR моделью по 20 скринам,
к каждой найденной строке подбираем окно в GT с максимальным совпадением (>=0.55 по CER
и conf>=0.45). Кропы сохраняем в mined.npz (uint8 32h float-масштаб + метка = точное окно GT).
Модель тренируется на настоящих пикселях с точным текстом — снимаем доменный зазор."""
import sys, json, os, re, unicodedata, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def best_window(gt_entries_n, hyp_n):
    """best (sim, gt_idx, start, end) окна в norm(GT) против hyp целиком"""
    best = (0.0, -1, 0, 0)
    for gi, g in enumerate(gt_entries_n):
        n = len(hyp_n)
        if n == 0 or n > len(g): continue
        for i in range(0, len(g) - n + 1):
            w = g[i:i + n]
            sim = 1 - lev(hyp_n, w) / n
            if sim > best[0]: best = (sim, gi, i, i + n)
    return best

def main():
    model = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'testset20', 'mined.npz')
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    src = open(os.path.join(ROOT, 'tools', 'ocr_engine_port.py')).read()
    ns = {}; exec(src.split('def main')[0].replace('if __name__', '#'), ns)
    from PIL import Image
    import onnxruntime as ort, math
    luma_px, segment_any, decode = ns['luma_px'], ns['segment_any'], ns['decode']
    LINE_H = ns['LINE_H']; MAX_DIM = ns['MAX_DIM']
    gt = json.load(open(os.path.join(ROOT, 'testset20', 'gt20.json')))
    sess = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
    imgs, texts, sims = [], [], []
    raw_gt = []
    for lines in gt.values(): raw_gt += lines
    gt_n = [norm(s) for s in raw_gt]
    def gt_window_text(gi, a, b):
        # восстанавливаем читабельный фрагмент: берём символы исходной строки в долях окна
        s = raw_gt[gi]; L = max(1, len(gt_n[gi]))
        i0 = round(a / L * len(s)); i1 = round(b / L * len(s))
        frag = s[i0:i1].strip(' -–—.,!?…«»()[]')
        return frag
    for key in sorted(gt):
        img = Image.open(f'{ROOT}/testset20/{key}.jpg').convert('RGB') if os.path.exists(f'{ROOT}/testset20/{key}.jpg') \
            else Image.open(f'{ROOT}/testset20/{key}.png').convert('RGB')
        w0, h0 = img.size
        sc = min(1.0, MAX_DIM / max(w0, h0))
        if sc < 1.0: img = img.resize((max(1, round(w0 * sc)), round(h0 * sc)), Image.LANCZOS)
        lum = luma_px(np.asarray(img))
        dark = float(lum.reshape(-1)[::7].mean()) < 100
        boxes = segment_any(lum, dark)
        hits = 0
        for (x, y, bw, bh, pol) in boxes:
            sw = min(bw, img.size[0] - x); sh = min(bh, img.size[1] - y)
            if sw <= 0 or sh <= 0: continue
            new_w = max(8, round(sw * LINE_H / sh)); new_w = min(444, new_w)
            pad_w = math.ceil((new_w + 4) / 8) * 8
            line = Image.new('RGB', (pad_w, LINE_H), (255, 255, 255))
            crop = img.crop((x, y, x + sw, y + sh)).resize((new_w, LINE_H), Image.LANCZOS)
            line.paste(crop, (4, 0))
            lpix = luma_px(np.asarray(line)).astype(np.float32) / 255.0
            if pol: lpix = 1.0 - lpix
            raw = sess.run(None, {'x': lpix[None, None, :, :]})[0]
            t, conf = decode(raw)
            hyp = norm(t)
            if len(hyp) < 3 or conf < 0.45: continue
            sim, gi, a, b = best_window(gt_n, hyp)
            if sim < 0.55 or gi < 0: continue
            frag = gt_window_text(gi, a, b)
            frag = ''.join(c for c in unicodedata.normalize('NFKC', frag)
                           if c.isalnum() or c in ' -–—…№')
            frag = ' '.join(frag.split())
            if len(norm(frag)) < 3: continue
            imgs.append((lpix * 255).astype(np.uint8)); texts.append(frag); sims.append(sim)
            hits += 1
        print(f'{key}: lines={len(boxes)} mined={hits}', flush=True)
    np.savez_compressed(outp, texts=np.array(texts), sims=np.array(sims, np.float32),
                        **{f'img{i}': im for i, im in enumerate(imgs)})
    print(f'SAVED {outp}: {len(imgs)} lines, sim avg={np.mean(sims):.3f}', flush=True)

if __name__ == '__main__':
    main()
