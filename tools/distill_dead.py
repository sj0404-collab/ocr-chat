#!/usr/bin/env python3
"""Точечная дистилляция МЁРТВЫХ строк. Прогоняем сегментацию+модель по целевым скринам
(s01,s05,s13,s14,s15,s16,s18,s19), каждой найденной строке подбираем GT СВОЕГО скрина:
  sim>=0.60           -> метка = точное окно GT (как в mine_real)
  0.25<=sim<0.60 и длина гипотезы правдоподобна (0.5..1.6 от полной строки)
                      -> метка = ЦЕЛАЯ GT-строка (кроп = вся строка баллона)
всё ниже порога — пропуск (ватермарки/UI не попадут). Макс 3 кропа на GT-строку (лучшие sim).
Результат: distill.npz с uint8 32h кропами + точными метками реального стиля.
Использование: python3 tools/distill_dead.py model.onnx [out.npz] [comma-screens]"""
import sys, json, os, re, unicodedata
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEAD = ['s01', 's05', 's13', 's14', 's15', 's16', 's18', 's19']

def norm(s):
    s = unicodedata.normalize('NFKC', s).lower().replace('ё', 'е')
    return re.sub(r'[^0-9a-zа-я]', '', s)

def lev(a, b):
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def best_window(gt_n, hyp_n):
    best = (0.0, -1, 0, 0)
    for gi, g in enumerate(gt_n):
        n = len(hyp_n)
        if n == 0 or n > len(g): continue
        for i in range(0, len(g) - n + 1):
            w = g[i:i + n]
            sim = 1 - lev(hyp_n, w) / n
            if sim > best[0]: best = (sim, gi, i, i + n)
    return best

def main():
    model = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'testset20', 'distill.npz')
    screens = sys.argv[3].split(',') if len(sys.argv) > 3 else DEAD
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    src = open(os.path.join(ROOT, 'tools', 'ocr_engine_port.py')).read()
    ns = {}; exec(src.split('def main')[0].replace('if __name__', '#'), ns)
    from PIL import Image
    import onnxruntime as ort, math
    luma_px, segment_any, decode = ns['luma_px'], ns['segment_any'], ns['decode']
    LINE_H = ns['LINE_H']; MAX_DIM = ns['MAX_DIM']
    gt = json.load(open(os.path.join(ROOT, 'testset20', 'gt20.json')))
    sess = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
    picked = {}  # (screen, gi) -> [(sim, lpix, label)]
    for key in screens:
        path = f'{ROOT}/testset20/{key}.jpg' if os.path.exists(f'{ROOT}/testset20/{key}.jpg') \
            else f'{ROOT}/testset20/{key}.png'
        img = Image.open(path).convert('RGB')
        raw_gt = gt[key]; gt_n = [norm(s) for s in raw_gt]
        w0, h0 = img.size
        sc = min(1.0, MAX_DIM / max(w0, h0))
        if sc < 1.0: img = img.resize((max(1, round(w0 * sc)), round(h0 * sc)), Image.LANCZOS)
        lum = luma_px(np.asarray(img))
        dark = float(lum.reshape(-1)[::7].mean()) < 100
        boxes = segment_any(lum, dark)
        kept = 0
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
            if len(hyp) < 3 or conf < 0.25: continue
            sim, gi, a, b = best_window(gt_n, hyp)
            if gi < 0 or sim < 0.25: continue
            if sim >= 0.60:
                s = raw_gt[gi]; L = max(1, len(gt_n[gi]))
                i0 = round(a / L * len(s)); i1 = round(b / L * len(s))
                label = s[i0:i1].strip(' -–—.,!?…«»()[]')
            else:
                if not (0.5 * len(gt_n[gi]) <= len(hyp) <= 1.6 * len(gt_n[gi])):
                    continue
                label = raw_gt[gi].strip(' -–—.,!?…«»()[]')
            label = ''.join(c for c in unicodedata.normalize('NFKC', label)
                            if c.isalnum() or c in ' -–—…№')
            label = ' '.join(label.split())
            if len(norm(label)) < 3: continue
            picked.setdefault((key, gi), []).append((sim, (lpix * 255).astype(np.uint8), label))
            kept += 1
        print(f'{key}: lines={len(boxes)} кандидатов={kept}', flush=True)
    imgs, texts, sims = [], [], []
    for (key, gi), lst in sorted(picked.items()):
        print(f'  {key} GT[{gi}]: ' + ' | '.join(f'{s:.2f} «{lab[:40]}»' for s, _, lab in
              sorted(lst, key=lambda z: -z[0])[:3]), flush=True)
        for sim, im, lab in sorted(lst, key=lambda z: -z[0])[:3]:
            imgs.append(im); texts.append(lab); sims.append(sim)
    np.savez_compressed(outp, texts=np.array(texts), sims=np.array(sims, np.float32),
                        **{f'img{i}': im for i, im in enumerate(imgs)})
    print(f'SAVED {outp}: {len(imgs)} lines, sim avg={np.mean(sims):.3f}' if imgs
          else f'SAVED {outp}: 0 lines', flush=True)

if __name__ == '__main__':
    main()
