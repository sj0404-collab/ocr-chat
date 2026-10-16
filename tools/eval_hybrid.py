#!/usr/bin/env python3
"""Гибрид: классическая сегментация (порт v1.6) + YOLO-детектор аниме-текста в непокрытых зонах.
Скоринг тот же, что в eval_screens.py (best-match + склейка соседних строк)."""
import sys, json, re, math
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
import ocr_engine_test as E

DET_ONNX = '/tmp/det/yolo12.onnx'
DSZ = 960

def lev(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]; dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[len(b)]

def norm(s): return re.sub(r'\s+', ' ', s).strip().upper()

def det_boxes(page_img, sess):
    w, h = page_img.size; r = min(DSZ / w, DSZ / h)
    nw, nh = round(w * r), round(h * r)
    im = cv2.resize(np.asarray(page_img), (nw, nh))
    pad = np.full((DSZ, DSZ, 3), 114, np.uint8); pad[:nh, :nw] = im
    o = sess.run(None, {'images': pad.transpose(2, 0, 1)[None].astype(np.float32) / 255.0})[0][0]
    cx, cy, bw, bh, cf = o
    keep = np.where(cf > 0.15)[0]; sc = cf[keep]
    cand = np.stack([cx[keep] - bw[keep] / 2, cy[keep] - bh[keep] / 2, bw[keep], bh[keep]], 1)
    idx = cv2.dnn.NMSBoxes(cand.tolist(), sc.tolist(), 0.0, 0.4)
    idx = idx.flatten().tolist() if len(idx) else []
    out = []
    for i in idx:
        x = max(0, cand[i][0] / r); y = max(0, cand[i][1] / r)
        out.append((round(x), round(y), round(cand[i][2] / r), round(cand[i][3] / r)))
    return out

def recognize(sess, page, boxes):
    results = []
    for (x, y, bw, bh, pol) in boxes:
        sw = min(bw, page.size[0] - x); sh = min(bh, page.size[1] - y)
        if sw <= 0 or sh <= 0: continue
        new_w = max(8, round(sw * E.LINE_H / sh)); new_w = min(444, new_w)
        pad_w = math.ceil((new_w + 4) / 8) * 8
        line = Image.new('RGB', (pad_w, E.LINE_H), (255, 255, 255))
        line.paste(page.crop((x, y, x + sw, y + sh)).resize((new_w, E.LINE_H), Image.LANCZOS), (4, 0))
        lpix = E.luma_px(np.asarray(line)).astype(np.float32) / 255.0
        if pol: lpix = 1.0 - lpix
        raw = sess.run(None, {'x': lpix[None, None, :, :]})[0]
        t, _ = E.decode(raw)
        if t.strip(): results.append(t)
    return results

DETPRIO=False
def segment_hybrid(page, lum, dark, detsess):
    h, w = lum.shape
    base = E.segment_any(lum, dark)          # классические строки (prio impl.)
    boxes = [tuple(b) for b in base]
    added = 0
    for (dx, dy, dw, dh) in det_boxes(page, detsess):
        cy = dy + dh / 2
        if cy < 0.070 * h or cy > 0.86 * h:  # статус-бар / навигация скриншота
            continue
        if dw < 14 or dh < 10 or dw > w * 0.95 and dh > h * 0.30:
            continue
        # если бокс почти целиком покрыт классическими строками — пропускаем
        overlap = 0
        for (x, y, bw, bh, _) in boxes:
            ix = max(0, min(dx + dw, x + bw) - max(dx, x))
            iy = max(0, min(dy + dh, y + bh) - max(dy, y))
            overlap += ix * iy
        if not DETPRIO and overlap > 0.5 * dw * dh:
            continue
        if DETPRIO and overlap:
            # выкидываем классические строки, которые пересекаются с дет-боксом >25% их площади
            boxes = [b for b in boxes if not (
                max(0, min(b[0]+b[2], dx+dw) - max(b[0], dx)) *
                max(0, min(b[1]+b[3], dy+dh) - max(b[1], dy)) > 0.25 * b[2] * b[3])]
        sub = lum[dy:min(h, dy + dh), dx:min(w, dx + dw)]
        if sub.size == 0: continue
        pol = 0 if np.median(sub) > 127 else 1
        for (lx, ly, lw, lh) in E.bubble_lines(sub[4:-4, 4:-4] if dh > 16 and dw > 16 else sub, pol):
            gx, gy = dx + 4 + lx, dy + 4 + ly
            dup = False
            for (x, y, bw, bh, _) in boxes:
                ix = max(0, min(gx + lw, x + bw) - max(gx, x))
                iy = max(0, min(gy + lh, y + bh) - max(gy, y))
                if ix * iy > 0.55 * lw * lh:
                    dup = True; break
            if not dup:
                boxes.append((gx, gy, lw, lh, pol)); added += 1
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes, added

def main(model, fixtures='gt_fixtures.json', updir='uploads', use_det=True):
    gt = json.load(open(fixtures))
    detsess = ort.InferenceSession(DET_ONNX, providers=['CPUExecutionProvider']) if use_det else None
    rec = None
    grand, gn = 0.0, 0
    for fname, gt_lines in gt.items():
        img = Image.open(f'{updir}/{fname}').convert('RGB')
        w0, h0 = img.size
        sc = min(1.0, E.MAX_DIM / max(w0, h0))
        if sc < 1.0:
            img = img.resize((max(1, round(w0 * sc)), max(1, round(h0 * sc))), Image.LANCZOS)
        lum = E.luma_px(np.asarray(img))
        dark = float(lum.reshape(-1)[::7].mean()) < 100
        if rec is None:
            rec = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
        if use_det:
            boxes, added = segment_hybrid(img, lum, dark, detsess)
        else:
            boxes = E.segment_any(lum, dark); added = 0
        ocr = [norm(t) for t in recognize(rec, img, boxes)]
        print(f"\n### {fname} lines={len(boxes)} (det +{added})")
        file_sim = 0.0
        for g in gt_lines:
            gn += 1
            g = norm(g)
            best, best_o = 0.0, ''
            pool = ocr + [ocr[i] + ' ' + ocr[i + 1] for i in range(len(ocr) - 1)] + \
                   [(ocr[i] + ' ' + ocr[i + 1]).replace('- ', '') for i in range(len(ocr) - 1)]
            pool += [o.replace('- ', '') for o in ocr]
            for o in pool:
                sim = 1 - lev(g, o) / max(1, len(g))
                if sim > best: best, best_o = sim, o
            file_sim += best
            print(f"  [{'OK' if best >= .99 else '~~' if best >= .8 else 'XX'} {best:.2f}] GT «{g}»  →  «{best_o}»")
        print(f"  == файл: {file_sim / len(gt_lines):.3f}")
        grand += file_sim
    print(f"\n######## ИТОГО по {gn} GT-строкам: {grand / gn:.3f} ########")

if __name__ == '__main__':
    if len(sys.argv) > 3 and sys.argv[3] == 'prio':
        globals()['DETPRIO'] = True
    main(sys.argv[1] if len(sys.argv) > 1 else '/tmp/r22.onnx',
         use_det=(len(sys.argv) < 3 or sys.argv[2] != 'nodet'))
