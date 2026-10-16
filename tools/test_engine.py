#!/usr/bin/env python3
"""Точный порт OcrEngine.kt v1.5 (баллоны на любых страницах + остаточный проход, лимит 1500 компонентов)."""
import sys, math
import numpy as np
from PIL import Image
import onnxruntime as ort

CHARSET = ("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
           "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
           "ЇїІіЄєҐґ"
           " .,!?;:\"'«»„“”()[]—–-…№*/0123456789")
BLANK, UNK, LINE_H, MAX_DIM = 109, 110, 32, 1400
assert len(CHARSET) == 109, len(CHARSET)

def luma_px(rgb):
    return (rgb[:, :, 0].astype(np.int32) * 299 + rgb[:, :, 1].astype(np.int32) * 587
            + rgb[:, :, 2].astype(np.int32) * 114) // 1000

def dilate(m, rx, ry):
    h, w = m.shape
    tmp = np.zeros_like(m)
    for y in range(h):
        run = 0
        for x in range(w + rx):
            if x < w: run += m[y, x]
            if x > rx * 2: run -= m[y, x - rx * 2 - 1]
            if x >= rx and run > 0: tmp[y, x - rx] = 1
    out = np.zeros_like(m)
    for x in range(w):
        run = 0
        for y in range(h + ry):
            if y < h: run += tmp[y, x]
            if y > ry * 2: run -= tmp[y - ry * 2 - 1, x]
            if y >= ry and run > 0: out[y - ry, x] = 1
    return out

def erode(m, rx, ry):
    return 1 - dilate(1 - m, rx, ry)

def components(mask, with_count=False):
    h, w = mask.shape
    lab = np.full((h, w), -1, np.int32)
    out = []
    cid = 0
    for s in range(h * w):
        y, x = divmod(s, w)
        if mask[y, x] == 0 or lab[y, x] >= 0: continue
        q = [(y, x)]; lab[y, x] = cid
        x0 = x1 = x; y0 = y1 = y; cnt = 0
        head = 0
        while head < len(q):
            cy, cx = q[head]; head += 1; cnt += 1
            if cx < x0: x0 = cx
            if cx > x1: x1 = cx
            if cy < y0: y0 = cy
            if cy > y1: y1 = cy
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0: continue
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] == 1 and lab[ny, nx] < 0:
                        lab[ny, nx] = cid; q.append((ny, nx))
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        if bh >= 10 and bw >= 12:
            out.append([x0, y0, bw, bh, cnt] if with_count else [x0, y0, bw, bh])
            if len(out) >= 1500: return out
        cid += 1
    return out

def group_rows(comps, w, h):
    comps = sorted(comps, key=lambda b: (b[1], b[0]))
    rows = []
    for b in comps:
        bh = b[3]; cy = b[1] + bh / 2.0
        placed = False
        for row in rows:
            ry0 = min(r[1] for r in row); ry1 = max(r[1] + r[3] for r in row)
            if ry0 - 0.8 * bh <= cy <= ry1 + 0.8 * bh:
                row.append(b); placed = True; break
        if not placed: rows.append([b])
    rows.sort(key=lambda row: min(r[1] for r in row))
    out = []
    for row in rows:
        row.sort(key=lambda r: r[0])
        x0 = min(r[0] for r in row); y0 = min(r[1] for r in row)
        x1 = max(r[0] + r[2] for r in row); y1 = max(r[1] + r[3] for r in row)
        cx0, cy0 = max(0, min(x0, w - 1)), max(0, min(y0, h - 1))
        cx1, cy1 = max(1, min(x1, w)), max(1, min(y1, h))
        if cx1 > cx0 and cy1 > cy0:
            out.append((cx0, cy0, cx1 - cx0, cy1 - cy0))
    return out

def seg_mask(lum, dark):
    """Маска текста: тёмный текст (l<170) или светлый (l>127) при dark."""
    return ((lum > 127) if dark else (lum < 170)).astype(np.uint8)

def segment(lum, dark):
    """Базовая сегментация v1.1: маска → замыкание ×2 → компоненты → строки."""
    mask = seg_mask(lum, dark)
    for _ in range(2):
        mask = erode(dilate(mask, 12, 1), 12, 1)
    return group_rows(components(mask), *lum.shape[::-1])

def find_bubbles(lum, dark_blob):
    """Однотонные баллоны на тёмной странице: СЫРЫЕ компоненты (без замыкания! — оно склеивает
    их с бордюрами скриншота), отсечка: размеры, сплошность fill>=0.45, не касается края."""
    h, w = lum.shape
    raw = ((lum < 45) if dark_blob else (lum > 210)).astype(np.uint8)
    out = []
    for b in components(raw, with_count=True):
        x, y, bw, bh, cnt = b
        area = bw * bh
        if bw < 60 or bh < 40 or area < w * h // 50 or area * 2 > w * h: continue
        if cnt < area * 0.45: continue
        if x == 0 or y == 0 or x + bw == w or y + bh == h: continue
        out.append((x, y, bw, bh, dark_blob))
    return out

def bubble_lines(sub, pol):
    """Строки внутри баллона: зануляем компоненты, касающиеся края (светлое кольцо контура) и
    мусор <3px (JPEG-шум), затем строки = горизонтальные полосы по rowsum."""
    ih, iw = sub.shape
    m = ((sub > 127) if pol else (sub < 127)).astype(np.uint8)
    lab = np.zeros((ih, iw), np.uint8)
    for s0 in range(ih * iw):
        yy, xx = divmod(s0, iw)
        if m[yy, xx] == 0 or lab[yy, xx]: continue
        q = [(yy, xx)]; lab[yy, xx] = 1
        touch = False; pts = []
        hd = 0
        while hd < len(q):
            cy, cx = q[hd]; hd += 1; pts.append((cy, cx))
            if cy == 0 or cx == 0 or cy == ih - 1 or cx == iw - 1: touch = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0: continue
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < iw and 0 <= ny < ih and m[ny, nx] == 1 and not lab[ny, nx]:
                        lab[ny, nx] = 1; q.append((ny, nx))
        if touch or len(pts) < 3:
            for (py, px) in pts: m[py, px] = 0
    rowsum = m.sum(axis=1)
    rows = []
    s = None
    for i in range(ih + 1):
        if i < ih and rowsum[i] >= 2:
            if s is None: s = i
            continue
        if s is not None:
            y0, y1 = s, i - 1; s = None
            band = m[y0:y1 + 1, :]
            cols = np.where(band.sum(axis=0) > 0)[0]
            if len(cols) == 0: continue
            x0, x1 = int(cols[0]), int(cols[-1])
            bh2, bw2 = y1 - y0 + 1, x1 - x0 + 1
            if 6 <= bh2 <= ih * 0.35 and bw2 >= 10:
                rows.append((x0, y0, bw2, bh2))
    return rows

def segment_any(lum, dark):
    """v1.5: баллоны на ЛЮБОЙ странице (чёрные l<45, белые l>210; строки-полосы внутри,
    полярность — от типа баллона) + остаточный полностраничный проход (со своей
    полярностью страницы) с отсечкой гигантских склеек арта и дублей баллонов.
    5-й элемент бокса = полярность инверсии яркости."""
    h, w = lum.shape
    out = []
    bubbles = find_bubbles(lum, True) + find_bubbles(lum, False)
    for (x, y, bw, bh, pol) in bubbles:
        ix = min(x + 8, w - 1); iy = min(y + 8, h - 1)
        iw = min(bw - 16, w - ix); ih = min(bh - 16, h - iy)
        if iw < 30 or ih < 16: continue
        sub = lum[iy:iy + ih, ix:ix + iw]
        for (sx, sy, sw2, sh2) in bubble_lines(sub, pol):
            x0 = max(ix, ix + sx - 2); y0 = max(iy, iy + sy - 1)
            x1 = min(ix + iw, ix + sx + sw2 + 2); y1 = min(iy + ih, iy + sy + sh2 + 1)
            out.append((x0, y0, x1 - x0, y1 - y0, pol))
    bubble_boxes = [(bx, by, bbw, bbh) for (bx, by, bbw, bbh, _) in bubbles]
    for (x, y, bw, bh) in segment(lum, dark):
        if bh > h * 0.06 or bw * bh * 12 > w * h: continue
        cx, cy = x + bw // 2, y + bh // 2
        if any(bx <= cx < bx + bbw and by <= cy < by + bbh for (bx, by, bbw, bbh) in bubble_boxes):
            continue
        out.append((x, y, bw, bh, dark))
    # дедуп: бокс из остаточного прохода, почти совпадающий с баллонной строкой, отбрасываем
    ded = []
    for b in sorted(out, key=lambda b: (b[1], b[0])):
        x, y, bw, bh, pol = b
        dup = False
        for d in ded:
            ox, oy = max(x, d[0]), max(y, d[1])
            ix2, iy2 = min(x + bw, d[0] + d[2]), min(y + bh, d[1] + d[3])
            inter = max(0, ix2 - ox) * max(0, iy2 - oy)
            if inter > 0.6 * bw * bh or inter > 0.6 * d[2] * d[3]:
                dup = True; break
        if not dup: ded.append(b)
    return ded

def decode(raw):
    raw = raw[0]
    sb, probs, prev = [], [], -2
    for step in raw:
        mx = step.max()
        e = np.exp(step - mx); ssum = e.sum()
        bi = int(step.argmax())
        if bi in (BLANK, UNK):
            prev = bi; continue
        if bi != prev and bi >= 0:
            sb.append(CHARSET[bi] if bi < len(CHARSET) else '�')
            probs.append(float(e[bi] / ssum))
        prev = bi
    conf = float(np.median(probs)) if probs else 0.0
    t = ' '.join(''.join(sb).split())
    while '……' in t: t = t.replace('……', '…')
    return t, conf

def main(path, model_path='/tmp/r13.onnx', dump_dir=None):
    img = Image.open(path).convert('RGB')
    w0, h0 = img.size
    sc = min(1.0, MAX_DIM / max(w0, h0))
    if sc < 1.0:
        img = img.resize((max(1, round(w0 * sc)), max(1, round(h0 * sc))), Image.LANCZOS)
    page = img
    lum = luma_px(np.asarray(page))
    dark = float(lum.reshape(-1)[::7].mean()) < 100
    print(f"== {path}  {w0}x{h0} -> {page.size[0]}x{page.size[1]}  darkPage={dark}")

    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    boxes = segment_any(lum, dark)
    print(f"lines: {len(boxes)}")
    results = []
    for idx, (x, y, bw, bh, pol) in enumerate(boxes):
        sw = min(bw, page.size[0] - x); sh = min(bh, page.size[1] - y)
        if sw <= 0 or sh <= 0: continue
        new_w = max(8, round(sw * LINE_H / sh)); new_w = min(444, new_w)
        pad_w = math.ceil((new_w + 4) / 8) * 8
        line = Image.new('RGB', (pad_w, LINE_H), (255, 255, 255))
        crop = page.crop((x, y, x + sw, y + sh)).resize((new_w, LINE_H), Image.LANCZOS)
        line.paste(crop, (4, 0))  # левый белый пад 4px, как в тренировочном рендере
        lpix = luma_px(np.asarray(line)).astype(np.float32) / 255.0
        if pol: lpix = 1.0 - lpix
        if dump_dir:
            Image.fromarray((lpix * 255).astype(np.uint8), 'L').save(
                f"{dump_dir}/ln{idx:02d}_{'inv' if pol else 'lit'}.png")
        raw = sess.run(None, {'x': lpix[None, None, :, :]})[0]
        t, conf = decode(raw)
        if t.strip():
            results.append((t, conf))
            print(f"  [{x},{y} {bw}x{bh} {'inv' if pol else '   '}] conf={conf:.2f}  «{t}»")
    good = [c for _, c in results]
    print(f"итого строк: {len(results)}, мед. конф: {np.median(good) if good else 0:.3f}")

if __name__ == '__main__':
    dump = None
    args = sys.argv[1:]
    if '--dump' in args:
        i = args.index('--dump'); dump = args[i + 1]
        import os; os.makedirs(dump, exist_ok=True)
        del args[i:i + 2]
    mdl = args[0] if len(args) > 1 else '/tmp/r13.onnx'
    main(args[-1], mdl, dump)
