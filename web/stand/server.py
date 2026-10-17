#!/usr/bin/env python3
"""OCR-стенд: статика + /api/ocr. OCR выполняется настоящим портом ocr_engine_port
(та же сегментация и декодер, что в APK/консоли) — браузер не трогает WASM.
POST /api/ocr  JSON: {"img": "data:image/png;base64,...", "name": "...", "model": "best_sp_distill.onnx"}
GET  /api/models JSON: ["file1.onnx", ...]"""
import os, json, base64, io, glob, sys, math, unicodedata, re
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, 'models')

# --- загрузка порта один раз ---
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np
from PIL import Image
import onnxruntime as ort

port_path = os.path.join(ROOT, 'tools', 'ocr_engine_port.py')
if not os.path.exists(port_path):
    # fallback: ищем в клоне репо
    port_path = '/home/user/ocrchat-work/tools/ocr_engine_port.py'
src = open(port_path).read()
ns = {}
exec(src.split('def main')[0].replace('if __name__', '#'), ns)
luma_px, segment_any, decode = ns['luma_px'], ns['segment_any'], ns['decode']
LINE_H, MAX_DIM = ns['LINE_H'], ns['MAX_DIM']

SESS = {}

def get_sess(name):
    p = os.path.join(MODELS_DIR, os.path.basename(name))
    if not os.path.exists(p):
        raise FileNotFoundError(name)
    if p not in SESS:
        SESS[p] = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
    return SESS[p]

def recognize(img_bgr, sess):
    """как в mine_real: полный пайплайн на PIL'е"""
    w0, h0 = img_bgr.size
    sc = min(1.0, MAX_DIM / max(w0, h0))
    img = img_bgr
    if sc < 1.0:
        img = img.resize((max(1, round(w0 * sc)), round(h0 * sc)), Image.LANCZOS)
    lum = luma_px(np.asarray(img))
    dark = float(lum.reshape(-1)[::7].mean()) < 100
    boxes = segment_any(lum, dark)
    out = []
    for (x, y, bw, bh, pol) in boxes:
        sw = min(bw, img.size[0] - x); sh = min(bh, img.size[1] - y)
        if sw <= 0 or sh <= 0:
            continue
        new_w = max(8, round(sw * LINE_H / sh)); new_w = min(444, new_w)
        pad_w = math.ceil((new_w + 4) / 8) * 8
        line = Image.new('RGB', (pad_w, LINE_H), (255, 255, 255))
        crop = img.crop((x, y, x + sw, y + sh)).resize((new_w, LINE_H), Image.LANCZOS)
        line.paste(crop, (4, 0))
        lpix = luma_px(np.asarray(line)).astype(np.float32) / 255.0
        if pol: lpix = 1.0 - lpix
        raw = sess.run(None, {'x': lpix[None, None, :, :]})[0]
        t, conf = decode(raw)
        if not t:
            continue
        out.append({'x': x, 'y': y, 'w': sw, 'h': sh, 'pol': int(pol), 'text': t, 'conf': round(float(conf), 3)})
    return out, round(img.size[0] * 1.0, 0), round(img.size[1] * 1.0, 0)

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith('/api/models'):
            names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(MODELS_DIR, '*.onnx')))
            self._json(names)
        else:
            super().do_GET()

    def do_POST(self):
        if not self.path.startswith('/api/ocr'):
            self.send_error(404); return
        try:
            n = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(n) or b'{}')
            dataurl = body['img']
            b64 = dataurl.split(',', 1)[1] if ',' in dataurl else dataurl
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')
            sess = get_sess(body.get('model', 'best_sp_distill.onnx'))
            lines, w, h = recognize(img, sess)
            self._json({'lines': lines, 'w': w, 'h': h})
        except Exception as e:
            self._json({'error': str(e)}, 500)

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f'OCR stand server on 0.0.0.0:{port}', flush=True)
    ThreadingHTTPServer(('0.0.0.0', port), Handler).serve_forever()
