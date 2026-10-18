#!/usr/bin/env python3
# Сборка YomiStand_MONO.html из шаблона /tmp/yomi_mono_template.html:
# заменяет маркер __EMBEDDED_JSON__ (внутри <script id="embeddedData" type="application/json">)
# на сырой JSON:
#   {"models":[{"name":"sp_distill","b64":...},{"name":"sp_distill2","b64":...}],
#    "gt":[...], "testset":{"s01":"data:image/jpeg;base64,...", ...}}
import base64, json, pathlib, re, sys

HOME = pathlib.Path('/home/user')
TEMPLATE = pathlib.Path('/tmp/yomi_mono_template.html')
OUT = HOME / 'YomiStand_MONO.html'
MODELS = HOME / 'ocr-web' / 'models'
TESTSET = HOME / 'ocr-web' / 'testset'

def b64(b: bytes) -> str:
    return base64.b64encode(b).decode('ascii')

def main():
    h = TEMPLATE.read_text(encoding='utf-8')
    assert h.count('__EMBEDDED_JSON__') == 1, f'маркер x{h.count("__EMBEDDED_JSON__")}'

    models = []
    for fname, label in (('best_sp_distill.onnx', 'sp_distill'),
                         ('best_sp_distill2.onnx', 'sp_distill2')):
        f = MODELS / fname
        assert f.exists(), f'нет файла {f}'
        raw = f.read_bytes()
        assert raw[:2] == b'\x08\x07', f'{fname} не похож на onnx'
        models.append({'name': label, 'b64': b64(raw)})
        print(f'  {label}: {len(raw)/1e6:.3f} MB')

    gt = json.loads((TESTSET / 'gt20.json').read_text(encoding='utf-8'))
    assert isinstance(gt, dict) and len(gt) == 20, 'gt20.json должен быть словарём s01..s20'
    assert sorted(gt.keys()) == [f's{i:02d}' for i in range(1, 21)], 'странные ключи gt20'

    testset = {}
    for i in range(1, 21):
        name = f's{i:02d}'
        f = TESTSET / f'{name}.jpg'
        mime = 'image/jpeg'
        if not f.exists():
            f = TESTSET / f'{name}.png'
            mime = 'image/png'
        assert f.exists(), f'нет {name}'
        testset[name] = f'data:{mime};base64,' + b64(f.read_bytes())

    emb = {'models': models, 'gt': gt, 'testset': testset}
    payload = json.dumps(emb, ensure_ascii=False, separators=(',', ':'))
    assert '</script' not in payload.lower()
    assert '</script' not in payload

    out = h.replace('__EMBEDDED_JSON__', payload)
    assert '__EMBEDDED_JSON__' not in out
    assert len(OUT.name) > 0
    OUT.write_text(out, encoding='utf-8')
    sz = OUT.stat().st_size
    # хвост файла должен заканчиваться корректно
    tail = out[-200:]
    assert tail.rstrip().endswith('</html>'), f'плохой хвост: {tail[-80:]!r}'
    print(f'OK: {OUT}  {sz/1e6:.3f} MB ({sz} байт)')

if __name__ == '__main__':
    main()
