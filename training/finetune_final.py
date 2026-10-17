"""Финальное дообучение: все шрифты равномерно + GT-фрагменты testset20 (словарь/фразы
реальных скринов) + расширенные аугментации (обводка, рваные штрихи, screentone, низкий
размер, шум, градиент). Периодический eval на 6 сложных скринах (tools/eval20.py --quick),
лучшая модель по SOFT сохраняется в --out. Ранняя остановка при SCORE=1.000 или --timelimit.

Использование:
  python3 finetune_final.py --inp base.onnx --out best.onnx \
      --fonts fonts/*.ttf fonts_extra/*.ttf --steps 20000 --evalevery 3000 --timelimit 11400
"""
import torch, torch.nn as nn, numpy as np, random, unicodedata, sys, time, os, argparse, glob, re, subprocess, shutil, json
sys.path.insert(0, '/home/user/ocr_test2')
from crnn_torch import CRNN
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import onnx, onnxruntime as ort
from onnx import numpy_helper

CHARSET = ("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
           "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
           "ЇїІіЄєҐґ"
           " .,!?;:\"'«»„“”()[]—–-…№*/0123456789")
BLANK, C2I = 109, {c: i for i, c in enumerate(CHARSET)}
LAT = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
LAT_OFF, NC = 111, 163
ALLOWED = set(CHARSET) | set(LAT)

DEV = [f'/usr/share/fonts/truetype/dejavu/{f}' for f in
       ('DejaVuSans.ttf', 'DejaVuSans-Bold.ttf', 'DejaVuSerif.ttf', 'DejaVuSerif-Bold.ttf',
        'DejaVuSansCondensed.ttf', 'DejaVuSansCondensed-Bold.ttf')]
DEV = [f for f in DEV if os.path.exists(f)]

WORDS = ('госпожа императрица замок бальный купол жемчуг бархат призрак маска вальс лёд '
         'крыло кошмар проклятие клятва оборотень магия свиток трон гвардия шпион её высочество '
         'дом дверь окно путь след голос шёпот крик тишина враг друг сила кровь учёба школа '
         'капитан князь стражник дух тень душа сердце разум пламя лёд ветер буря звезда '
         'луна солнце небо земля вода огонь камень железо золото правда ложь закон храм '
         'ворота стена башня мост лес поле море пустыня остров флот армия полк битва мир '
         'война месть долг честь страх надежда судьба выбор тайна дар цена утро вечер '
         'голубка птичка окошко утречко работа отец мама брат сестра герой злодей слабак').split()
PHRASES = [
 'Ты слышал об этом?!', 'Он пришёл на рассвете № 5.', 'ВСЕ ОСТАЛЬНЫЕ ВЫШЛИ НАВСТРЕЧУ!',
 'Женя сказал: «Мир уже не тот»...', '123 подростка, 45 страниц и 0 сомнений!',
 'Не скажу врагу ни слова?', 'Град обрушился на город в 17:45.', 'Чудеса, да и только!..',
 'Ёжик, ты где? Отзовись!', 'Завтра, ровно в полночь…', 'ПРИНИМАЙ МОЙ ПОСЛЕДНИЙ УДАР!',
 'Горы не сойдутся, а люди — сойдутся.', 'Брат, сдавайся! Твоя ци уже истощена.',
 'Значит, всё кончено… 12 лет зря.', 'ГОТЭМ. 2:45 НОЧИ. Город спит — мы нет.',
 'МОЛЧАТЬ! Я сказал — нет!', 'Город не простит тебя, напарник!',
 'Ваше высочество прибыло раньше срока.', 'Ты единственный, кому я верю.',
 'Будь осторожен в саду: там ночью кто-то ходит.', 'Дождь не кончается…',
 'Слушай, старший брат, я всё понял.', 'Он совсем не изменился…', '…Прости.',
 'АГРХ?! Ну и ну!', 'Внимание… тихо!', 'Сегодня решается ВСЁ.', 'Оружие к ногам, живо!',
 'Ты что, оглох?! Марш вперёд!', 'Глава 12. Начало конца', 'Э-э… а ты уверен?',
 'Хе-хе… наивный.', 'Так вот оно что…', 'НЕ СМЕЙ прикасаться к ней!',
 'Если мы проиграем — всё кончено.', 'Уходим. Быстро. Сейчас же.',
 'Внимание! Объект № 34 пропал.', 'Стой! Кто идёт?!', 'Пароль: «Север-7».',
 'Беги, не оглядывайся!!', 'Я знал это с самого начала…',
]
EWORDS = ('CRACK FLASH BOOM SWITCH BANG SNAP WHOOSH THUD GASP BLINK CREAK POP CLANG DRIP '
          'TAP CHIRP TUK AH HEY YES PLEASE SORRY SAVE CONFIG MOD VAULT PAGE GUEST SAVE').split()
EPHRASES = ["WAIT!", "IT'S OVER.", 'No way back…', 'Open the door!', '1 page of 11',
            'Chapter 64. Epilogue', 'Turn it off — NOW.', 'mieruko-chan', 'mangabuff.ru',
            'AteRa', 'DZX', 'NSZ', 'LADYBUG', 'config.ini', 'save', 'Switch', 'GOOD MORNING!']

GT_LINES, GT_VOCAB = [], []
def load_gt(path):
    gt = json.load(open(path))
    for lines in gt.values():
        for s in lines:
            s = sanitize(' '.join(unicodedata.normalize('NFKC', s).split()))
            if not s: continue
            GT_LINES.append(s)
            for w in s.split():
                w2 = w.strip('.,!?…-«»()[]')
                if len(w2) > 2: GT_VOCAB.append(w2)
    print(f'gt: {len(GT_LINES)} lines, vocab +{len(set(GT_VOCAB))} words', flush=True)

def sanitize(s):
    return ''.join(c for c in s if c in ALLOWED)

def gen_eng(rng):
    if rng.random() < 0.45: return rng.choice(EPHRASES)
    s = ' '.join(rng.choice(EWORDS) for _ in range(rng.randint(1, 4)))
    if rng.random() < 0.6: s = s.upper()
    elif rng.random() < 0.5: s = s.capitalize()
    if rng.random() < 0.4: s += rng.choice(['!', '?', '…', '.', ''])
    return s[:40]

def gen_gt_frag(rng):
    s = rng.choice(GT_LINES)
    ws = s.split()
    if len(ws) > 3:
        n = rng.randint(1, min(9, len(ws)))
        i = rng.randrange(0, len(ws) - n + 1)
        s = ' '.join(ws[i:i + n])
    r = rng.random()
    if r < 0.35: s = s.upper()
    elif r < 0.5: s = s.capitalize()
    if rng.random() < 0.08 and len(s) > 4: s = s.rstrip(' .,!?…') + '-'
    return s[:44] or rng.choice(PHRASES)

def gen_line(rng):
    r = rng.random()
    if r < 0.15: return gen_eng(rng)
    if GT_LINES and r < 0.50: return gen_gt_frag(rng)
    if r < 0.70: return rng.choice(PHRASES)
    pool = WORDS + (GT_VOCAB if rng.random() < 0.4 else [])
    ws = [rng.choice(pool) for _ in range(rng.randint(2, 8))]
    s = ' '.join(ws)
    if rng.random() < 0.5: s = s.capitalize()
    if rng.random() < 0.3: s = s.upper()
    if rng.random() < 0.5: s += rng.choice(['.', '!', '?', '…', '—', ''])
    if rng.random() < 0.05: s = s.rstrip(' .,!?…') + '-'
    if rng.random() < 0.2: s = s.replace(' ', ', ', 1)
    if rng.random() < 0.2: s = f'{rng.randint(1, 999)} {s}'
    return s[:48]

def screentone(w, h, rng):
    """сетка точек (тон ретуши манги) как numpy-маска яркости 0.75..0.98"""
    arr = np.ones((h, w), np.float32)
    step = rng.randint(3, 6); off = rng.randint(0, step)
    val = rng.uniform(0.75, 0.98)
    for yy in range(off % step, h, step):
        for xx in range((off + (yy // step) % 2 * (step // 2)) % step, w, step):
            arr[yy, xx] = val
    return arr

def render(s, rng, fpath):
    f = ImageFont.truetype(fpath, rng.randint(11, 46))
    d0 = ImageDraw.Draw(Image.new('L', (10, 10)))
    bbox = d0.textbbox((0, 0), s, font=f)
    w, h = bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8
    ink, bg = rng.randint(5, 80), rng.randint(215, 255)
    img = Image.new('L', (w, h), bg); d = ImageDraw.Draw(img)
    x0, y0 = 4 - bbox[0], 4 - bbox[1]
    r = rng.random()
    if r < 0.28:      # контрастная обводка (светлая рубашка тёмного шрифта и наоборот)
        sw = rng.randint(1, 3)
        sf = bg if ink < bg // 2 else rng.randint(0, 90)
        d.text((x0, y0), s, font=f, fill=ink, stroke_width=sw, stroke_fill=sf)
        if rng.random() < 0.4:  # вторая тонкая внешняя обводка (декоративные титульные)
            d.text((x0, y0), s, font=f, fill=ink, stroke_width=1,
                   stroke_fill=(ink if sf == bg else bg))
    elif r < 0.45:    # glow-ореол
        d.text((x0, y0), s, font=f, fill=ink,
               stroke_width=rng.randint(1, 3), stroke_fill=rng.randint(90, 210))
    elif r < 0.60:    # жирный дубль-ход (титульный капс)
        d.text((x0, y0), s, font=f, fill=ink)
        d.text((x0 + 1, y0), s, font=f, fill=ink)
    else:
        d.text((x0, y0), s, font=f, fill=ink)
    arr = np.asarray(img, np.float32) / 255.0
    if rng.random() < 0.28:  # рваные штрихи: эрозия/дилатация + мелкие дыры
        k = rng.choice([ImageFilter.MinFilter(3), ImageFilter.MaxFilter(3)])
        if rng.random() < 0.5: img = img.filter(k)
        a = np.asarray(img, np.float32) / 255.0
        holes = np.random.default_rng(rng.randrange(1 << 30)).random(a.shape) < rng.uniform(0.002, 0.012)
        a = np.where(holes, bg / 255.0, a)
        arr = a
    if rng.random() < 0.22:  # screentone под текст
        arr = np.where(arr < 0.6, arr, arr * screentone(w, h, rng))
    if rng.random() < 0.20:  # вертикальный градиент чернил
        mask = arr < 0.6
        top = rng.randint(5, 110) / 255.0; bot = rng.randint(120, 240) / 255.0
        if rng.random() < 0.5: top, bot = bot, top
        g = np.linspace(top, bot, h, dtype=np.float32)[:, None]
        arr = np.where(mask, g, arr)
    if rng.random() < 0.30:  # соль-перец / грязь скана
        nr = np.random.default_rng(rng.randrange(1 << 30))
        arr = np.clip(arr + nr.normal(0, rng.uniform(0.01, 0.045), arr.shape), 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    if rng.random() < 0.20:  # низкое разрешение (апскейл скринов)
        img = img.resize((max(6, int(w * 0.66)), max(6, int(h * 0.66))), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
    if rng.random() < 0.35: img = img.filter(ImageFilter.GaussianBlur(rng.random() * 0.9))
    if rng.random() < 0.35: img = img.rotate(rng.uniform(-1.4, 1.4), expand=False, fillcolor=bg)
    if rng.random() < 0.28:  # курсив-скос
        sh = rng.uniform(-0.28, 0.28)
        img = img.transform((img.width, img.height), Image.AFFINE, (1, sh, 0, 0, 1, 0), fillcolor=bg)
    if rng.random() < 0.13:  # повёрнутые FX
        img = img.rotate(rng.uniform(-8, 8), expand=True, fillcolor=bg)
    if rng.random() < 0.42:  # JPEG
        import io
        buf = io.BytesIO(); img.save(buf, 'JPEG', quality=rng.randint(45, 95)); buf.seek(0)
        img = Image.open(buf).convert('L')
    nw = max(8, min(440, round(img.width * (32 / img.height))))
    return img.resize((nw, 32), Image.BILINEAR)

def to_canvas(img, rng):
    w = img.width
    paddedW = ((w + 7) // 8) * 8
    if rng.random() < 0.15: paddedW = min(448, paddedW + 8 * rng.randint(1, 6))
    x = np.ones((1, 32, paddedW), np.float32)
    a = np.asarray(img, np.float32) / 255.0
    x[0, :, :min(w, paddedW)] = a[:, :min(w, paddedW)]
    return x

def pick_font(rng, fonts):
    return rng.choice(fonts)

MINED_ITEMS = []  # [(array 32xW float32, text)] — реальные кропы со скринов с точными метками
def load_mined(path):
    d = np.load(path, allow_pickle=True)
    texts = d['texts']
    n = 0
    for i in range(len(texts)):
        a2 = d[f'img{i}'].astype(np.float32) / 255.0
        s = sanitize(' '.join(unicodedata.normalize('NFKC', str(texts[i])).split()))
        if a2.shape[0] != 32 or a2.shape[1] < 10 or len(s) < 2: continue
        MINED_ITEMS.append((a2, s)); n += 1
    print(f'mined: {n} real lines loaded', flush=True)

def mined_to_canvas(a2, rng):
    w = a2.shape[1]
    paddedW = min(448, ((w + 7) // 8) * 8)
    if rng.random() < 0.15: paddedW = min(448, paddedW + 8 * rng.randint(1, 6))
    x = np.ones((1, 32, paddedW), np.float32)
    n = min(w, paddedW)
    x[0, :, :n] = a2[:, :n]
    return x

def make_batch(rng, bs, fonts, minedp=0.0):
    xs, ys, ylens, ws = [], [], [], []
    for _ in range(bs):
        if MINED_ITEMS and rng.random() < minedp:
            a2, s = MINED_ITEMS[rng.randrange(len(MINED_ITEMS))]
            xs.append(mined_to_canvas(a2, rng))
        else:
            s = gen_line(rng)
            img = render(s, rng, pick_font(rng, fonts))
            xs.append(to_canvas(img, rng))
        lab = [v for v in ((C2I[c] if c in C2I else (LAT_OFF + LAT.index(c) if c in LAT else -1)) for c in s) if v >= 0]
        ys.append(lab); ylens.append(len(lab)); ws.append(xs[-1].shape[2])
    maxw = max(x.shape[2] for x in xs)
    xb = np.ones((len(xs), 1, 32, maxw), np.float32)
    for i, x in enumerate(xs): xb[i, :, :, :x.shape[2]] = x
    return xb, ys, ylens, ws, maxw

def lev(a, b):
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def decode_ids(ids):
    out, prev = [], -2
    for i in ids:
        if i == BLANK or i == 110: prev = i; continue
        if i != prev and LAT_OFF <= i < NC: out.append(LAT[i - LAT_OFF])
        if i != prev and 0 <= i < len(CHARSET): out.append(CHARSET[i])
        prev = i
    return ''.join(out).strip()

def evaluate(net, samples):
    net.eval(); tot, den = 0, 0
    with torch.no_grad():
        for s, x in samples:
            hyp = decode_ids(net(torch.from_numpy(x[None]))[0].argmax(-1).numpy())
            a = ' '.join(unicodedata.normalize('NFKC', s).lower().split())
            b = ' '.join(unicodedata.normalize('NFKC', hyp).lower().split())
            d = lev(a, b) / max(1, len(a)); tot += d * len(a); den += len(a)
    net.train()
    return tot / max(1, den)

def build_eval(rng, fonts, n=100, gen=None):
    gen = gen or gen_line
    return [(s := gen(rng), to_canvas(render(s, rng, rng.choice(fonts)), rng)) for _ in range(n)]

def export_onnx(net, path):
    net.eval()
    torch.onnx.export(net, torch.ones(1, 1, 32, 128), path,
                      input_names=['x'], output_names=['logit'],
                      dynamic_axes={'x': {3: 'width'}, 'logit': {0: 'T'}},
                      opset_version=14, dynamo=False)
    net.train()

def quick_eval(root, onnx_path):
    try:
        out = subprocess.run([sys.executable, os.path.join(root, 'tools', 'eval20.py'), onnx_path, '--quick'],
                             capture_output=True, text=True, timeout=1500).stdout
        m = re.search(r'SCORE=([\d.]+) SOFT=([\d.]+) HIT=(\d+)/(\d+)', out)
        if m: return float(m.group(1)), float(m.group(2)), m.group(3), m.group(4), out
    except Exception as e:
        print('quick_eval err:', e, flush=True)
    return 0.0, 0.0, '0', '0', ''

def load_chain(model, path):
    m = onnx.load(path)
    W = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    mm = model.state_dict()
    for di in (0, 3, 6, 9, 12):
        mm[f'conv.{di}.weight'] = torch.from_numpy(W[f'conv.{di}.weight'])
        mm[f'conv.{di}.bias'] = torch.from_numpy(W[f'conv.{di}.bias'])
    def shape(t): return tuple(W[t].shape)
    gruw = sorted([k for k in W if shape(k) == (2, 384, 256)])
    grur = sorted([k for k in W if shape(k) == (2, 384, 128)])
    grub = sorted([k for k in W if shape(k) == (2, 768)])
    assert len(gruw) == 2 and len(grur) == 2 and len(grub) == 2
    n_in = 163 if [k for k in W if shape(k) == (256, 163)] else 111
    heads = [k for k in W if shape(k) == (256, n_in)]
    hb = [k for k in W if shape(k) == (n_in,)]
    oldW, oldB = W[heads[0]].T.copy(), W[hb[-1]].copy()
    if n_in == 111:
        TW = {'A': 'А', 'a': 'а', 'B': 'В', 'C': 'С', 'c': 'с', 'E': 'Е', 'e': 'е', 'H': 'Н', 'h': 'н', 'I': 'І', 'i': 'і',
              'K': 'К', 'k': 'к', 'M': 'М', 'm': 'м', 'O': 'О', 'o': 'о', 'P': 'Р', 'p': 'р', 'R': 'Р', 'T': 'Т', 't': 'т',
              'W': 'ш', 'X': 'Х', 'x': 'х', 'Y': 'У', 'y': 'у', 'Z': 'З', 'z': 'з'}
        rngn = np.random.default_rng(123).uniform
        sd = max(1e-3, float(oldW.std()))
        newW = np.empty((163, 256), np.float32); newB = np.empty((163,), np.float32)
        newW[:111], newB[:111] = oldW, oldB
        for i, ch in enumerate(LAT):
            dst = LAT_OFF + i; tw = TW.get(ch)
            if tw and tw in C2I:
                newW[dst], newB[dst] = oldW[C2I[tw]], oldB[C2I[tw]]
            else:
                newW[dst] = rngn(-sd, sd, 256).astype(np.float32); newB[dst] = 0
        oldW, oldB = newW, newB
    mm['head.weight'] = torch.from_numpy(oldW)
    mm['head.bias'] = torch.from_numpy(oldB)
    def put(gru, wk, rk, bk):
        H = 128
        w, r, b = W[wk], W[rk], W[bk]
        def rear(t): return np.concatenate([t[H:2 * H], t[0:H], t[2 * H:3 * H]], axis=0)
        for d in range(2):
            sfx = '' if d == 0 else '_reverse'
            mm[f'{gru}.weight_ih_l0{sfx}'] = torch.from_numpy(rear(w[d]).copy())
            mm[f'{gru}.weight_hh_l0{sfx}'] = torch.from_numpy(rear(r[d]).copy())
            mm[f'{gru}.bias_ih_l0{sfx}'] = torch.from_numpy(rear(b[d][:3 * H]).copy())
            mm[f'{gru}.bias_hh_l0{sfx}'] = torch.from_numpy(rear(b[d][3 * H:6 * H]).copy())
    for mode in ('rearr', 'direct'):
        model.load_state_dict(mm)
        put('gru1', gruw[0], grur[0], grub[0]); put('gru2', gruw[1], grur[1], grub[1])
        model.load_state_dict(mm); model.eval()
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        x = np.random.rand(1, 1, 32, 64).astype(np.float32)
        ref = sess.run(None, {'x': x})[0]
        with torch.no_grad(): out = model(torch.from_numpy(x)).numpy()
        o = out.reshape(-1, out.shape[-1]); r_ = ref.reshape(-1, ref.shape[-1])
        n_cmp = min(o.shape[1], r_.shape[1])
        d = np.abs(o[:, :n_cmp] - r_[:, :n_cmp]).max()
        if d < 1e-4:
            print(f'load_chain OK (max|diff|={d:.2e})', flush=True); break
        print(f'load_chain mismatch {d:.3e}, retry…', flush=True)
    else:
        raise RuntimeError('gate order undetected')
    model.train()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--fonts', nargs='+', required=True)
    ap.add_argument('--gtfile', default='')
    ap.add_argument('--steps', type=int, default=20000); ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--evalevery', type=int, default=3000)
    ap.add_argument('--mined', default='', help='npz с реальными кропами (tools/mine_real.py)')
    ap.add_argument('--minedp', type=float, default=0.22, help='доля mined-строк в батче')
    ap.add_argument('--timelimit', type=int, default=0, help='секунд; 0 = без лимита')
    ap.add_argument('--lr', type=float, default=6e-5)
    ap.add_argument('--tag', default='final'); ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fonts = [f for f in a.fonts for f2 in (glob.glob(f) or [f]) if os.path.exists(f2)]
    assert fonts, 'no fonts found'
    if not a.gtfile: a.gtfile = os.path.join(root, 'testset20', 'gt20.json')
    if os.path.exists(a.gtfile): load_gt(a.gtfile)
    if a.mined and os.path.exists(a.mined): load_mined(a.mined)
    torch.set_num_threads(2); torch.manual_seed(a.seed)
    print('FONTS:', len(fonts), [os.path.basename(f) for f in fonts][:24], flush=True)
    # DEV добавляем с удвоенным весом к списку (системные sans/serif полезны для UI-строчек)
    mix_fonts = fonts + DEV + DEV
    net = load_chain(CRNN(), a.inp)
    evg = random.Random(7)
    e_t = build_eval(evg, mix_fonts, 160)
    e_lat = build_eval(evg, mix_fonts, 80, gen=gen_eng)
    e_gt = build_eval(evg, mix_fonts, 120, gen=gen_gt_frag) if GT_LINES else e_t
    print(f'[{a.tag}] BEFORE: synth={evaluate(net, e_t):.4f} lat={evaluate(net, e_lat):.4f} gtfrag={evaluate(net, e_gt):.4f}', flush=True)
    rng = random.Random(a.seed)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr * 0.15)
    t0 = time.time()
    best_soft, best_path = -1.0, os.path.join(os.path.dirname(a.out) or '.', '_fbest.onnx')
    if os.path.exists(best_path): os.remove(best_path)
    stop = False
    for step in range(1, a.steps + 1):
        xb, ys, ylens, ws, maxw = make_batch(rng, a.bs, mix_fonts, minedp=a.minedp)
        logit = net(torch.from_numpy(xb)).permute(1, 0, 2).log_softmax(-1)
        T = logit.shape[0]
        lens = torch.tensor([max(1, min(T, round(T * ws[i] / maxw))) for i in range(a.bs)])
        loss = ctc(logit, torch.tensor([c for l in ys for c in l]), lens, torch.tensor(ylens))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
        if step % 100 == 0:
            print(f'[{a.tag}] step {step}/{a.steps} loss={loss.item():.3f} ({(time.time() - t0) / step * 1000:.0f}ms)', flush=True)
        if step % 1000 == 0:
            print(f'[{a.tag}] cer@{step}: synth={evaluate(net, e_t):.4f} lat={evaluate(net, e_lat):.4f} gtfrag={evaluate(net, e_gt):.4f}', flush=True)
        if a.evalevery and step % a.evalevery == 0:
            export_onnx(net, best_path + '.tmp')
            sc, so, h, n, log = quick_eval(root, best_path + '.tmp')
            print(f'[{a.tag}] SCREEN@{step}: score={sc:.3f} soft={so:.3f} hit={h}/{n} best={best_soft:.3f}', flush=True)
            for ln in log.strip().splitlines():
                if ln.startswith('s'): print('   ', ln, flush=True)
            if so > best_soft:
                best_soft = so; shutil.copy(best_path + '.tmp', best_path)
            os.remove(best_path + '.tmp')
            if sc >= 0.999:
                print(f'[{a.tag}] TARGET REACHED score=1.000 @ {step}', flush=True); stop = True
        if stop or (a.timelimit and time.time() - t0 > a.timelimit):
            print(f'[{a.tag}] STOP @ {step} (target={stop})', flush=True); break
    if best_soft < 0:  # eval не был включён — экспортируем последнюю
        export_onnx(net, best_path)
    shutil.copy(best_path, a.out)
    print(f'[{a.tag}] EXPORTED {a.out} ({os.path.getsize(a.out)} B, best_soft={best_soft:.3f})', flush=True)
    # финальный ПОЛНЫЙ прогон 20 скринов
    out = subprocess.run([sys.executable, os.path.join(root, 'tools', 'eval20.py'), a.out, '--fail'],
                         capture_output=True, text=True, timeout=3600).stdout
    print('===== FULL EVAL 20 =====', flush=True)
    print(out, flush=True)

if __name__ == '__main__':
    main()
