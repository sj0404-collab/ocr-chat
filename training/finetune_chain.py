"""Chain fine-tune: input ONNX (torch-gate ort), target font + anti-forget mix,
CER before/after, export ONNX. Same pipeline as finetune.py."""
import torch, torch.nn as nn, numpy as np, random, unicodedata, sys, time, os, argparse
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
LAT_OFF, NC = 111, 163   # BLANK 109 / UNK 110 не сдвигаем: старые onnx и декодеры совместимы; латиница с 111

EWORDS = ('CRACK FLASH BOOM SWITCH BANG SNAP WHOOSH THUD GASP BLINK CREAK POP CLANG DRIP '
          'SCRATCH OH WOW HMPH YEAH HUH COFFEE SUGAR MILK BREAD BUTTER SISTER FATHER MOTHER '
          'SCHOOL RIVER BLACK WHITE GREEN ORANGE VIOLET TODAY TOMORROW NIGHT HEART MOON STORY '
          'FRIEND KING QUEEN SWORD GATE ROSE GLASS HOUSE WORLD MAKE TAKE GIVE WAIT COME LEFT '
          'RIGHT OPEN DOOR LIGHT MACHINE REPAIR ANDROID ENGLISH UNKNOWN QUESTION ANSWER TAP '
          'CHIRP TUK AH HEY YES PLEASE SORRY').split()
EPHRASES = ["WAIT!", "IT'S OVER.", 'No way back…', 'Open the door!', '1 page of 11',
            'Chapter 64. Epilogue', 'Turn it off — NOW.', 'GOOD MORNING, MASTER!',
            'Unknown error #7.', 'Something is wrong here.', 'PLEASE STAND BY…',
            'THE ANSWER IS 42!', 'Single page only.', 'MEET ME AT FIVE?!',
            'Author: Cat & Spices', 'TUK! TUK!', 'CHIRP CHIRP…', 'AH~']

FONTDIR = '/home/user/ocr_train3/fonts'
DEV = []
for f in ['DejaVuSans.ttf', 'DejaVuSans-Bold.ttf', 'DejaVuSansCondensed.ttf',
          'DejaVuSansCondensed-Bold.ttf', 'DejaVuSerif.ttf', 'DejaVuSerif-Bold.ttf',
          'DejaVuSansMono.ttf', 'DejaVuSansMono-Bold.ttf']:
    p = f'/usr/share/fonts/truetype/dejavu/{f}'
    if os.path.exists(p): DEV.append(p)

WORDS = ('дождь город улица брат сестра время ночь день свет тьма меч клинок гора река '
         'дом дверь окно путь след голос шёпот крик тишина враг друг врага сила кровь '
         'капитан князь стражник дух тень душа сердце разум пламя лёд ветер буря звезда '
         'луна солнце небо земля вода огонь камень железо золото правда ложь закон храм '
         'ворота стена башня мост лес поле море пустыня остров флот армия полк битва мир '
         'война месть долг честь страх надежда судьба выбор тайна дар цена утро вечер').split()
PHRASES = [
 'Ты слышал об этом?!', 'Он пришёл на рассвете № 5.', 'ВСЕ ОСТАЛЬНЫЕ ВЫШЛИ НАВСТРЕЧУ!',
 'Женя сказал: «Мир уже не тот»...', '123 подростка, 45 страниц и 0 сомнений!',
 'Не скажу врагу ни слова?', 'Град обрушился на город в 17:45.', 'Чудеса, да и только!..',
 'Ёжик, ты где? Отзовись!', 'Завтра, ровно в полночь…', 'ПРИНИМАЙ МОЙ ПОСЛЕДНИЙ УДАР!',
 'Горы не сойдутся, а люди — сойдутся.', 'Брат, сдавайся! Твоя ци уже истощена.',
 'Значит, всё кончено… 12 лет зря.', 'ГОТЭМ. 2:45 НОЧИ. Город спит — мы нет.',
 'МОЛЧАТЬ! Я сказал — нет!', 'Город не простит тебя, напарник!',''
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

def gen_eng(rng):
    if rng.random() < 0.45: return rng.choice(EPHRASES)
    ws = [rng.choice(EWORDS) for _ in range(rng.randint(1, 4))]
    s = ' '.join(ws)
    if rng.random() < 0.6: s = s.upper()
    elif rng.random() < 0.5: s = s.capitalize()
    if rng.random() < 0.4: s += rng.choice(['!', '?', '…', '.', ''])
    return s[:40]

def gen_line(rng):
    if rng.random() < 0.25: return gen_eng(rng)
    if rng.random() < 0.5: return rng.choice(PHRASES)
    ws = [rng.choice(WORDS) for _ in range(rng.randint(2, 8))]
    s = ' '.join(ws)
    if rng.random() < 0.5: s = s.capitalize()
    if rng.random() < 0.3: s = s.upper()
    if rng.random() < 0.5: s += rng.choice(['.', '!', '?', '…', '—', ''])
    if rng.random() < 0.04:  # дефисный конец строки (перенос в манге: «ВЫ-», «УНИВЕР-»)
        s = s.rstrip(' .,!?…') + '-'
    if rng.random() < 0.04:  # дефис внутри длинного слова (мягкий перенос)
        words = s.split()
        long_w = [i for i, wd in enumerate(words) if len(wd) > 5]
        if long_w:
            i = rng.choice(long_w)
            k = rng.randint(3, len(words[i]) - 2)
            words[i] = words[i][:k] + '-' + words[i][k:]
            s = ' '.join(words)
    if rng.random() < 0.2: s = s.replace(' ', ', ', 1)
    if rng.random() < 0.2: s = f'{rng.randint(1, 999)} {s}'
    return s[:48]

def render(s, rng, fpath):
    f = ImageFont.truetype(fpath, rng.randint(12, 42))
    d0 = ImageDraw.Draw(Image.new('L', (10, 10)))
    bbox = d0.textbbox((0, 0), s, font=f)
    w, h = bbox[2]-bbox[0]+8, bbox[3]-bbox[1]+8
    img = Image.new('L', (w, h), 250); d = ImageDraw.Draw(img)
    ink, bg = rng.randint(5, 70), rng.randint(225, 255)
    img.paste(bg, (0, 0, w, h))
    # glow-аугментация 0.3: серый ореол вокруг тёмных штрихов (~ свечение светлого текста
    # на чёрном баллоне после инверсии яркости в пайплайне)
    if rng.random() < 0.3:
        d.text((4-bbox[0], 4-bbox[1]), s, font=f, fill=ink,
               stroke_width=rng.randint(1, 3), stroke_fill=rng.randint(80, 200))
    else:
        d.text((4-bbox[0], 4-bbox[1]), s, font=f, fill=ink)
    if rng.random() < 0.2:  # вертикальный градиент чернил (декоративный капс титульных)
        arr = np.asarray(img, np.float32) / 255.0
        mask = arr < 0.55
        top = rng.randint(5, 110) / 255.0; bot = rng.randint(120, 240) / 255.0
        if rng.random() < 0.5: top, bot = bot, top
        g = np.linspace(top, bot, img.height, dtype=np.float32)[:, None]
        r2 = np.where(mask, g, arr)
        img = Image.fromarray((r2 * 255).astype(np.uint8))
    if rng.random() < 0.35: img = img.filter(ImageFilter.GaussianBlur(rng.random()*0.8))
    if rng.random() < 0.35: img = img.rotate(rng.uniform(-1.2, 1.2), expand=False, fillcolor=bg)
    if rng.random() < 0.25:  # курсив (аффинный скос ±0.25, титры/звуковые FX манги)
        sh = rng.uniform(-0.25, 0.25)
        img = img.transform((img.width, img.height), Image.AFFINE,
                            (1, sh, 0, 0, 1, 0), fillcolor=bg)
    if rng.random() < 0.15:  # повёрнутые FX-надписи (±8°)
        img = img.rotate(rng.uniform(-8, 8), expand=True, fillcolor=bg)
    if rng.random() < 0.4:  # JPEG-артефакты манга-скриншотов
        import io
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=rng.randint(55, 95))
        buf.seek(0)
        img = Image.open(buf).convert('L')
    nw = max(8, min(440, round(img.width * (32 / img.height))))
    return img.resize((nw, 32), Image.BILINEAR)

def to_canvas(img, rng):
    w = img.width
    paddedW = ((w + 7) // 8) * 8
    if rng.random() < 0.15: paddedW = min(448, paddedW + 8 * rng.randint(1, 6))
    paddedW = min(448, paddedW)
    x = np.ones((1, 32, paddedW), np.float32)
    a = np.asarray(img, np.float32) / 255.0
    x[0, :, :min(w, paddedW)] = a[:, :min(w, paddedW)]
    return x

def pick_font(rng, mix):
    r, acc = rng.random(), 0.0
    for fpath, p in mix:
        acc += p
        if r < acc: return fpath
    return mix[-1][0]

REAL_ITEMS = []  # [(array 1x32xW float32, text)]

def load_real(realdir):
    import json as _json
    recs = _json.load(open(f'{realdir}/real.json'))
    for r in recs:
        img = Image.open(f"{realdir}/images/{r['img']}").convert('L')
        crop = img.crop((r['x'], r['y'], r['x'] + r['w'], r['y'] + r['h']))
        nw = max(8, min(440, round(crop.width * (32 / crop.height))))
        crop = crop.resize((nw, 32), Image.BILINEAR)
        a2 = np.asarray(crop, np.float32) / 255.0
        if r['pol'] == 1: a2 = 1.0 - a2
        REAL_ITEMS.append((a2[None, :, :], r['text']))
    print(f"real: {len(REAL_ITEMS)} lines loaded")

def real_to_canvas(a2, rng):
    w = a2.shape[1]
    paddedW = min(448, ((w + 7) // 8) * 8)
    x = np.ones((1, 32, paddedW), np.float32)
    x[0, :, :min(w, paddedW)] = a2[0, :, :min(w, paddedW)]
    return x

def make_batch(rng, bs, mix, seedlabel=None, real_p=0.0):
    xs, ys, ylens, ws = [], [], [], []
    for _ in range(bs):
        if REAL_ITEMS and rng.random() < real_p:
            a2, s = REAL_ITEMS[rng.randrange(len(REAL_ITEMS))]
            xs.append(real_to_canvas(a2, rng))
        else:
            s = gen_line(rng)
            img = render(s, rng, pick_font(rng, mix))
            xs.append(to_canvas(img, rng))
        lab = [v for v in ((C2I[c] if c in C2I else (LAT_OFF + LAT.index(c) if c in LAT else -1)) for c in s) if v >= 0]
        ys.append(lab); ylens.append(len(lab)); ws.append(xs[-1].shape[2])
    maxw = max(x.shape[2] for x in xs)
    xb = np.ones((len(xs), 1, 32, maxw), np.float32)
    for i, x in enumerate(xs): xb[i, :, :, :x.shape[2]] = x
    return xb, ys, ylens, ws, maxw

def lev(a, b):
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1): cur.append(min(prev[j]+1, cur[-1]+1, prev[j-1]+(ca != cb)))
        prev = cur
    return prev[-1]

def evaluate(net, samples):
    net.eval(); tot, den = 0, 0
    with torch.no_grad():
        for s, x in samples:
            ids = net(torch.from_numpy(x[None]))[0].argmax(-1).numpy()
            out, prev = [], -2
            for i in ids:
                if i == BLANK or i == 110: prev = i; continue
                if i != prev and LAT_OFF <= i < NC: out.append(LAT[i - LAT_OFF])
                if i != prev and 0 <= i < len(CHARSET): out.append(CHARSET[i])
                prev = i
            hyp = ''.join(out).strip()
            a = ' '.join(unicodedata.normalize('NFKC', s).lower().split())
            b = ' '.join(unicodedata.normalize('NFKC', hyp).lower().split())
            d = lev(a, b) / max(1, len(a)); tot += d * len(a); den += len(a)
    net.train()
    return tot / max(1, den)

def build_eval(rng, fontpath, n=100):
    return [(s := gen_line(rng), to_canvas(render(s, rng, fontpath), rng)) for _ in range(n)]

def build_eval_lat(rng, fontpath, n=60):
    return [(s := gen_eng(rng), to_canvas(render(s, rng, fontpath), rng)) for _ in range(n)]

def load_chain(model, path):
    """Load ONNX from this chain (torch-style conv names + onnx::GRU_*/MatMul_*)."""
    m = onnx.load(path)
    W = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    mm = model.state_dict()
    for di in (0, 3, 6, 9, 12):
        mm[f'conv.{di}.weight'] = torch.from_numpy(W[f'conv.{di}.weight'])
        mm[f'conv.{di}.bias'] = torch.from_numpy(W[f'conv.{di}.bias'])
    # GRU tensors by shape: W=(2,384,256) R=(2,384,128) B=(2,768)
    def shape(t): return tuple(W[t].shape)
    gruw = sorted([k for k in W if shape(k) == (2, 384, 256)])
    grur = sorted([k for k in W if shape(k) == (2, 384, 128)])
    grub = sorted([k for k in W if shape(k) == (2, 768)])
    assert len(gruw) == 2 and len(grur) == 2 and len(grub) == 2, (gruw, grur, grub)
    # head: уже-163 (след. циклы) или старый-111 → достраиваем близнецами из кириллицы
    n_in = 163 if [k for k in W if shape(k) == (256, 163)] else 111
    heads = [k for k in W if shape(k) == (256, n_in)]
    hb = [k for k in W if shape(k) == (n_in,)]
    oldW, oldB = W[heads[0]].T.copy(), W[hb[-1]].copy()
    if n_in == 111:
        TW = {'A':'А','a':'а','B':'В','C':'С','c':'с','E':'Е','e':'е','H':'Н','h':'н','I':'І','i':'і',
              'K':'К','k':'к','M':'М','m':'м','O':'О','o':'о','P':'Р','p':'р','R':'Р','T':'Т','t':'т',
              'W':'ш','X':'Х','x':'х','Y':'У','y':'у','Z':'З','z':'з'}
        rngn = np.random.default_rng(123).uniform
        sd = max(1e-3, float(oldW.std()))
        newW = np.empty((163, 256), np.float32); newB = np.empty((163,), np.float32)
        newW[:111], newB[:111] = oldW, oldB
        randn = 0
        for i, ch in enumerate(LAT):
            dst = LAT_OFF + i; tw = TW.get(ch)
            if tw and tw in C2I:
                newW[dst], newB[dst] = oldW[C2I[tw]], oldB[C2I[tw]]
            else:
                newW[dst] = rngn(-sd, sd, 256).astype(np.float32); newB[dst] = 0; randn += 1
        oldW, oldB = newW, newB
        print(f'head extend 111→163: twins={52 - randn} rand={randn}', flush=True)
    mm['head.weight'] = torch.from_numpy(oldW)
    mm['head.bias'] = torch.from_numpy(oldB)
    def put(gru, wk, rk, bk, gates_mode):
        H, S = 128, [3*128]
        w, r, b = W[wk], W[rk], W[bk]
        def rear(t, first=True):  # onnx(z,r,n) → torch(r,z,n)
            return np.concatenate([t[H:2*H], t[0:H], t[2*H:3*H]], axis=0) if first else t
        for d in range(2):
            sfx = '' if d == 0 else '_reverse'
            mm[f'{gru}.weight_ih_l0{sfx}'] = torch.from_numpy(rear(w[d]).copy())
            mm[f'{gru}.weight_hh_l0{sfx}'] = torch.from_numpy(rear(r[d]).copy())
            mm[f'{gru}.bias_ih_l0{sfx}'] = torch.from_numpy(rear(b[d][:3*H]).copy())
            mm[f'{gru}.bias_hh_l0{sfx}'] = torch.from_numpy(rear(b[d][3*H:6*H]).copy())
    # auto-detect gate order via equivalence
    for mode in ('rearr', 'direct'):
        test = {k: v for k, v in mm.items()}
        model.load_state_dict(mm)
        put('gru1', gruw[0], grur[0], grub[0], mode)
        put('gru2', gruw[1], grur[1], grub[1], mode)
        model.load_state_dict(mm)
        model.eval()
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        x = np.random.rand(1, 1, 32, 64).astype(np.float32)
        ref = sess.run(None, {'x': x})[0]
        with torch.no_grad(): out = model(torch.from_numpy(x)).numpy()
        n_cmp = min(out.shape[-1], ref.shape[-1])
        d = np.abs(out.reshape(-1, n_cmp) - ref.reshape(-1, ref.shape[-1])[:, :n_cmp].reshape(-1, n_cmp)).max()
        if d < 1e-4:
            print(f'load_chain gates={mode} OK (max|diff|={d:.2e})', flush=True)
            break
        print(f'load_chain gates={mode} mismatch {d:.3e}, retry…', flush=True)
    else:
        raise RuntimeError('gate order undetected')
    model.train()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--font', required=True); ap.add_argument('--p', type=float, default=0.45)
    ap.add_argument('--prev', nargs='*', default=[], help='anti-forget fonts (each p=(1-p_target-0.1)/n)')
    ap.add_argument('--steps', type=int, default=900); ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--realdir', default='', help='папка с real.json + images/')
    ap.add_argument('--realp', type=float, default=0.25, help='доля реальных строк в батче')
    ap.add_argument('--tag', default='stage'); ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    if a.realdir: load_real(a.realdir)
    torch.set_num_threads(2); torch.manual_seed(a.seed)
    mix = [(a.font, a.p)]
    if a.prev:
        pp = (1.0 - a.p - 0.1) / len(a.prev)
        mix += [(f, pp) for f in a.prev]
    mix.append(('DEV', 0.1))
    mix = [(random.choice(DEV) if f == 'DEV' else f, p) for f, p in mix]
    print('MIX:', [(os.path.basename(f), round(p, 3)) for f, p in mix], flush=True)
    net = load_chain(CRNN(), a.inp)
    # eval sets
    evg = random.Random(7)
    e_t = build_eval(evg, a.font, 120); e_d = build_eval(evg, random.choice(DEV), 120)
    e_mix = build_eval(evg, a.font, 60) + build_eval(evg, random.choice(DEV), 60)
    e_lat = build_eval_lat(evg, a.font)
    print(f'[{a.tag}] BEFORE: font={evaluate(net, e_t):.4f} dejavu={evaluate(net, e_d):.4f} lat={evaluate(net, e_lat):.4f}', flush=True)
    rng = random.Random(a.seed)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-5, weight_decay=1e-5)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        xb, ys, ylens, ws, maxw = make_batch(rng, a.bs, mix, real_p=a.realp)
        logit = net(torch.from_numpy(xb)).permute(1, 0, 2).log_softmax(-1)
        T = logit.shape[0]
        lens = torch.tensor([max(1, min(T, round(T * ws[i] / maxw))) for i in range(a.bs)])
        loss = ctc(logit, torch.tensor([c for l in ys for c in l]), lens, torch.tensor(ylens))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        if step % 100 == 0:
            print(f'[{a.tag}] step {step}/{a.steps} loss={loss.item():.3f} ({(time.time()-t0)/step*1000:.0f}ms)', flush=True)
        if step % 300 == 0:
            print(f'[{a.tag}] eval@{step}: font={evaluate(net, e_t):.4f} dejavu={evaluate(net, e_d):.4f}', flush=True)
    print(f'[{a.tag}] AFTER: font={evaluate(net, e_t):.4f} dejavu={evaluate(net, e_d):.4f} mixed={evaluate(net, e_mix):.4f}', flush=True)
    net.eval()
    torch.onnx.export(net, torch.ones(1, 1, 32, 128), a.out,
                      input_names=['x'], output_names=['logit'],
                      dynamic_axes={'x': {3: 'width'}, 'logit': {0: 'T'}},
                      opset_version=14, dynamo=False)
    print(f'[{a.tag}] EXPORTED {a.out} ({os.path.getsize(a.out)} B)', flush=True)
    # sanity load exported
    sess = ort.InferenceSession(a.out, providers=['CPUExecutionProvider'])
    y = sess.run(None, {'x': np.zeros((1, 1, 32, 64), np.float32)})[0]
    print(f'[{a.tag}] EXPORT OK shape={y.shape}', flush=True)

if __name__ == '__main__':
    main()
