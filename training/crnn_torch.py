import torch, torch.nn as nn
import onnx
from onnx import numpy_helper
import numpy as np

class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),      # 32→16, W/2
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),     # 16→8, W/4
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)),  # 8→4
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)), # 4→2
            nn.Conv2d(128, 128, (3, 2), 1, (1, 0)), nn.ReLU(),             # H:2→2? pad(1,0): (2+2-3)/1+1=2… точно как ONNX
        )
        self.gru1 = nn.GRU(256, 128, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(256, 128, bidirectional=True, batch_first=True)
        self.head = nn.Linear(256, 111)

    def forward(self, x):              # x: (1,1,32,W) → (1,T,111)
        f = self.conv(x)               # (1,128,H',W')
        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, w, c * h)  # (1,W',256)
        f, _ = self.gru1(f)
        f, _ = self.gru2(f)
        return self.head(f)            # (1,W',111)

def load_onnx_weights(model, path):
    m = onnx.load(path)
    W = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    mm = model.state_dict()
    convs = [('conv.0', 'convs.0'), ('conv.3', 'convs.3')]
    src_convs = ['conv.0', 'conv.3', 'onnx::Conv_364', 'onnx::Conv_367', 'onnx::Conv_370']
    dst_idx = [0, 3, 6, 9, 12]
    for src, di in zip(src_convs, dst_idx):
        mm[f'conv.{di}.weight'] = torch.from_numpy(W[f'{src}.weight' if f'{src}.weight' in W else src])
        mm[f'conv.{di}.bias'] = torch.from_numpy(W[f'{src}.bias' if f'{src}.bias' in W else
              ('onnx::Conv_365' if src=='onnx::Conv_364' else 'onnx::Conv_368' if src=='onnx::Conv_367' else 'onnx::Conv_371')])
    # GRU: ONNX gates (z,r,n) → torch (r,z,n); W (2,3H,in), R (2,3H,H), B (2,6H)
    for li, key in ((1, '41'), (2, '45')):
        pass
    for torch_name, onn_w, onn_r, onn_b in (
        ('gru1', 'onnx::GRU_415', 'onnx::GRU_416', 'onnx::GRU_414'),
        ('gru2', 'onnx::GRU_458', 'onnx::GRU_459', 'onnx::GRU_457')):
        gru = getattr(model, torch_name)
        H = gru.hidden_size
        for d in range(2):
            w = W[onn_w][d]; r = W[onn_r][d]; b = W[onn_b][d]
            perm = [H, 2*H, 0*H]  # z,r,n → r,z,n: берём блоки в порядке r(H..2H), z(0..H), n(2H..3H)
            def rearr(t): return np.concatenate([t[H:2*H], t[0:H], t[2*H:3*H]], axis=0)
            mm[f'{torch_name}.weight_ih_l0' + ('_reverse' if d else '')] = torch.from_numpy(rearr(w).copy())
            mm[f'{torch_name}.weight_hh_l0' + ('_reverse' if d else '')] = torch.from_numpy(rearr(r).copy())
            b_ih, b_hh = b[:3*H], b[3*H:6*H]
            mm[f'{torch_name}.bias_ih_l0' + ('_reverse' if d else '')] = torch.from_numpy(rearr(b_ih).copy())
            mm[f'{torch_name}.bias_hh_l0' + ('_reverse' if d else '')] = torch.from_numpy(rearr(b_hh).copy())
    mm['head.weight'] = torch.from_numpy(W['onnx::MatMul_460'].T.copy())
    mm['head.bias'] = torch.from_numpy(W['head.bias'])
    model.load_state_dict(mm)
    return model

if __name__ == '__main__':
    torch.manual_seed(0)
    net = load_onnx_weights(CRNN().eval(), '/home/user/yomikai-ocr/yomikai_ocr_check/best.onnx')
    import onnxruntime as ort
    sess = ort.InferenceSession('/home/user/yomikai-ocr/yomikai_ocr_check/best.onnx', providers=['CPUExecutionProvider'])
    ok = True
    for W_ in (64, 128, 256, 448, 96):
        x = np.random.rand(1, 1, 32, W_).astype(np.float32)
        ref = sess.run(None, {'x': x})[0]
        with torch.no_grad():
            out = net(torch.from_numpy(x)).numpy()
        ref2 = ref.reshape(ref.shape[0]*ref.shape[1], 111) if ref.ndim == 3 else ref
        d = np.abs(out.reshape(-1, 111) - ref2).max()
        print(f'W={W_}: torch{out.shape} onnx{ref.shape} max|diff|={d:.2e}')
        ok &= d < 1e-4
    print('EQUIVALENT' if ok else 'MISMATCH!')
