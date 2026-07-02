# ui.py  —  FP4-SPARSA Demo  (redesigned — crisp typography)
import tkinter as tk
from tkinter import font as tkfont
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image, ImageTk
import threading, math, random, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cifar10_loader import load_cifar10
from models.resnet20 import ResNet20
from quantize import quantize_to_fp4, quantize_model_weights, compute_sparsity_stats

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/resnet20_fp32.pth"
CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']
BATCH_SIZE = 256

# ── palette ───────────────────────────────────────────────────────────────────
BG      = "#080b12"
PANEL   = "#0d1117"
BORDER  = "#1a2236"
ACCENT  = "#00e5ff"
GREEN   = "#00ff88"
AMBER   = "#ffb300"
RED     = "#ff4757"
TEXT    = "#cdd6f4"
DIM     = "#6b7a99"

# ── Font system ───────────────────────────────────────────────────────────────
# Prefer Consolas (Windows) → Menlo (Mac) → DejaVu Sans Mono (Linux)
# These render sharply at all sizes unlike Courier New
MONO    = "Consolas"
SANS    = "Segoe UI"

# ── inference helpers ─────────────────────────────────────────────────────────
def _loader(te_x, te_y):
    from torch.utils.data import DataLoader, TensorDataset
    return DataLoader(TensorDataset(te_x, te_y),
                      batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

def evaluate(model, te_x, te_y):
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for x, y in _loader(te_x, te_y):
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total

def evaluate_fp4(model, qw, te_x, te_y):
    orig = {}
    for name, mod in model.named_modules():
        if name in qw:
            orig[name] = mod.weight.data.clone()
            mod.weight.data = qw[name]['quantized'].to(DEVICE)
    acc = evaluate(model, te_x, te_y)
    for name, mod in model.named_modules():
        if name in orig:
            mod.weight.data = orig[name]
    return acc

def count_act_sparsity(model, qw, te_x, te_y, n=5):
    az = at = 0
    hooks = []
    def make_hook():
        def h(mod, inp, out):
            nonlocal az, at
            r = F.relu(out)
            az += (r == 0).sum().item(); at += r.numel()
        return h
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            hooks.append(mod.register_forward_hook(make_hook()))
    orig = {}
    for name, mod in model.named_modules():
        if name in qw:
            orig[name] = mod.weight.data.clone()
            mod.weight.data = qw[name]['quantized'].to(DEVICE)
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(_loader(te_x, te_y)):
            if i >= n: break
            model(x.to(DEVICE))
    for name, mod in model.named_modules():
        if name in orig: mod.weight.data = orig[name]
    for h in hooks: h.remove()
    return (100.0 * az / at) if at > 0 else 0.0


# ── animated glow bar ─────────────────────────────────────────────────────────
class GlowBar(tk.Canvas):
    def __init__(self, parent, color, **kw):
        super().__init__(parent, bg=PANEL, highlightthickness=0, height=28, **kw)
        self.color = color
        self._pct = 0.0
        self._target = 0.0
        self._animating = False

    def set_value(self, pct):
        self._target = pct
        if not self._animating:
            self._animating = True
            self._animate()

    def _animate(self):
        diff = self._target - self._pct
        if abs(diff) < 0.05:
            self._pct = self._target
            self._animating = False
        else:
            self._pct += diff * 0.1
            self.after(16, self._animate)
        self._draw()

    def _draw(self):
        self.delete("all")
        w = self.winfo_width() or 400
        h = 28
        label = f"{self._pct:.2f}%"
        label_w = 68          # fixed px for label column — keeps text outside bar
        bar_w   = w - label_w
        track_y0, track_y1 = 11, 17
        # track
        self.create_rectangle(0, track_y0, bar_w, track_y1, fill="#111827", outline="")
        bw = int((self._pct / 100.0) * bar_w)
        if bw > 2:
            self.create_rectangle(0, track_y0, bw, track_y1, fill=self.color, outline="")
            # bright tip
            self.create_rectangle(max(0, bw - 4), track_y0 - 2,
                                   bw, track_y1 + 2, fill="white", outline="")
        # label — Consolas 10 bold is crisp and sharp
        self.create_text(w - 4, h // 2, text=label,
                         anchor="e", fill=self.color,
                         font=(MONO, 10, "bold"))


# ── pulsing status dot ────────────────────────────────────────────────────────
class PulsingDot(tk.Canvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, highlightthickness=0,
                         width=12, height=12, **kw)
        self.color = AMBER
        self._phase = 0
        self._run()

    def _run(self):
        self._phase = (self._phase + 0.18) % (2 * math.pi)
        r = 3 + 2 * abs(math.sin(self._phase))
        cx, cy = 6, 6
        self.delete("all")
        self.create_oval(cx-r, cy-r, cx+r, cy+r, fill=self.color, outline="")
        self.after(50, self._run)


# ── font helpers ──────────────────────────────────────────────────────────────
def _f(size, bold=False, mono=False):
    family = MONO if mono else SANS
    weight = "bold" if bold else "normal"
    return (family, size, weight)


# ── main app ──────────────────────────────────────────────────────────────────
class FP4DemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FP4-SPARSA  ·  Neural Inference Accelerator")
        self.root.configure(bg=BG)
        self.root.geometry("1280x840")
        self.root.minsize(1100, 760)
        self.root.resizable(True, True)

        self.model = self.qw = None
        self.te_img_raw = self.te_x = self.te_y = None
        self.current_idx = 0
        self.results = {}

        self._build_ui()
        threading.Thread(target=self._load_all, daemon=True).start()

    # ── widget helpers ────────────────────────────────────────────────────────
    def _card(self, parent, pady=0):
        outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
        inner = tk.Frame(outer, bg=PANEL)
        inner.pack(fill="both", expand=True)
        tk.Frame(inner, bg=ACCENT, width=3).place(x=0, y=0, relheight=1)
        return outer, inner

    def _label_tag(self, parent, text):
        """Small ALL-CAPS section label — Consolas 10 bold, readable DIM colour."""
        return tk.Label(parent, text=text, bg=PANEL, fg=DIM,
                        font=_f(10, bold=True, mono=True))

    def _sep(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=12, pady=6)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(18, 0))

        title_f = tk.Frame(hdr, bg=BG)
        title_f.pack(side="left")
        tk.Label(title_f, text="FP4-SPARSA", bg=BG, fg=ACCENT,
                 font=_f(26, bold=True, mono=True)).pack(side="left")
        tk.Label(title_f, text="  Accelerator Demo", bg=BG, fg=TEXT,
                 font=_f(14)).pack(side="left", pady=4)

        st = tk.Frame(hdr, bg=BG)
        st.pack(side="right", pady=4)
        self._dot = PulsingDot(st)
        self._dot.pack(side="left", padx=(0, 7))
        self.status_var = tk.StringVar(value="Initialising…")
        tk.Label(st, textvariable=self.status_var, bg=BG, fg=AMBER,
                 font=_f(10, mono=True)).pack(side="left")

        # ── tag pills ─────────────────────────────────────────────────────────
        pills = tk.Frame(self.root, bg=BG)
        pills.pack(fill="x", padx=24, pady=(8, 16))
        tags = ["ResNet-20", "CIFAR-10", "FP4 E2M1",
                "Kintex-7 xc7k160t", "350 MHz", "64× DSP48E1"]
        for tag in tags:
            tk.Label(pills, text=f"  {tag}  ", bg=BORDER, fg=DIM,
                     font=_f(9, mono=True), pady=4).pack(side="left", padx=3)

        # ── body ──────────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=(0, 18))
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        lf = tk.Frame(body, bg=BG, width=275)
        lf.grid(row=0, column=0, sticky="ns", padx=(0, 14))
        lf.grid_propagate(False)
        self._build_left(lf)

        rf = tk.Frame(body, bg=BG)
        rf.grid(row=0, column=1, sticky="nsew")
        rf.columnconfigure(0, weight=1)
        rf.rowconfigure(2, weight=1)
        self._build_right(rf)

    def _build_left(self, parent):
        outer, inner = self._card(parent)
        outer.pack(fill="x", pady=(0, 10))

        self._label_tag(inner, "TEST IMAGE").pack(anchor="w", padx=16, pady=(12, 6))

        frame = tk.Frame(inner, bg=ACCENT, padx=2, pady=2)
        frame.pack(padx=20, pady=(0, 10))
        self.img_label = tk.Label(frame, bg="#000")
        self.img_label.pack()

        self._sep(inner)

        for title, attr, color in [
            ("TRUE LABEL",      "true_lbl",  TEXT),
            ("FP32 PREDICTION", "pred_fp32", GREEN),
            ("FP4  PREDICTION", "pred_fp4",  ACCENT),
        ]:
            self._label_tag(inner, title).pack(anchor="w", padx=16, pady=(8, 0))
            var = tk.StringVar(value="—")
            setattr(self, attr, var)
            tk.Label(inner, textvariable=var, bg=PANEL, fg=color,
                     font=_f(15, bold=True)).pack(anchor="w", padx=16)

        self._sep(inner)
        self._label_tag(inner, "CONFIDENCE").pack(anchor="w", padx=16, pady=(4, 0))
        self.conf_var = tk.StringVar(value="FP32 —      FP4 —")
        tk.Label(inner, textvariable=self.conf_var, bg=PANEL, fg=DIM,
                 font=_f(10, mono=True)).pack(anchor="w", padx=16, pady=(3, 12))

        # buttons
        self.next_btn = tk.Button(
            parent, text="NEXT IMAGE  ▶", command=self._next_image,
            bg=ACCENT, fg=BG, font=_f(11, bold=True, mono=True),
            relief="flat", padx=12, pady=9, state="disabled",
            activebackground="#00bcd4", cursor="hand2")
        self.next_btn.pack(fill="x", pady=(0, 6))

        self.rand_btn = tk.Button(
            parent, text="⟳  RANDOM IMAGE", command=self._random_image,
            bg=BORDER, fg=DIM, font=_f(10, mono=True),
            relief="flat", padx=12, pady=7, state="disabled",
            activebackground="#1a2236", cursor="hand2")
        self.rand_btn.pack(fill="x")

    def _build_right(self, parent):
        # ── accuracy ──────────────────────────────────────────────────────────
        outer, inner = self._card(parent)
        outer.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        top = tk.Frame(inner, bg=PANEL)
        top.pack(fill="x", padx=16, pady=(12, 8))
        self._label_tag(top, "ACCURACY COMPARISON").pack(side="left")

        nums = tk.Frame(top, bg=PANEL)
        nums.pack(side="right")
        self.acc_fp32_var = tk.StringVar(value="—")
        self.acc_fp4_var  = tk.StringVar(value="—")
        self.acc_drop_var = tk.StringVar(value="—")
        for lbl, var, col in [("FP32", self.acc_fp32_var, GREEN),
                               ("FP4",  self.acc_fp4_var,  ACCENT),
                               ("DROP", self.acc_drop_var, AMBER)]:
            c = tk.Frame(nums, bg=PANEL)
            c.pack(side="left", padx=12)
            tk.Label(c, text=lbl, bg=PANEL, fg=DIM,
                     font=_f(9, bold=True, mono=True)).pack()
            tk.Label(c, textvariable=var, bg=PANEL, fg=col,
                     font=_f(16, bold=True, mono=True)).pack()

        self._sep(inner)

        for lbl, attr, col in [("FP32", "bar_fp32", GREEN),
                                ("FP4 ", "bar_fp4",  ACCENT)]:
            row = tk.Frame(inner, bg=PANEL)
            row.pack(fill="x", padx=16, pady=3)
            tk.Label(row, text=lbl, bg=PANEL, fg=DIM,
                     font=_f(10, bold=True, mono=True), width=5).pack(side="left")
            bar = GlowBar(row, col)
            bar.pack(side="left", fill="x", expand=True)
            setattr(self, attr, bar)
        tk.Label(inner, bg=PANEL).pack(pady=5)

        # ── sparsity ──────────────────────────────────────────────────────────
        outer2, inner2 = self._card(parent)
        outer2.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        self._label_tag(inner2, "SPARSITY  &  EFFICIENCY METRICS").pack(
            anchor="w", padx=16, pady=(12, 10))

        g = tk.Frame(inner2, bg=PANEL)
        g.pack(fill="x", padx=16, pady=(0, 10))
        g.columnconfigure((0, 1, 2, 3), weight=1)

        metrics = [
            ("WGT SPARSITY", "wgt_sp_var", GREEN),
            ("ACT SPARSITY", "act_sp_var", ACCENT),
            ("MACs SKIPPED", "macs_var",   AMBER),
            ("POWER SAVED",  "power_var",  RED),
        ]
        for i, (lbl, attr, col) in enumerate(metrics):
            cell = tk.Frame(g, bg=BORDER, padx=12, pady=10)
            cell.grid(row=0, column=i, padx=(0, 6 if i < 3 else 0), sticky="ew")
            tk.Label(cell, text=lbl, bg=BORDER, fg=DIM,
                     font=_f(9, bold=True, mono=True)).pack(anchor="w")
            var = tk.StringVar(value="—")
            setattr(self, attr, var)
            tk.Label(cell, textvariable=var, bg=BORDER, fg=col,
                     font=_f(18, bold=True, mono=True)).pack(anchor="w")

        for lbl, attr, col in [("WGT", "sp_wgt", GREEN),
                                ("ACT", "sp_act", ACCENT),
                                ("MAC", "sp_mac", AMBER)]:
            row = tk.Frame(inner2, bg=PANEL)
            row.pack(fill="x", padx=16, pady=2)
            tk.Label(row, text=lbl, bg=PANEL, fg=DIM,
                     font=_f(10, bold=True, mono=True), width=5).pack(side="left")
            bar = GlowBar(row, col)
            bar.pack(side="left", fill="x", expand=True)
            setattr(self, attr, bar)
        tk.Label(inner2, bg=PANEL).pack(pady=5)

        # ── hardware ──────────────────────────────────────────────────────────
        outer3, inner3 = self._card(parent)
        outer3.grid(row=2, column=0, sticky="nsew")

        self._label_tag(inner3,
            "HARDWARE CORRELATION  —  Kintex-7 xc7k160tfbg676-2").pack(
            anchor="w", padx=16, pady=(12, 10))

        hg = tk.Frame(inner3, bg=PANEL)
        hg.pack(fill="x", padx=16, pady=(0, 14))
        for i in range(5): hg.columnconfigure(i, weight=1)

        hw_stats = [
            ("Clock",       "350 MHz",       ACCENT),
            ("WNS",         "+0.059 ns",      GREEN),
            ("Throughput",  "44.8 GOPS",      ACCENT),
            ("Efficiency",  "278.3 GOPS/W",   GREEN),
            ("DSP48E1",     "64 / 600",        ACCENT),
            ("LUTs",        "1,546",           GREEN),
            ("Flip-Flops",  "2,014",           ACCENT),
            ("Dyn. Power",  "0.049 W",         AMBER),
            ("Total Power", "0.161 W",         AMBER),
            ("Sparsity",    "Lane-wise v15c",  GREEN),
        ]
        for i, (lbl, val, col) in enumerate(hw_stats):
            r, c = divmod(i, 5)
            cell = tk.Frame(hg, bg=BORDER, padx=10, pady=8)
            cell.grid(row=r, column=c,
                      padx=(0, 4 if c < 4 else 0), pady=3, sticky="ew")
            tk.Label(cell, text=lbl, bg=BORDER, fg=DIM,
                     font=_f(9, mono=True)).pack(anchor="w")
            tk.Label(cell, text=val, bg=BORDER, fg=col,
                     font=_f(11, bold=True, mono=True)).pack(anchor="w")

    # ── loading ───────────────────────────────────────────────────────────────
    def _set_status(self, msg, color=AMBER):
        def _do():
            self.status_var.set(msg)
            self._dot.color = color
        self.root.after(0, _do)

    def _load_all(self):
        try:
            self._set_status("Loading CIFAR-10…")
            _, _, te_img, te_lbl = load_cifar10()
            self.te_img_raw = te_img
            self.te_x = torch.tensor(te_img)
            self.te_y = torch.tensor(te_lbl, dtype=torch.long)

            self._set_status("Loading checkpoint…")
            model = ResNet20().to(DEVICE)
            ckpt  = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt['state_dict'])
            model.eval()
            self.model = model

            self._set_status("Evaluating FP32…")
            fp32_acc = evaluate(model, self.te_x, self.te_y)

            self._set_status("Quantizing to FP4 E2M1…")
            self.qw = quantize_model_weights(model, scale_mode='per_channel')
            _, _, _, wgt_sp = compute_sparsity_stats(self.qw)

            self._set_status("Evaluating FP4…")
            fp4_acc = evaluate_fp4(model, self.qw, self.te_x, self.te_y)

            self._set_status("Counting activation sparsity…")
            act_sp = count_act_sparsity(model, self.qw, self.te_x, self.te_y)

            mac_skip    = wgt_sp + act_sp - (wgt_sp * act_sp / 100)
            power_saved = mac_skip * 0.4

            self.results = dict(fp32_acc=fp32_acc, fp4_acc=fp4_acc,
                                wgt_sp=wgt_sp, act_sp=act_sp,
                                mac_skip=mac_skip, power_saved=power_saved)
            self.root.after(0, self._on_loaded)

        except Exception as e:
            import traceback; traceback.print_exc()
            self._set_status(f"Error: {e}", RED)

    def _on_loaded(self):
        r = self.results
        drop = r['fp32_acc'] - r['fp4_acc']

        self.acc_fp32_var.set(f"{r['fp32_acc']:.2f}%")
        self.acc_fp4_var .set(f"{r['fp4_acc']:.2f}%")
        self.acc_drop_var.set(f"{drop:.2f}%")

        self.bar_fp32.set_value(r['fp32_acc'])
        self.bar_fp4 .set_value(r['fp4_acc'])

        self.wgt_sp_var.set(f"{r['wgt_sp']:.1f}%")
        self.act_sp_var.set(f"{r['act_sp']:.1f}%")
        self.macs_var  .set(f"{r['mac_skip']:.1f}%")
        self.power_var .set(f"{r['power_saved']:.1f}%")

        self.sp_wgt.set_value(r['wgt_sp'])
        self.sp_act.set_value(r['act_sp'])
        self.sp_mac.set_value(r['mac_skip'])

        self.next_btn.config(state="normal")
        self.rand_btn.config(state="normal")
        self._set_status(
            f"Ready  ·  {DEVICE.type.upper()}  ·  {len(self.te_y):,} images", GREEN)
        self._show_image(0)

    # ── image display ─────────────────────────────────────────────────────────
    def _show_image(self, idx):
        if self.te_img_raw is None: return
        self.current_idx = idx

        MEAN = np.array([0.4914, 0.4822, 0.4465])
        STD  = np.array([0.2023, 0.1994, 0.2010])
        img  = self.te_img_raw[idx].transpose(1, 2, 0)
        img  = np.clip(img * STD + MEAN, 0, 1)
        img  = (img * 255).astype(np.uint8)
        pil  = Image.fromarray(img).resize((152, 152), Image.NEAREST)
        tk_img = ImageTk.PhotoImage(pil)
        self.img_label.config(image=tk_img)
        self.img_label.image = tk_img

        label = int(self.te_y[idx].item())
        self.true_lbl.set(CIFAR10_CLASSES[label])

        x = self.te_x[idx].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            l32    = self.model(x)
            pred32 = l32.argmax(1).item()
            c32    = torch.softmax(l32, 1)[0, pred32].item() * 100

        orig = {}
        for name, mod in self.model.named_modules():
            if name in self.qw:
                orig[name] = mod.weight.data.clone()
                mod.weight.data = self.qw[name]['quantized'].to(DEVICE)
        with torch.no_grad():
            l4     = self.model(x)
            pred4  = l4.argmax(1).item()
            c4     = torch.softmax(l4, 1)[0, pred4].item() * 100
        for name, mod in self.model.named_modules():
            if name in orig: mod.weight.data = orig[name]

        self.pred_fp32.set(CIFAR10_CLASSES[pred32])
        self.pred_fp4 .set(CIFAR10_CLASSES[pred4])
        self.conf_var .set(f"FP32  {c32:.1f}%      FP4  {c4:.1f}%")

    def _next_image(self):
        self._show_image((self.current_idx + 1) % len(self.te_y))

    def _random_image(self):
        self._show_image(random.randint(0, len(self.te_y) - 1))


if __name__ == "__main__":
    root = tk.Tk()
    FP4DemoApp(root)
    root.mainloop() 