"""Draw the placement/performance figures for the demo docs with Pillow.

No matplotlib in the bundled runtime, so the charts are drawn directly.
Data comes from the 2026-10-15 measurements (raw CSVs belong to the earlier analysis workspace, not this repo
and the demo run log).

Usage: python make-charts.py
"""
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")

FONT_REG = "C:/Windows/Fonts/msyh.ttc"
FONT_BOLD = "C:/Windows/Fonts/msyhbd.ttc"

BLUE = (31, 59, 115)
BLUE_L = (138, 168, 214)
RED = (192, 64, 58)
GREEN = (46, 139, 87)
GRAY = (128, 128, 128)
GRAY_L = (232, 232, 232)
INK = (34, 34, 34)
GRID = (214, 214, 214)

# measured data (median of 5 runs unless noted)
PERF = [
    ("未打补丁 Vulkan0", 19.48, RED),
    ("打补丁 Vulkan0", 10.48, BLUE),
    ("CUDA0", 1.81, GREEN),
    ("纯 CPU（12 线程）", 18.21, GRAY),
]
PLACEMENT = [
    ("未打补丁", 60.9, 68, 68),
    ("已打补丁", 0.0, 19, 19),
]
EPOCHS = [
    ("epoch 0", 7.11, 1.79, "8.90 s"),
    ("epoch 1", 0.0, 1.79, "1.79 s"),
]


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


def text(d, xy, s, size=24, bold=False, fill=INK, anchor=None):
    d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


def wrap_center(d, cx, y, s, size=22, fill=GRAY, bold=False, width=1600):
    """Draw a centered line, shrinking the font until it fits."""
    f = font(size, bold)
    while d.textlength(s, font=f) > width and size > 10:
        size -= 1
        f = font(size, bold)
    d.text((cx, y), s, font=f, fill=fill, anchor="ma")


def fig_placement():
    w, h = 1640, 620
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    text(d, (40, 30), "落点对比：OUT_PROD 算在哪个后端上", 36, True, BLUE)
    text(d, (40, 82), "stories15M 微调，1 个 batch，逐节点统计（GGML_SCHED_DEBUG=2）", 22, False, GRAY)

    # ---- left: OUT_PROD CPU/GPU share (100% stacked bars)
    lx, lw = 210, 520
    text(d, (40, 150), "OUT_PROD 实例占比", 26, True, INK)
    y = 210
    for name, cpu_pct, _, _ in PLACEMENT:
        text(d, (lx - 20, y + 22), name, 24, False, INK, anchor="ra")
        d.rectangle([lx, y, lx + lw, y + 56], fill=GRAY_L)
        cpu_w = int(lw * cpu_pct / 100.0)
        if cpu_w > 0:
            d.rectangle([lx, y, lx + cpu_w, y + 56], fill=RED)
            text(d, (lx + cpu_w / 2, y + 28), "CPU %.1f%%" % cpu_pct, 22, True, "white", anchor="mm")
        gpu_w = lw - cpu_w
        if gpu_w > 0:
            d.rectangle([lx + cpu_w, y, lx + lw, y + 56], fill=BLUE)
            label = "GPU %.1f%%" % (100.0 - cpu_pct)
            text(d, (lx + cpu_w + gpu_w / 2, y + 28), label, 22, True, "white", anchor="mm")
        y += 100
    text(d, (lx, y + 6), "共 110 个 OUT_PROD / batch", 20, False, GRAY)

    # ---- right: scheduler segments (stacked columns)
    rx = 980
    text(d, (rx, 150), "每 batch 调度段数", 26, True, INK)
    base_y, max_h = 430, 220
    max_seg = 136.0
    for i, (name, _, cpu_seg, gpu_seg) in enumerate(PLACEMENT):
        cx = rx + 90 + i * 250
        total = cpu_seg + gpu_seg
        ch = max_h * cpu_seg / max_seg
        gh = max_h * gpu_seg / max_seg
        d.rectangle([cx, base_y - ch, cx + 130, base_y], fill=RED)
        d.rectangle([cx, base_y - ch - gh, cx + 130, base_y - ch], fill=BLUE)
        text(d, (cx + 65, base_y - ch / 2), str(cpu_seg), 22, True, "white", anchor="mm")
        text(d, (cx + 65, base_y - ch - gh / 2), str(gpu_seg), 22, True, "white", anchor="mm")
        text(d, (cx + 65, base_y - ch - gh - 30), "%d 段" % total, 24, True, INK, anchor="ma")
        text(d, (cx + 65, base_y + 12), name, 22, False, INK, anchor="ma")
    d.line([rx - 30, base_y, w - 40, base_y], fill=GRID, width=2)
    d.rectangle([rx + 30, 500, rx + 60, 524], fill=RED)
    text(d, (rx + 70, 500), "CPU 段", 20, False, INK)
    d.rectangle([rx + 160, 500, rx + 190, 524], fill=BLUE)
    text(d, (rx + 200, 500), "GPU 段", 20, False, INK)
    text(d, (rx + 30, 548), "未打补丁时那 68 个 CPU 段，就是权重梯度回落的代价。", 20, False, GRAY)

    im.save(os.path.join(OUT, "fig-placement.png"))
    return "fig-placement.png"


def fig_perf():
    w, h = 1640, 760
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    text(d, (40, 30), "微调性能对比：Vulkan / CUDA / 纯 CPU", 36, True, BLUE)
    text(d, (40, 82), "stories15M，34 batch，1 epoch，墙钟中位数（各 5 轮，测前停掉后台构建）", 22, False, GRAY)

    base_y, max_h = 560, 380
    vmax = 22.0
    x0 = 130
    bw = 200
    gap = 90
    for i, (name, secs, col) in enumerate(PERF):
        cx = x0 + i * (bw + gap)
        bh = max_h * secs / vmax
        d.rectangle([cx, base_y - bh, cx + bw, base_y], fill=col)
        text(d, (cx + bw / 2, base_y - bh - 34), "%.2f s" % secs, 28, True, INK, anchor="ma")
        wrap_center(d, cx + bw / 2, base_y + 14, name, 22, INK, width=bw + gap - 20)
    d.line([x0 - 40, base_y, w - 40, base_y], fill=GRID, width=2)

    # CPU reference line
    cpu_y = base_y - max_h * 18.21 / vmax
    for x in range(x0 - 40, w - 40, 18):
        d.line([x, cpu_y, x + 9, cpu_y], fill=GRAY, width=2)
    text(d, (w - 50, cpu_y - 30), "纯 CPU 基线 18.21 s", 20, False, GRAY, anchor="ra")

    # annotation
    box_y = 620
    d.rectangle([x0 - 40, box_y, w - 40, box_y + 96], fill=(244, 247, 252), outline=BLUE_L, width=2)
    text(d, (x0 - 10, box_y + 14),
         "打补丁后 Vulkan 比纯 CPU 快 1.84×（此前慢 1.07×）；仍比 CUDA 慢 5.7×。", 24, True, BLUE)
    text(d, (x0 - 10, box_y + 54),
         "补丁本身给 Vulkan 带来 1.86×；CUDA 基本不变（1.81→1.83）；代价是 CPU 后端约 +5.8%。", 21, False, GRAY)

    im.save(os.path.join(OUT, "fig-perf.png"))
    return "fig-perf.png"


def fig_epoch():
    w, h = 1400, 720
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    text(d, (40, 30), "演示程序两个 epoch 的耗时构成（Vulkan0）", 34, True, BLUE)
    text(d, (40, 80), "首次用到才会创建 Vulkan pipeline（1566 个 SPIR-V 变体），所以首轮明显更慢", 21, False, GRAY)

    base_y, max_h = 540, 340
    vmax = 10.0
    bw, gap, x0 = 230, 200, 260
    for i, (name, setup, steady, total) in enumerate(EPOCHS):
        cx = x0 + i * (bw + gap)
        sh = max_h * setup / vmax
        th = max_h * steady / vmax
        top = base_y - (sh + th)
        # steady part at the bottom, one-time pipeline creation stacked on top
        d.rectangle([cx, base_y - th, cx + bw, base_y], fill=BLUE)
        if sh > 0:
            d.rectangle([cx, top, cx + bw, base_y - th], fill=BLUE_L)
            text(d, (cx + bw / 2, (top + base_y - th) / 2), "管线创建约 %.1f s" % setup,
                 20, True, BLUE, anchor="mm")
        if th > 40:
            text(d, (cx + bw / 2, base_y - th / 2), "%.1f s" % steady, 22, True, "white", anchor="mm")
        text(d, (cx + bw / 2, top - 36), total, 28, True, INK, anchor="ma")
        text(d, (cx + bw / 2, base_y + 16), name, 24, False, INK, anchor="ma")
    d.line([x0 - 60, base_y, w - 60, base_y], fill=GRID, width=2)

    box_y = 600
    d.rectangle([x0 - 60, box_y, w - 60, box_y + 84], fill=(244, 247, 252), outline=BLUE_L, width=2)
    text(d, (x0 - 30, box_y + 14), "稳态每个 epoch 约 1.8 s；只跑 1 个 epoch 的计时会把管线创建也算进去。", 22, True, BLUE)
    text(d, (x0 - 30, box_y + 50), "比较性能时应预热，或看第 2 个 epoch / 多轮中位数。", 21, False, GRAY)

    im.save(os.path.join(OUT, "fig-epoch.png"))
    return "fig-epoch.png"


def fig_lr():
    """Two panels: val loss vs lr after 1 epoch, and val loss vs cumulative updates."""
    w, h = 1640, 700
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    text(d, (40, 28), "学习率与语料规模：完整 wiki.train.raw 上的实测", 34, True, BLUE)
    text(d, (40, 76), "stories15M，1 轮 = 10,488 次参数更新 / 5,369,856 token，Vulkan0", 21, False, GRAY)

    # ---------------- left: val loss vs lr (log x)
    lx0, lx1 = 120, 760
    ly0, ly1 = 560, 180          # y for loss 7.4 .. 5.5
    lr_pts = [(3e-7, 5.899), (1e-6, 5.725), (3e-6, 6.155), (1e-5, 7.171)]
    import math
    lmin, lmax = 2e-7, 2e-5

    def sx(lr):
        return lx0 + (math.log10(lr) - math.log10(lmin)) / (math.log10(lmax) - math.log10(lmin)) * (lx1 - lx0)

    def sy(v):
        return ly0 - (v - 5.5) / (7.4 - 5.5) * (ly0 - ly1)

    text(d, (40, 128), "1 轮后的 val loss vs 学习率", 25, True, INK)
    for v in (5.5, 6.0, 6.5, 7.0):
        y = sy(v)
        d.line([lx0, y, lx1, y], fill=GRID, width=1)
        text(d, (lx0 - 12, y - 12), "%.1f" % v, 18, False, GRAY, anchor="ra")
    for lr in (3e-7, 1e-6, 3e-6, 1e-5):
        x = sx(lr)
        d.line([x, ly0, x, ly0 + 8], fill=INK, width=2)
        text(d, (x, ly0 + 14), ("%g" % lr), 18, False, INK, anchor="ma")
    d.line([lx0, ly0, lx1, ly0], fill=INK, width=2)
    d.line([lx0, ly1 - 20, lx0, ly0], fill=INK, width=2)
    pts = [(sx(lr), sy(v)) for lr, v in lr_pts]
    d.line(pts, fill=BLUE, width=3)
    offs = [(0, -34), (-62, -24), (58, -14), (0, -34)]   # keep labels off the polyline
    for (lr, v), (x, y), (dx, dy) in zip(lr_pts, pts, offs):
        best = abs(lr - 1e-6) < 1e-9
        r = 11 if best else 7
        d.ellipse([x - r, y - r, x + r, y + r], fill=RED if best else BLUE, outline="white")
        text(d, (x + dx, y + dy), "%.3f" % v, 19, True, RED if best else INK, anchor="ma")
    text(d, (sx(1e-6) + 26, sy(5.725) + 10), "最佳", 20, True, RED)

    # ---------------- right: val loss vs cumulative updates
    rx0, rx1 = 900, 1560
    ry0, ry1 = 560, 180
    umax = 100000.0
    vmin, vmax = 5.2, 8.1

    def ux(u):
        return rx0 + u / umax * (rx1 - rx0)

    def uy(v):
        return ry0 - (v - vmin) / (vmax - vmin) * (ry0 - ry1)

    text(d, (820, 128), "val loss vs 累计更新次数", 25, True, INK)
    for v in (5.5, 6.0, 6.5, 7.0, 7.5, 8.0):
        y = uy(v)
        d.line([rx0, y, rx1, y], fill=GRID, width=1)
        text(d, (rx0 - 12, y - 12), "%.1f" % v, 18, False, GRAY, anchor="ra")
    for u in (0, 25000, 50000, 75000):
        x = ux(u)
        d.line([x, ry0, x, ry0 + 8], fill=INK, width=2)
        text(d, (x, ry0 + 14), "%dk" % (u // 1000), 18, False, INK, anchor="ma")
    d.line([rx0, ry0, rx1, ry0], fill=INK, width=2)
    d.line([rx0, ry1 - 20, rx0, ry0], fill=INK, width=2)

    series = [
        ("1e-5", RED, [(10488, 7.171), (20976, 7.619), (62928, 7.819)], (12, -16)),
        ("1e-6", BLUE, [(10488, 5.725), (20976, 5.507), (31464, 5.448),
                        (41952, 5.412), (52440, 5.344), (62928, 5.312),
                        (73416, 5.292), (83904, 5.281), (94392, 5.273)], (14, -24)),
        ("3e-7", GREEN, [(10488, 5.899), (20976, 5.512)], (14, 6)),
    ]
    for name, col, s, (dx, dy) in series:
        p = [(ux(u), uy(v)) for u, v in s]
        d.line(p, fill=col, width=3)
        for x, y in p:
            d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=col, outline="white")
        text(d, (p[-1][0] + dx, p[-1][1] + dy), name, 19, True, col)
    # notes go below the axes so they never cross the lines
    text(d, (40, ry0 + 46), "1e-5 在 6 万步后已劣化（7.17 → 7.82）；1e-6 跑到 9.4 万步仍未回升，但第 7 轮起每轮只降 <0.01 —— 已进入平台期。",
         19, False, GRAY)

    im.save(os.path.join(OUT, "fig-lr.png"))
    return "fig-lr.png"


def fig_pretrain():
    """Two panels: val loss per from-scratch config, and val loss vs the lr*step budget."""
    w, h = 1640, 760
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    text(d, (40, 26), "从零训练（预训练）实测：配置与步长预算", 34, True, BLUE)
    text(d, (40, 74), "stories15M 随机初始化 · TinyStories 切片 · Vulkan0 · 每轮约 10,000–42,000 步", 21, False, GRAY)

    # (label, val, sum_lr_step)
    runs = [
        ("40 MB  lr 7.5e-6  1 轮", 4.331, 0.32, BLUE),
        ("10 MB  lr 3e-5 衰减 3 轮", 4.785, 0.55, GRAY),
        ("10 MB  lr 3e-5  1 轮", 5.405, 0.31, GRAY),
        ("40 MB  lr 3e-5  1 轮", 5.995, 1.26, RED),
        ("40 MB  lr 3e-5 衰减 3 轮", 6.533, 2.21, RED),
        ("40 MB  lr 3e-4  1 轮", 13.29, 12.6, RED),
    ]

    # ---------------- left: horizontal bars
    text(d, (40, 120), "各配置的 val loss（越低越好）", 24, True, INK)
    x0, xmax = 470, 780
    y0, step = 170, 78
    vmax = 14.0
    for i, (label, val, _, col) in enumerate(runs):
        y = y0 + i * step
        text(d, (x0 - 16, y + 12), label, 21, False, INK, anchor="ra")
        bw = (xmax - x0) * min(val / vmax, 1.0)
        d.rectangle([x0, y, x0 + bw, y + 40], fill=col)
        text(d, (x0 + bw + 12, y + 10), "%.3f" % val, 22, True, col)
    d.line([x0, y0 + len(runs) * step - 20, xmax, y0 + len(runs) * step - 20], fill=GRID, width=2)
    text(d, (x0, y0 + len(runs) * step + 2), "蓝 = 最优；灰 = 健康；红 = 预算超标", 19, False, GRAY)

    # ---------------- right: val vs sum(lr*step), log x
    import math
    rx0, rx1 = 1010, 1560
    ry0, ry1 = 620, 170
    xl, xr = 0.15, 20.0

    def sx(v):
        return rx0 + (math.log10(v) - math.log10(xl)) / (math.log10(xr) - math.log10(xl)) * (rx1 - rx0)

    def sy(v):
        return ry0 - (v - 3.5) / (14.0 - 3.5) * (ry0 - ry1)

    text(d, (960, 120), "val loss vs Σ lr×step（步长预算）", 24, True, INK)
    # healthy band 0.3 - 0.5
    d.rectangle([sx(0.3), ry1 - 10, sx(0.5), ry0], fill=(226, 236, 246))
    text(d, (sx(0.52) + 10, sy(11.3)), "健康区 0.3–0.5", 18, True, BLUE)
    for v in (4.0, 6.0, 8.0, 10.0, 12.0):
        y = sy(v)
        d.line([rx0, y, rx1, y], fill=GRID, width=1)
        text(d, (rx0 - 12, y - 12), "%.0f" % v, 18, False, GRAY, anchor="ra")
    for t in (0.3, 0.5, 1, 2, 5, 10):
        x = sx(t)
        d.line([x, ry0, x, ry0 + 8], fill=INK, width=2)
        text(d, (x, ry0 + 14), ("%g" % t), 18, False, INK, anchor="ma")
    d.line([rx0, ry0, rx1, ry0], fill=INK, width=2)
    d.line([rx0, ry1 - 10, rx0, ry0], fill=INK, width=2)
    for label, val, prod, col in runs:
        x, y = sx(prod), sy(val)
        r = 9
        d.ellipse([x - r, y - r, x + r, y + r], fill=col, outline="white")
        text(d, (x + 14, y - 14), "%.2f" % val, 18, True, col)
    text(d, (rx0, ry0 + 46), "同一预算下，数据越多越好（0.31 → 5.405 vs 0.32 → 4.331）", 19, False, GRAY)

    im.save(os.path.join(OUT, "fig-pretrain.png"))
    return "fig-pretrain.png"


def main():
    os.makedirs(OUT, exist_ok=True)
    made = [fig_placement(), fig_perf(), fig_epoch(), fig_lr(), fig_pretrain()]
    print("font:", FONT_REG, "|", FONT_BOLD)
    for name in made:
        p = os.path.join(OUT, name)
        with Image.open(p) as im:
            print("  %-22s %dx%d  %.0f KB" % (name, im.size[0], im.size[1], os.path.getsize(p) / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
