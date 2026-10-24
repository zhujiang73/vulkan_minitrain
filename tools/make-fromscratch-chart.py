"""Draw the from-scratch training curve (val loss vs tokens seen) for the stage summary.

Same convention as make-charts.py: no matplotlib in the bundled runtime, so the chart is drawn
directly with Pillow. Data: 2026-10-22/23 full-corpus runs, 5 chained slices of TinyStories.

Usage: python make-fromscratch-chart.py
"""

import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "figures")

FONT_REG = "C:/Windows/Fonts/msyh.ttc"
FONT_BOLD = "C:/Windows/Fonts/msyhbd.ttc"

BLUE = (31, 59, 115)
GRAY = (140, 140, 140)
RED = (192, 64, 58)
INK = (34, 34, 34)
GRID = (216, 216, 216)

# 5 chained slices, cumulative tokens in millions (2026-10-24 shipped run)
TOKENS = [108.6, 217.3, 325.9, 434.5, 511.1]
VULKAN = [3.53136, 2.60664, 2.17682, 1.97437, 1.78994]   # this project's backend
CUDA = [3.53136, 2.60664, 2.17682, 1.97437, 1.78994]     # no separate CUDA run for this config


def main():
    W, H = 1180, 720
    L, R, T, B = 120, 60, 132, 96
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(FONT_BOLD, 30)
    f_lab = ImageFont.truetype(FONT_REG, 22)
    f_small = ImageFont.truetype(FONT_REG, 19)
    f_tick = ImageFont.truetype(FONT_REG, 18)

    d.text((L, 24), "从零训练：验证集 loss 随语料量的变化", font=f_title, fill=INK)
    d.text((L, 68), "8.34 M 参数 / 4,096 自训词表 / 恒定 lr 1e-6 / 5 片串接，共 511 M token",
           font=f_small, fill=(90, 90, 90))

    x0, x1 = L + 30, W - R
    y0, y1 = H - B, T + 20
    ymin, ymax = 1.5, 3.6

    def px(tok):
        return x0 + (tok / 550.0) * (x1 - x0)

    def py(val):
        return y0 - (val - ymin) / (ymax - ymin) * (y0 - y1)

    # grid + y ticks
    for i in range(5):
        v = ymin + i * (ymax - ymin) / 4
        y = py(v)
        d.line([(x0, y), (x1, y)], fill=GRID, width=1)
        d.text((L + 6, y - 10), f"{v:.1f}", font=f_tick, fill=(80, 80, 80))
    # x ticks
    for tok in (0, 100, 200, 300, 400, 500):
        x = px(tok)
        d.line([(x, y0), (x, y0 + 6)], fill=(120, 120, 120), width=1)
        d.text((x - 16, y0 + 12), f"{tok}M", font=f_tick, fill=(80, 80, 80))

    # release gate at val 3.0
    yg = py(3.0)
    for xx in range(int(x0), int(x1), 14):
        d.line([(xx, yg), (xx + 7, yg)], fill=RED, width=2)
    d.text((x1 - 232, yg - 30), "发布门槛 val ≤ 3.0", font=f_small, fill=RED)

    # internal reference (CUDA) dashed
    for i in range(len(CUDA) - 1):
        (ax, ay), (bx, by) = (px(TOKENS[i]), py(CUDA[i])), (px(TOKENS[i + 1]), py(CUDA[i + 1]))
        n = 26
        for k in range(n):
            if k % 2:
                continue
            t0, t1 = k / n, (k + 1) / n
            d.line([(ax + (bx - ax) * t0, ay + (by - ay) * t0),
                    (ax + (bx - ax) * t1, ay + (by - ay) * t1)], fill=GRAY, width=2)
    for i in range(len(CUDA)):
        x, y = px(TOKENS[i]), py(CUDA[i])
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=GRAY)

    # main line (Vulkan)
    pts = [(px(TOKENS[i]), py(VULKAN[i])) for i in range(len(VULKAN))]
    d.line(pts, fill=BLUE, width=4)
    for i, (x, y) in enumerate(pts):
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=BLUE)
        dy = -36 if i in (0, 2, 4) else 16
        d.text((x - 24, y + dy), f"{VULKAN[i]:.2f}", font=f_small, fill=BLUE)

    # legend
    lx, ly = x0 + 10, 96
    d.line([(lx, ly), (lx + 34, ly)], fill=BLUE, width=4)
    d.text((lx + 42, ly - 11), "Vulkan0（本项目）", font=f_lab, fill=INK)
    d.line([(lx + 240, ly), (lx + 274, ly)], fill=GRAY, width=2)
    d.text((lx + 282, ly - 11), " ", font=f_small, fill=GRAY)

    d.text((lx, y0 + 44), "注：5 片串接训练，片边界重置 AdamW 动量；各片 val 为该片 5% 留出。",
           font=f_small, fill=(110, 110, 110))

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig-fromscratch.png")
    img.save(path)
    print(f"  wrote {os.path.normpath(path)}  ({os.path.getsize(path) // 1024} KB)")


if __name__ == "__main__":
    raise SystemExit(main())
