#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Иконка Herald 3.0: раскрытая книга, из которой поднимается звуковая волна.

Рисуется кодом (PIL) в 4-кратном разрешении и уменьшается - края гладкие.
Сетка значков macOS: скруглённый квадрат 824 из 1024, тень, градиент.
Цвета - акцент нового окна (тёмно-синий LKD) и тёплое золото.

    .venv/bin/python "Инструменты разработки/make_icon.py"
    -> assets/icon.png (1024), assets/Herald.iconset/, assets/Herald.icns
"""
import math, os, shutil, subprocess
from PIL import Image, ImageDraw, ImageFilter

S = 4
N = 1024 * S
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(len(a)))


def vgrad(w, h, top, bot):
    g = Image.new("RGBA", (w, h))
    d = ImageDraw.Draw(g)
    for y in range(h):
        d.line([(0, y), (w, y)], fill=lerp(top, bot, y / (h - 1)))
    return g


def squircle_mask(size, box, r):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle(box, radius=r, fill=255)
    return m


def bez(p0, p1, p2, n=80):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
            for t in (i / n for i in range(n + 1))]


def sc(pts):
    return [(x * S, y * S) for x, y in pts]


def draw():
    img = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    m0, m1 = 100, 924                          # поле значка по сетке Apple
    R = 185

    # тень под значком
    sh = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([m0 * S, (m0 + 14) * S, m1 * S, (m1 + 14) * S],
                                         radius=R * S, fill=(0, 0, 0, 110))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(22 * S)))

    # фон: глубокий синий с лёгким свечением сверху
    bg = vgrad(N, N, (44, 74, 128, 255), (14, 24, 46, 255))
    glow = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([250 * S, 120 * S, 774 * S, 560 * S], fill=(110, 150, 220, 70))
    bg.alpha_composite(glow.filter(ImageFilter.GaussianBlur(90 * S)))
    mask = squircle_mask(N, [m0 * S, m0 * S, m1 * S, m1 * S], R * S)
    img.paste(bg, (0, 0), mask)

    d = ImageDraw.Draw(img)
    cx = 512

    # звуковая волна над корешком - мягкий горб, золото
    heights = [56, 104, 158, 206, 236, 206, 158, 104, 56]
    bw, gap, base = 28, 20, 540
    total = len(heights) * bw + (len(heights) - 1) * gap
    x = cx - total / 2
    for h in heights:
        bar = vgrad(bw * S, h * S, (255, 220, 150, 255), (230, 150, 60, 255))
        bm = Image.new("L", (bw * S, h * S), 0)
        ImageDraw.Draw(bm).rounded_rectangle([0, 0, bw * S - 1, h * S - 1], radius=bw * S // 2, fill=255)
        img.paste(bar, (int(x * S), int((base - h) * S)), bm)
        x += bw + gap

    def page_pts(side, dy=0, grow=0):
        ox = cx + side * (300 + grow)
        top = bez((cx, 604 + dy), (cx + side * 140, 540 + dy), (ox, 566 + dy))
        bottom = bez((ox, 724 + dy), (cx + side * 150, 696 + dy), (cx, 756 + dy))
        return top + bottom

    # стопка страниц под верхними - объём книги; самая нижняя - тёмная обложка
    for dy, grow, col in ((26, 14, (12, 20, 40, 255)), (16, 8, (196, 186, 166, 255)),
                          (8, 4, (222, 213, 194, 255))):
        for side in (-1, 1):
            d.polygon(sc(page_pts(side, dy, grow)), fill=col)
    for side in (-1, 1):
        page = vgrad(N, N, (255, 251, 242, 255), (228, 220, 202, 255)) if side < 0 else \
            vgrad(N, N, (247, 241, 229, 255), (216, 207, 188, 255))
        pm = Image.new("L", (N, N), 0)
        ImageDraw.Draw(pm).polygon(sc(page_pts(side)), fill=255)
        img.paste(page, (0, 0), pm)
        # три строки текста, повторяют изгиб страницы и не касаются краёв
        for k in range(3):
            yy = 628 + k * 32
            p0 = (cx + side * 46, yy + 2)
            p1 = (cx + side * 150, yy - 40)
            p2 = (cx + side * 258, yy - 22)
            d.line(sc(bez(p0, p1, p2, 40)), fill=(178, 168, 148, 255), width=7 * S, joint="curve")
    # корешок
    d.line(sc([(cx, 606), (cx, 756)]), fill=(200, 140, 62, 255), width=8 * S)

    # тонкий блик по верхнему краю значка
    hl = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    ImageDraw.Draw(hl).rounded_rectangle([m0 * S, m0 * S, m1 * S, m1 * S], radius=R * S,
                                         outline=(255, 255, 255, 40), width=3 * S)
    img.alpha_composite(hl)
    return img.resize((1024, 1024), Image.LANCZOS)


def main():
    icon = draw()
    a = os.path.join(ROOT, "assets")
    icon.save(os.path.join(a, "icon.png"))
    iset = os.path.join(a, "Herald.iconset")
    shutil.rmtree(iset, ignore_errors=True)
    os.makedirs(iset)
    for sz in (16, 32, 128, 256, 512):
        icon.resize((sz, sz), Image.LANCZOS).save(os.path.join(iset, f"icon_{sz}x{sz}.png"))
        icon.resize((sz * 2, sz * 2), Image.LANCZOS).save(os.path.join(iset, f"icon_{sz}x{sz}@2x.png"))
    subprocess.run(["iconutil", "-c", "icns", iset, "-o", os.path.join(a, "Herald.icns")], check=True)
    print("готово:", os.path.join(a, "icon.png"))


if __name__ == "__main__":
    main()
