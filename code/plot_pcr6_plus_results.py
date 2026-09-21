from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parents[1] / "fig_4_pcr6_plus_results.png"
W, H = 2000, 820
image = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(image)


def font(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


TITLE = font(37, True)
SUBTITLE = font(27, True)
LABEL = font(22)
SMALL = font(18)
SMALL_BOLD = font(18, True)

draw.text((W // 2, 28), "PCR6+ improves PCR6 under controlled partial ignorance, but not on RAMDocs", fill="#17212B", font=TITLE, anchor="ma")

rules = ["Dempster", "PCR6", "PCR6+"]
colors = ["#5B6B7A", "#D17A22", "#2166AC"]
panels = [
    (
        "A. Controlled multi-source benchmark",
        ["Partial\nignorance", "Vacuous\nsource", "Near-vacuous\nsource"],
        [[0.1877638, 0.3185086, 0.2732892], [0.1877638, 0.4107633, 0.2732892], [0.1880495, 0.4041948, 0.2738958]],
        0.46,
    ),
    (
        "B. RAMDocs external validation",
        ["All eligible\n(n=428)", "Applicability\n(n=260)"],
        [[0.2998027, 0.2363021, 0.2574241], [0.3263318, 0.2353634, 0.2628501]],
        0.38,
    ),
]

for panel_index, (panel_title, categories, values, ymax) in enumerate(panels):
    left = 95 + panel_index * 980
    right = left + 855
    top = 150
    bottom = 685
    draw.text((left, 103), panel_title, fill="#17212B", font=SUBTITLE)
    draw.line((left, bottom, right, bottom), fill="#38444F", width=2)
    draw.line((left, top, left, bottom), fill="#38444F", width=2)
    for tick in range(5):
        value = ymax * tick / 4
        y = bottom - (bottom - top) * tick / 4
        draw.line((left, y, right, y), fill="#D9DEE3", width=1)
        draw.text((left - 12, y), f"{value:.2f}", fill="#38444F", font=SMALL, anchor="rm")
    group_width = (right - left) / len(categories)
    bar_width = 62
    for group_index, (category, group_values) in enumerate(zip(categories, values)):
        center = left + group_width * (group_index + 0.5)
        for rule_index, value in enumerate(group_values):
            x0 = center + (rule_index - 1) * 75 - bar_width / 2
            x1 = x0 + bar_width
            y0 = bottom - value / ymax * (bottom - top)
            draw.rectangle((x0, y0, x1, bottom), fill=colors[rule_index])
            draw.text(((x0 + x1) / 2, y0 - 9), f"{value:.3f}", fill="#17212B", font=SMALL_BOLD, anchor="ms")
        for line_index, line in enumerate(category.split("\n")):
            draw.text((center, bottom + 20 + 24 * line_index), line, fill="#17212B", font=SMALL, anchor="ma")

legend_y = 782
legend_start = 690
for idx, (rule, color) in enumerate(zip(rules, colors)):
    x = legend_start + idx * 240
    draw.rectangle((x, legend_y - 12, x + 28, legend_y + 16), fill=color)
    draw.text((x + 40, legend_y + 2), rule, fill="#17212B", font=LABEL, anchor="lm")

image.save(OUT, dpi=(220, 220))
print(OUT)

