"""Ortak format bir segmenti 2D mini harita animasyonuna çevirir (hücum soldan sağa).

Kullanım:
    .venv/Scripts/python scripts/mini_harita.py J03WMX            # 15–25 s'lik en ileri giden segment
    .venv/Scripts/python scripts/mini_harita.py J03WMX --segment 42
    .venv/Scripts/python scripts/mini_harita.py 3840 --kaynak pff   # tahmin konumlar içi boş
Çıktı: outputs/mini_harita/<maç>_seg<no>.gif (ffmpeg varsa .mp4 de).
Hayalet katmanı: ciz(..., hayalet=[N, 11, 2]) verildiğinde savunmanın yanında içi boş çizilir.
"""
import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from mplsoccer import Pitch

KOK = Path(__file__).resolve().parents[1]
HUCUM, SAVUNMA, TOP_R, HAYALET = "#d1495b", "#00798c", "#f2f2f2", "#00798c"


def segment_sec(d, min_s=15.0, maks_s=25.0) -> int:
    """Uzunluğu aralıkta olan ve topun en çok ileri gittiği segment."""
    seg, hz, x = d["segment_id"], float(d["hz"]), d["pos"][:, 22, 0]
    en_iyi, puan = 0, -np.inf
    for s in np.unique(seg):
        k = seg == s
        if min_s * hz <= k.sum() <= maks_s * hz and x[k].max() - x[k][0] > puan:
            en_iyi, puan = int(s), x[k].max() - x[k][0]
    return en_iyi


def ciz(d, s: int, cikis: Path, hayalet=None) -> list:
    k = d["segment_id"] == s
    pos, mask, z = d["pos"][k], d["mask"][k], d["top_z"][k]
    # Görünür = dolu daire; tahmin (PFF ESTIMATED) = takım renginde içi boş daire.
    gor = d["gorunur"][k] if "gorunur" in d else mask
    tahmin = mask & ~gor
    t, hz = d["t"][k], float(d["hz"])
    pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68,
                  pitch_color="#22312b", line_color="#c7d5cc")
    fig, ax = pitch.draw(figsize=(10, 6.8))
    h = ax.scatter([], [], s=140, c=HUCUM, edgecolors="white", zorder=3, label="Hücum")
    v = ax.scatter([], [], s=140, c=SAVUNMA, edgecolors="white", zorder=3, label="Savunma")
    th = ax.scatter([], [], s=140, facecolors="none", edgecolors=HUCUM, linewidths=2, zorder=3)
    tv = ax.scatter([], [], s=140, facecolors="none", edgecolors=SAVUNMA, linewidths=2, zorder=3)
    if tahmin[:, :22].any():
        ax.scatter([], [], s=140, facecolors="none", edgecolors="#888888", linewidths=2, label="Tahmin")
    g = ax.scatter([], [], s=140, facecolors="none", edgecolors=HAYALET, linestyles="--",
                   zorder=2, label="Hayalet") if hayalet is not None else None
    b = ax.scatter([], [], s=50, c=TOP_R, edgecolors="black", zorder=4)
    baslik = ax.set_title("", color="black")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.08), ncol=4, frameon=False)

    def kare(i):
        h.set_offsets(pos[i, :11][gor[i, :11]])
        v.set_offsets(pos[i, 11:22][gor[i, 11:22]])
        th.set_offsets(pos[i, :11][tahmin[i, :11]])
        tv.set_offsets(pos[i, 11:22][tahmin[i, 11:22]])
        b.set_offsets(pos[i, 22:23])
        b.set_sizes([50 + 12 * z[i]])  # top yüksekliği boyutla
        if g is not None:
            g.set_offsets(hayalet[i])
        baslik.set_text(f"{d['mac_id']} · segment {s} · {int(d['period'][k][0])}. yarı "
                        f"{int(t[i] // 60):02d}:{t[i] % 60:04.1f} · top z={z[i]:.1f} m")
        return h, v, th, tv, b

    anim = FuncAnimation(fig, kare, frames=len(pos), interval=1000 / hz)
    cikis.parent.mkdir(parents=True, exist_ok=True)
    yazilan = [cikis.with_suffix(".gif")]
    anim.save(yazilan[0], writer=PillowWriter(fps=hz))
    if shutil.which("ffmpeg"):
        yazilan.append(cikis.with_suffix(".mp4"))
        anim.save(yazilan[1], writer=FFMpegWriter(fps=hz))
    plt.close(fig)
    return yazilan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mac")
    ap.add_argument("--segment", type=int)
    ap.add_argument("--kaynak", choices=["idsse", "pff"], default="idsse")
    a = ap.parse_args()
    d = np.load(KOK / "data" / "processed" / a.kaynak / f"{a.mac}.npz")
    s = a.segment if a.segment is not None else segment_sec(d)
    for y in ciz(d, s, KOK / "outputs" / "mini_harita" / f"{a.mac}_seg{s}"):
        print(y)
