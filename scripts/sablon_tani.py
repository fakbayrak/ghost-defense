"""Şablon eşleştirme tanısı: 7 katın şablonları (kararlılık) + maç başına şekil hatası (eski vs şablon).

Kullanım:
    .venv/Scripts/python scripts/sablon_tani.py
Çıktı: outputs/sablon/sablonlar.png + konsola Markdown tablo.

Şekil hatası = oyuncunun segment ortalama şeklinin (takım merkezine göre) atandığı rolün ortalamasına
uzaklığı (m). Eski: rol grubu sırası, rol ortalaması eğitim maçlarından. Yeni: şablon ataması.
Her kat için yalnız test maçı ölçülür; şablon/ortalama o katın eğitim maçlarından.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch

from ghostdef.data.bolme import mac_disari_katlar
from ghostdef.data.idsse import MACLAR
from ghostdef.data.pencere import VERI_KOK
from ghostdef.data.sablon import Sablon, sablona_ata, segment_sekilleri

KOK = Path(__file__).resolve().parents[1]
KLASOR = VERI_KOK / "idsse"
RENK = {0: "#d1495b", 1: "#00798c"}
TARAF_ADI = {0: "Hücum", 1: "Savunma"}


def sekiller(maclar, taraf):
    p = [segment_sekilleri(d["pos"], d["mask"], d["rol"], d["segment_id"], taraf)[:2]
         for d in (np.load(KLASOR / f"{m}.npz") for m in maclar)]
    return np.concatenate([a for a, _ in p]), np.concatenate([b for _, b in p])


def hata(sekil, gecerli, atama, ort):
    """Atanan rolün ortalamasına ortalama uzaklık (geçerli roller üzerinden)."""
    al = np.take_along_axis(sekil, np.maximum(atama, 0)[..., None], axis=1)
    d = np.linalg.norm(al - ort[None], axis=-1)
    return float(d[atama >= 0].mean())


def main():
    cikis = KOK / "outputs" / "sablon"
    cikis.mkdir(parents=True, exist_ok=True)
    pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68, line_color="#888")
    fig, eksenler = pitch.draw(nrows=1, ncols=2, figsize=(14, 5))
    satirlar = []
    for kat in mac_disari_katlar(MACLAR):
        sab = Sablon.ogren(kat["egitim"], KLASOR)
        test = kat["test"][0]
        for taraf, s in ((0, sab.hucum), (1, sab.savunma)):
            ax = eksenler[taraf]
            ax.scatter(s[:, 0] + 52.5, s[:, 1] + 34, s=40, c=RENK[taraf], alpha=0.35, edgecolors="none")
            # eski: rol grubu sırası (sekil zaten bu sırada), ortalama eğitimden
            se, ge = sekiller(kat["egitim"], taraf)
            eski_ort = (se * ge[..., None]).sum(0) / np.maximum(ge.sum(0), 1)[:, None]
            st, gt = sekiller([test], taraf)
            eski_atama = np.where(gt, np.arange(10)[None], -1)
            satirlar.append((test, taraf, hata(st, gt, eski_atama, eski_ort),
                             hata(st, gt, sablona_ata(st, gt, s), s)))
    son = Sablon.ogren(MACLAR[1:], KLASOR)   # numaralar için bir şablon (J03WMX katı)
    for taraf, s in ((0, son.hucum), (1, son.savunma)):
        for r, (x, y) in enumerate(s):
            eksenler[taraf].text(x + 52.5, y + 34 + 2.2, str(r + 1), ha="center", fontsize=9)
        eksenler[taraf].set_title(f"{TARAF_ADI[taraf]} şablonu (7 kat üst üste, merkez = saha ortası)")
    fig.savefig(cikis / "sablonlar.png", dpi=130, bbox_inches="tight")

    print("Şekil hatası (m): eski sıra → şablon\n")
    print("| Test maçı | Hücum eski | Hücum şablon | Savunma eski | Savunma şablon |")
    print("|---|---|---|---|---|")
    for m in MACLAR:
        h = {t: (e, y) for mm, t, e, y in satirlar if mm == m}
        print(f"| {m} | {h[0][0]:.2f} | {h[0][1]:.2f} | {h[1][0]:.2f} | {h[1][1]:.2f} |")
    print(f"\nŞekil: {cikis / 'sablonlar.png'}")


if __name__ == "__main__":
    main()
