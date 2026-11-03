"""Basit baseline'ları IDSSE'de bir-maçı-dışarıda-bırak ile değerlendirir.

Kullanım:
    .venv/Scripts/python scripts/baseline_degerlendir.py [--sablon]
Çıktı: outputs/sonuclar/baseline_idsse.csv (şablonla: baseline_sablon_idsse.csv) + konsola Markdown özet.
--sablon: yuvalar her katta eğitim maçlarından öğrenilen şablona göre sıralanır.

Metrik aralıkları: "gelecek" = son F kare (A ve B kıyaslanabilir), "tam" = tüm pencere (yalnız A).
Rol ortalamaları savunmayı görmediği için A ve B'de aynıdır; tek satır "A/B" olarak raporlanır.
Baseline'ların hiperparametresi olmadığından doğrulama maçı kullanılmaz.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from ghostdef import metrikler
from ghostdef.baseline.basit import RolOrtalamasi, SabitHiz, TopaGoreRolOrtalamasi
from ghostdef.data.bolme import mac_disari_katlar
from ghostdef.data.idsse import MACLAR
from ghostdef.data.ortak_format import N_TAKIM, TOP
from ghostdef.data.pencere import VERI_KOK, PencereVeriSeti
from ghostdef.data.sablon import Sablon

SONUC_KLASOR = Path(__file__).resolve().parents[1] / "outputs" / "sonuclar"
MODELLER = [("Rol ortalaması", RolOrtalamasi, "A/B"),
            ("Topa göre rol ort.", TopaGoreRolOrtalamasi, "A/B"),
            ("Sabit hız", SabitHiz, "B")]


def olc(tahmin, y, bas, son):
    mask = y["mask"][:, bas:son, N_TAKIM:TOP]
    return metrikler.hepsi(tahmin[:, :, bas:son], y["hedef"][:, bas:son], mask)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sablon", action="store_true")
    a = p.parse_args()
    ek = " (şablon)" if a.sablon else ""
    CIKIS = SONUC_KLASOR / ("baseline_sablon_idsse.csv" if a.sablon else "baseline_idsse.csv")
    satirlar = []
    for kat in mac_disari_katlar(MACLAR):
        sab = Sablon.ogren(kat["egitim"], VERI_KOK / "idsse") if a.sablon else None
        egitim = PencereVeriSeti(kat["egitim"], sablon=sab)
        test = PencereVeriSeti(kat["test"], kurulum="B", adim=10, sablon=sab)
        y = test.yigin()
        H, L = test.gecmis, test.uzunluk
        for ad, sinif, kurulum in MODELLER:
            tahmin = sinif().fit(egitim).tahmin(y)
            g = olc(tahmin, y, H, L)
            tam = olc(tahmin, y, 0, L) if kurulum != "B" else {}
            satirlar.append({"model": ad + ek, "kurulum": kurulum, "test_mac": kat["test"][0],
                             "pencere": len(test), "ADE": g["ADE"], "FDE": g["FDE"],
                             "ADE_tam": tam.get("ADE", np.nan)})
        print(f"{kat['test'][0]}: {len(test)} pencere tamam")

    CIKIS.parent.mkdir(parents=True, exist_ok=True)
    with open(CIKIS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)

    print(f"\nKat başına ADE (m, gelecek {F_S(test)} s):\n")
    print("| Model | " + " | ".join(MACLAR) + " |")
    print("|---|" + "---|" * len(MACLAR))
    for ad, _, _ in MODELLER:
        d = {s["test_mac"]: s["ADE"] for s in satirlar if s["model"] == ad + ek}
        print(f"| {ad} | " + " | ".join(f"{d[m]:.2f}" for m in MACLAR) + " |")

    print("\nÖzet (7 kat, ortalama ± std, metre):\n")
    print("| Model | Kurulum | ADE (gelecek) | FDE | ADE (tam pencere) |")
    print("|---|---|---|---|---|")
    for ad, _, kurulum in MODELLER:
        s = [r for r in satirlar if r["model"] == ad + ek]
        ms = lambda k: f"{np.mean([r[k] for r in s]):.2f} ± {np.std([r[k] for r in s]):.2f}"
        print(f"| {ad} | {kurulum} | {ms('ADE')} | {ms('FDE')} | "
              f"{ms('ADE_tam') if kurulum != 'B' else '—'} |")
    print(f"\nCSV: {CIKIS}")


def F_S(ds):
    return ds.gelecek / 10


if __name__ == "__main__":
    main()
