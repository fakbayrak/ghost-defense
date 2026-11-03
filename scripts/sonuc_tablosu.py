"""outputs/sonuclar/*_<veri>.csv dosyalarını birleştirip Markdown özet basar.

Kullanım:
    .venv/Scripts/python scripts/sonuc_tablosu.py            # IDSSE
"""
import csv
import sys
from pathlib import Path

import numpy as np

KLASOR = Path(__file__).resolve().parents[1] / "outputs" / "sonuclar"
SIRA = ["baseline", "lstm", "transformer", "cvae", "difuzyon"]


def oku(veri: str) -> list:
    dosyalar = sorted(KLASOR.glob(f"*_{veri}.csv"),
                      key=lambda p: (SIRA.index(p.stem.split("_")[0]) if p.stem.split("_")[0] in SIRA
                                     else len(SIRA), p.name))
    satirlar = []
    for d in dosyalar:
        with open(d, encoding="utf-8") as f:
            satirlar += list(csv.DictReader(f))
    return satirlar


def ms(degerler):
    v = np.array([float(x) for x in degerler if x not in ("", "nan")])
    return f"{v.mean():.2f} ± {v.std():.2f}" if len(v) else "—"


def main(veri: str = "idsse"):
    satirlar = oku(veri)
    if not satirlar:
        raise SystemExit(f"{KLASOR} altında *_{veri}.csv yok")
    gruplar = list(dict.fromkeys((r["model"], r["kurulum"]) for r in satirlar))
    maclar = list(dict.fromkeys(r["test_mac"] for r in satirlar))

    print(f"Özet ({len(maclar)} kat, ortalama ± std, metre; ADE/FDE son 3 s):\n")
    print("| Model | Kurulum | Kat | ADE | FDE | ADE (tam pencere) |")
    print("|---|---|---|---|---|---|")
    for model, kurulum in gruplar:
        s = [r for r in satirlar if r["model"] == model and r["kurulum"] == kurulum]
        print(f"| {model} | {kurulum} | {len(s)} | {ms(r['ADE'] for r in s)} | "
              f"{ms(r['FDE'] for r in s)} | {ms(r['ADE_tam'] for r in s)} |")

    print("\nKat başına ADE (m):\n")
    print("| Model | Kurulum | " + " | ".join(maclar) + " |")
    print("|---|---|" + "---|" * len(maclar))
    for model, kurulum in gruplar:
        d = {r["test_mac"]: float(r["ADE"]) for r in satirlar
             if r["model"] == model and r["kurulum"] == kurulum}
        print(f"| {model} | {kurulum} | " + " | ".join(f"{d[m]:.2f}" if m in d else "—" for m in maclar) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:])
