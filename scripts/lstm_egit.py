"""Rol başına LSTM'i IDSSE'de bir-maçı-dışarıda-bırak ile eğitir ve test eder.

Kullanım:
    .venv/Scripts/python scripts/lstm_egit.py                       # 7 kat × A, B
    .venv/Scripts/python scripts/lstm_egit.py --kurulum B --kat J03WMX --epoch 3
    .venv/Scripts/python scripts/lstm_egit.py --sablon     # şablon yuvaları → lstm_sablon/
Çıktılar: outputs/modeller/lstm/<kurulum>/<test_mac>.pt (+ _gunluk.csv),
          outputs/sonuclar/lstm_idsse.csv (aynı model/kurulum/maç satırı varsa üzerine yazılır).
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from ghostdef.data.bolme import mac_disari_katlar
from ghostdef.data.idsse import MACLAR
from ghostdef.data.pencere import VERI_KOK, PencereVeriSeti
from ghostdef.data.sablon import Sablon
from ghostdef.egitim import EgitimAyari, cihaza, degerlendir, egit, kaydet
from ghostdef.modeller.lstm import RolLSTM

KOK = Path(__file__).resolve().parents[1] / "outputs"
SONUC = KOK / "sonuclar" / "lstm_idsse.csv"
MODEL_ADI = "LSTM (rol başına)"
KLASOR_ADI = "lstm"


def calistir(kat, kurulum, ayar, cihaz, sablonlu=False):
    sab = Sablon.ogren(kat["egitim"], VERI_KOK / "idsse") if sablonlu else None
    yiginlar = {}
    for ad, maclar, adim in [("egitim", kat["egitim"], 5), ("dogrulama", kat["dogrulama"], 10),
                             ("test", kat["test"], 10)]:
        ds = PencereVeriSeti(maclar, kurulum=kurulum, adim=adim, sablon=sab)
        yiginlar[ad] = cihaza(ds.yigin(), cihaz)
    H, L = ds.gecmis, ds.uzunluk
    ayar.kayip_bas, ayar.metrik_bas = (0 if kurulum == "A" else H), H
    test_mac = kat["test"][0]
    hedef_klasor = KOK / "modeller" / KLASOR_ADI / kurulum
    t0 = time.time()
    model = RolLSTM(kurulum).to(cihaz)
    model, gunluk, en_iyi_ep = egit(model, yiginlar["egitim"], yiginlar["dogrulama"], ayar,
                                    gunluk_yolu=hedef_klasor / f"{test_mac}_gunluk.csv")
    ek = {"kat": kat}
    if sab is not None:
        ek["sablon"] = {"hucum": sab.hucum.tolist(), "savunma": sab.savunma.tolist()}
    kaydet(hedef_klasor / f"{test_mac}.pt", model, ayar, ek)
    g = degerlendir(model, yiginlar["test"], H, L)
    tam = degerlendir(model, yiginlar["test"], 0, L) if kurulum == "A" else {}
    sure = time.time() - t0
    print(f"{kurulum} {test_mac}: ADE {g['ADE']:.2f} FDE {g['FDE']:.2f} | en iyi epoch {en_iyi_ep}"
          f"/{len(gunluk)} | {sure:.0f} s")
    return {"model": MODEL_ADI, "kurulum": kurulum, "test_mac": test_mac,
            "pencere": len(yiginlar["test"]["top"]), "ADE": g["ADE"], "FDE": g["FDE"],
            "ADE_tam": tam.get("ADE", np.nan), "en_iyi_epoch": en_iyi_ep, "sure_s": round(sure)}


def sonuclari_yaz(yeni):
    eski = []
    if SONUC.exists():
        with open(SONUC, encoding="utf-8") as f:
            eski = list(csv.DictReader(f))
    anahtar = {(r["model"], r["kurulum"], r["test_mac"]) for r in yeni}
    satirlar = [r for r in eski if (r["model"], r["kurulum"], r["test_mac"]) not in anahtar] + yeni
    SONUC.parent.mkdir(parents=True, exist_ok=True)
    with open(SONUC, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(yeni[0]))
        w.writeheader()
        w.writerows(satirlar)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kurulum", default="hepsi", choices=["A", "B", "hepsi"])
    p.add_argument("--kat", default="hepsi", help="test maçı kimliği veya 'hepsi'")
    p.add_argument("--epoch", type=int, default=EgitimAyari.epoch)
    p.add_argument("--devam", action="store_true", help="CSV'de sonucu olan kat/kurulumları atla")
    p.add_argument("--sablon", action="store_true", help="yuvaları kat başına öğrenilen şablona göre sırala")
    a = p.parse_args()
    global SONUC, MODEL_ADI, KLASOR_ADI
    if a.sablon:
        SONUC = KOK / "sonuclar" / "lstm_sablon_idsse.csv"
        MODEL_ADI, KLASOR_ADI = "LSTM (rol başına, şablon)", "lstm_sablon"
    bitmis = set()
    if a.devam and SONUC.exists():
        with open(SONUC, encoding="utf-8") as f:
            bitmis = {(r["kurulum"], r["test_mac"]) for r in csv.DictReader(f) if r["model"] == MODEL_ADI}
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {cihaz}" + (f" ({torch.cuda.get_device_name()})" if cihaz == "cuda" else ""))
    kurulumlar = ["A", "B"] if a.kurulum == "hepsi" else [a.kurulum]
    katlar = [k for k in mac_disari_katlar(MACLAR) if a.kat in ("hepsi", k["test"][0])]
    if not katlar:
        raise SystemExit(f"Bilinmeyen maç: {a.kat}")
    yeni = []
    for kurulum in kurulumlar:
        for kat in katlar:
            if (kurulum, kat["test"][0]) in bitmis:
                print(f"{kurulum} {kat['test'][0]}: zaten var, atlandı")
                continue
            yeni.append(calistir(kat, kurulum, EgitimAyari(epoch=a.epoch), cihaz, a.sablon))
            sonuclari_yaz(yeni)   # her kattan sonra yaz: yarıda kesilirse kayıp olmasın
    print(f"CSV: {SONUC}")


if __name__ == "__main__":
    main()
