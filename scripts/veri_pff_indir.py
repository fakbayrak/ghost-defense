"""PFF FC (Gradient Sports) 2022 Dünya Kupası verisini data/raw/pff/ altına indirir.

Kaynak: PFF'nin paylaştığı herkese açık Google Drive klasörü (64 maç tracking + olay + meta + kadro).
Lisans/kullanım koşulu klasörde yazmıyor ("PFFA Acquisition LLC proprietary information").
Ham veri git'e eklenmez, yeniden dağıtılmaz.

Kullanım:
    .venv/Scripts/python scripts/veri_pff_indir.py              # 64 maç + belgeler
    .venv/Scripts/python scripts/veri_pff_indir.py 3840 3812    # yalnız bu maçlar + belgeler
    .venv/Scripts/python scripts/veri_pff_indir.py --liste      # Drive listesini tazele
Var olan dosyalar atlanır. Eski olay sürümleri (Event Data/<tarih>/) indirilmez.
Gerektirir: gdown (pip install gdown).
"""
import sys
from pathlib import Path

import gdown

KLASOR_ID = "1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa"
HEDEF = Path(__file__).resolve().parents[1] / "data" / "raw" / "pff"
LISTE = HEDEF / "_drive_liste.tsv"
MAC_KLASORLERI = ["Tracking Data", "Event Data", "Metadata", "Rosters"]


def liste_al(tazele: bool = False) -> list[tuple[str, str]]:
    """(drive_id, göreli_yol) listesi; yol ayırıcı '/'. Önbellek: _drive_liste.tsv."""
    if tazele or not LISTE.exists():
        dosyalar = gdown.download_folder(id=KLASOR_ID, skip_download=True, quiet=True)
        with open(LISTE, "w", encoding="utf-8") as fh:
            for f in dosyalar:
                fh.write(f"{f.id}\t{f.path}\n")
    satirlar = [s.rstrip("\n").split("\t") for s in open(LISTE, encoding="utf-8")]
    return [(fid, yol.replace("\\", "/")) for fid, yol in satirlar]


def secilsin_mi(yol: str, maclar: set[str] | None) -> bool:
    parcalar = yol.split("/")
    if len(parcalar) == 1:  # kök: csv'ler, değişiklik günlüğü
        return True
    if len(parcalar) != 2 or parcalar[0] not in MAC_KLASORLERI:
        return False  # Event Data/<tarih>/... eski sürümler
    ad = parcalar[1]
    if ad.endswith(".pdf"):  # spesifikasyonlar
        return True
    return maclar is None or ad.split(".")[0] in maclar


def indir(fid: str, yol: str) -> None:
    cikti = HEDEF / yol
    if yol.endswith(".docx"):  # Google Docs belgesi → docx olarak dışa aktar
        url, kimlik = f"https://docs.google.com/document/d/{fid}/export?format=docx", None
    else:
        url, kimlik = None, fid
    if cikti.exists() and cikti.stat().st_size > 0:
        print(f"var: {yol}")
        return
    cikti.parent.mkdir(parents=True, exist_ok=True)
    gecici = cikti.with_name(cikti.name + ".part")
    gdown.download(url=url, id=kimlik, output=str(gecici), quiet=True)
    gecici.rename(cikti)
    print(f"{cikti.stat().st_size / 1e6:8.2f} MB  {yol}", flush=True)


if __name__ == "__main__":
    HEDEF.mkdir(parents=True, exist_ok=True)
    argumanlar = sys.argv[1:]
    tazele = "--liste" in argumanlar
    maclar = {a for a in argumanlar if a != "--liste"} or None
    dosyalar = liste_al(tazele)
    if tazele and maclar is None and len(argumanlar) == 1:
        print(f"{len(dosyalar)} dosya listelendi → {LISTE}")
        sys.exit(0)
    secilen = [(fid, yol) for fid, yol in dosyalar if secilsin_mi(yol, maclar)]
    print(f"{len(secilen)} dosya seçildi")
    hatalar = []
    for fid, yol in secilen:
        try:
            indir(fid, yol)
        except Exception as e:  # Drive kota/ağ hatası: devam et, sonda raporla
            hatalar.append(yol)
            print(f"HATA: {yol}: {e}", flush=True)
    if hatalar:
        print(f"{len(hatalar)} dosya inmedi; betiği tekrar çalıştırın (var olanlar atlanır).")
        sys.exit(1)
