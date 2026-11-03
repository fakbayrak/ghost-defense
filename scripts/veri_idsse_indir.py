"""IDSSE ham XML dosyalarını data/raw/idsse/ altına indirir (CC-BY 4.0, Bassek vd. 2025).

Kullanım:
    .venv/Scripts/python scripts/veri_idsse_indir.py            # 7 maç
    .venv/Scripts/python scripts/veri_idsse_indir.py J03WMX     # tek maç
Var olan dosyalar atlanır. Ham veri git'e eklenmez, yeniden dağıtılmaz.
"""
import sys
import urllib.request
from pathlib import Path

from kloppy._providers.sportec import get_IDSSE_url

MACLAR = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
TURLER = ["meta", "tracking", "event"]
HEDEF = Path(__file__).resolve().parents[1] / "data" / "raw" / "idsse"


def indir(mac_id: str) -> None:
    for tur in TURLER:
        url = get_IDSSE_url(mac_id, tur)
        yol = HEDEF / url.rsplit("/", 1)[1]
        if yol.exists():
            print(f"var: {yol.name}")
            continue
        gecici = yol.with_suffix(".part")
        print(f"indiriliyor: {yol.name}")
        urllib.request.urlretrieve(url, gecici)
        gecici.rename(yol)
        print(f"  {yol.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    HEDEF.mkdir(parents=True, exist_ok=True)
    for mac in sys.argv[1:] or MACLAR:
        indir(mac)
