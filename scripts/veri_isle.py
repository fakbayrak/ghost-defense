"""Ham maçları ortak formata işler: data/processed/<kaynak>/<maç>.npz + kontrol raporu.

Kullanım:
    .venv/Scripts/python scripts/veri_isle.py                     # IDSSE, indirilmiş tüm maçlar
    .venv/Scripts/python scripts/veri_isle.py J03WMX
    .venv/Scripts/python scripts/veri_isle.py --kaynak pff        # PFF, 64 maç (6 süreç paralel)
    .venv/Scripts/python scripts/veri_isle.py --kaynak pff 3840
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ghostdef.data import idsse, pff
from ghostdef.data.ortak_format import kontrol, ortak_formata, segmentler, yeniden_ornekle

KOK = Path(__file__).resolve().parents[1] / "data" / "processed"
KAYNAKLAR = {"idsse": idsse, "pff": pff}


def isle(kaynak: str, mac_id: str) -> dict:
    t0 = time.time()
    ham = KAYNAKLAR[kaynak].yukle(mac_id)
    on = yeniden_ornekle(ham, hz=10.0)
    seg = segmentler(on, min_sure=2.0)
    of = ortak_formata(on, seg)
    np.savez_compressed(KOK / kaynak / f"{mac_id}.npz", **of)
    rapor = {"mac": mac_id, "oyuncu_sayisi": len(ham.oyuncu_id),
             "ham_kare": len(ham.t), "kare_10hz": len(on.t),
             "donemler": sorted(int(p) for p in np.unique(ham.period)),
             "top_z_var": bool(np.isfinite(ham.top[:, 2]).any()),
             "top_z_maks": float(np.nanmax(ham.top[:, 2])),
             **kontrol(of), "sure_s": round(time.time() - t0, 1)}
    print(json.dumps(rapor, ensure_ascii=False), flush=True)
    return rapor


def mevcut_maclar(kaynak: str) -> list[str]:
    if kaynak == "pff":
        return pff.maclar()
    return [m for m in idsse.MACLAR if any(idsse.KOK.glob(f"DFL_04_03*{m}.xml"))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("maclar", nargs="*")
    ap.add_argument("--kaynak", choices=sorted(KAYNAKLAR), default="idsse")
    ap.add_argument("--surec", type=int, default=6, help="PFF için paralel süreç sayısı")
    a = ap.parse_args()
    (KOK / a.kaynak).mkdir(parents=True, exist_ok=True)
    maclar = a.maclar or mevcut_maclar(a.kaynak)
    if a.kaynak == "pff" and len(maclar) > 1 and a.surec > 1:
        with ProcessPoolExecutor(max_workers=a.surec) as ex:
            raporlar = list(ex.map(isle, [a.kaynak] * len(maclar), maclar))
    else:  # IDSSE maç başına ≈1,7 GB RAM → sıralı
        raporlar = [isle(a.kaynak, m) for m in maclar]
    # Tek maçlık çalıştırma önceki raporu ezmesin: maç bazında birleştir.
    yol = KOK / a.kaynak / "kontrol_raporu.json"
    eski = {r["mac"]: r for r in json.loads(yol.read_text(encoding="utf-8"))} if yol.exists() else {}
    eski.update({r["mac"]: r for r in raporlar})
    yol.write_text(json.dumps(list(eski.values()), ensure_ascii=False, indent=1), encoding="utf-8")
