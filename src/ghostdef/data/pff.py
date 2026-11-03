"""PFF FC / Gradient Sports WC 2022 (Sportlogiq yayın tracking'i, 29,97 fps) → HamMac.

Kararlar (plan 2026-11-03-pff-okuyucu):
- Konumlar yumuşatılmış alanlardan (`*PlayersSmoothed`, `ballsSmoothed`); ham izler etkin 15 Hz ve
  fiziksel olmayan sıçramalı. Görünürlük (`visibility` VISIBLE/ESTIMATED) → HamMac.gorunur.
- Oyunda durumu olay durum makinesiyle: OUT/END/VID → ölü; OTB/başlama vuruşu → oyunda.
  Sahiplik `game_event.home_ball`, sonraki olaya kadar geçerli. Topsuz kareler (kamera dışı)
  ortak formattaki segment kuralında (top NaN → segment dışı) zaten atılır.
- Aynı frameNum'lu yinelenen kareler (birden çok sahip olayı) atlanır; olay durumu yine işlenir.
kloppy'nin PFF okuyucusu görünürlüğü attığı ve dosyayı tümüyle belleğe aldığı için kullanılmıyor.
"""
import bz2
import json
import warnings
from pathlib import Path

import numpy as np

from .ortak_format import SAHA_X, SAHA_Y, HamMac

KOK = Path(__file__).resolve().parents[3] / "data" / "raw" / "pff"
OLU = {"OUT", "END", "VID"}
CANLI = {"FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF", "OTB"}


def maclar(kok: Path = KOK) -> list[str]:
    """Tracking dosyası indirilmiş maç kimlikleri (sayısal sırada)."""
    return sorted((p.name.split(".")[0] for p in (kok / "Tracking Data").glob("*.jsonl.bz2")), key=int)


def _top(b) -> tuple:
    if isinstance(b, list):
        b = b[0] if b else None
    if not b or b.get("x") is None or b.get("y") is None:
        return (np.nan, np.nan, np.nan), False
    z = b.get("z")
    return (b["x"], b["y"], np.nan if z is None else z), b.get("visibility") == "VISIBLE"


def yukle(mac_id: str, kok: Path = KOK) -> HamMac:
    meta = json.loads((kok / "Metadata" / f"{mac_id}.json").read_text(encoding="utf-8"))[0]
    kadro = json.loads((kok / "Rosters" / f"{mac_id}.json").read_text(encoding="utf-8"))
    takim_no = {str(meta["homeTeam"]["id"]): 0, str(meta["awayTeam"]["id"]): 1}
    oyuncular = [(str(k["player"]["id"]), takim_no[str(k["team"]["id"])],
                  k.get("positionGroupType") or "?") for k in kadro]
    pidx = {(takim_no[str(k["team"]["id"])], int(k["shirtNumber"])): i for i, k in enumerate(kadro)}
    P = len(oyuncular)
    saha = meta["stadium"]["pitches"][0]
    lx, ly = float(saha["length"]), float(saha["width"])

    t, period, oyunda_l, sahip_l, top_l, top_g = [], [], [], [], [], []
    xy_l, gor_l = [], []
    bilinmeyen = set()
    oyunda, sahip, onceki_fn = False, -1, None
    with bz2.open(kok / "Tracking Data" / f"{mac_id}.jsonl.bz2", "rt", encoding="utf-8") as fh:
        for satir in fh:
            f = json.loads(satir)
            g = f.get("game_event")
            if g:
                tip = g.get("game_event_type")
                if tip in OLU:
                    oyunda = False
                elif tip in CANLI:
                    oyunda = True
                if g.get("home_ball") is not None:
                    sahip = 0 if g["home_ball"] else 1
                if tip == "END":
                    sahip = -1
            if f.get("period") is None or f["frameNum"] == onceki_fn:
                continue
            onceki_fn = f["frameNum"]

            xy = np.full((P, 2), np.nan, dtype=np.float32)
            gor = np.zeros(P, dtype=bool)
            for alan, takim in (("homePlayersSmoothed", 0), ("awayPlayersSmoothed", 1)):
                for p in f.get(alan) or []:
                    i = pidx.get((takim, int(p["jerseyNum"])))
                    if i is None:
                        bilinmeyen.add((takim, int(p["jerseyNum"])))
                        continue
                    if p.get("x") is not None and p.get("y") is not None:
                        xy[i] = (p["x"], p["y"])
                        gor[i] = p.get("visibility") == "VISIBLE"
            b, bg = _top(f.get("ballsSmoothed"))
            t.append(f["periodElapsedTime"])
            period.append(f["period"])
            oyunda_l.append(oyunda)
            sahip_l.append(sahip)
            top_l.append(b)
            top_g.append(bg)
            xy_l.append(xy)
            gor_l.append(gor)

    if bilinmeyen:
        warnings.warn(f"{mac_id}: kadroda olmayan (takım, forma) atlandı: {sorted(bilinmeyen)}")

    # PFF: orijin saha merkezi → sol alt köşe; gerekirse 105×68'e ölçekle.
    olcek = np.array([SAHA_X / lx, SAHA_Y / ly], dtype=np.float32)
    kayma = np.array([lx / 2, ly / 2], dtype=np.float32)
    xy = (np.stack(xy_l) + kayma) * olcek
    top = np.array(top_l, dtype=np.float32)
    top[:, :2] = (top[:, :2] + kayma) * olcek

    # Hiç görünmeyen oyuncuları (yedekte kalanlar) at.
    secim = np.flatnonzero(~np.all(np.isnan(xy[..., 0]), axis=0))
    return HamMac(
        mac_id=str(mac_id), fps=float(meta["fps"]), t=np.array(t, dtype=np.float64),
        period=np.array(period, dtype=np.int8), xy=xy[:, secim], top=top,
        oyunda=np.array(oyunda_l, dtype=bool), sahip=np.array(sahip_l, dtype=np.int8),
        oyuncu_id=[oyuncular[i][0] for i in secim],
        oyuncu_takim=np.array([oyuncular[i][1] for i in secim], dtype=np.int8),
        oyuncu_rol=[oyuncular[i][2] for i in secim],
        gorunur=np.stack(gor_l)[:, secim], top_gorunur=np.array(top_g, dtype=bool),
    )
