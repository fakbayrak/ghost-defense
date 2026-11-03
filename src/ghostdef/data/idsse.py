"""IDSSE (Bassek vd., 2025; DFL/Sportec formatı, 25 Hz) → HamMac.

Ayrıştırma kloppy ile; pozisyon kodları doğrudan meta XML'den (PlayingPosition) okunur,
çünkü kloppy 3.19'un PositionType hiyerarşisinde hata var (WingBack anahtarı yok).
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from kloppy import sportec
from kloppy.domain import BallState

from .ortak_format import SAHA_X, SAHA_Y, HamMac

KOK = Path(__file__).resolve().parents[3] / "data" / "raw" / "idsse"
MACLAR = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]


def dosyalar(mac_id: str, kok: Path = KOK) -> dict:
    bul = lambda desen: next(kok.glob(f"{desen}*DFL-MAT-{mac_id}.xml"))
    return {"meta": bul("DFL_02_01"), "tracking": bul("DFL_04_03"), "event": bul("DFL_03_02")}


def meta_oku(meta_yol: Path) -> dict:
    """PersonId → PlayingPosition kodu ('?' = sahaya çıkmamış / kod yok) ve saha ölçüsü."""
    kok = ET.parse(meta_yol).getroot()
    env = kok.find(".//Environment")
    roller = {p.get("PersonId"): p.get("PlayingPosition", "?") for p in kok.iter("Player")}
    return {"rol": roller, "saha": (float(env.get("PitchX")), float(env.get("PitchY")))}


def yukle(mac_id: str, kok: Path = KOK) -> HamMac:
    d = dosyalar(mac_id, kok)
    meta = meta_oku(d["meta"])
    ds = sportec.load_tracking(raw_data=d["tracking"], meta_data=d["meta"], coordinates="sportec")
    ev, dep = ds.metadata.teams

    oyuncular = [(p, 0) for p in ev.players] + [(p, 1) for p in dep.players]
    pidx = {p.player_id: i for i, (p, _) in enumerate(oyuncular)}
    T, P = len(ds.frames), len(oyuncular)

    xy = np.full((T, P, 2), np.nan, dtype=np.float32)
    top = np.full((T, 3), np.nan, dtype=np.float32)
    t = np.empty(T)
    period = np.empty(T, dtype=np.int8)
    oyunda = np.zeros(T, dtype=bool)
    sahip = np.full(T, -1, dtype=np.int8)

    for i, f in enumerate(ds.frames):
        t[i] = f.timestamp.total_seconds()
        period[i] = f.period.id
        oyunda[i] = f.ball_state == BallState.ALIVE
        if f.ball_owning_team is not None:
            sahip[i] = 0 if f.ball_owning_team is ev else 1
        b = f.ball_coordinates
        if b is not None:
            top[i] = (b.x, b.y, b.z if b.z is not None else np.nan)
        for p, pd in f.players_data.items():
            if pd.coordinates is not None:
                xy[i, pidx[p.player_id]] = (pd.coordinates.x, pd.coordinates.y)

    # Sportec: orijin saha merkezi → sol alt köşe; gerekirse 105×68'e ölçekle.
    lx, ly = meta["saha"]
    olcek = np.array([SAHA_X / lx, SAHA_Y / ly], dtype=np.float32)
    xy = (xy + np.array([lx / 2, ly / 2], dtype=np.float32)) * olcek
    top[:, :2] = (top[:, :2] + np.array([lx / 2, ly / 2], dtype=np.float32)) * olcek

    # Hiç görünmeyen oyuncuları (yedekte kalanlar) at.
    gorunen = ~np.all(np.isnan(xy[..., 0]), axis=0)
    secim = np.flatnonzero(gorunen)
    return HamMac(
        mac_id=mac_id, fps=float(ds.metadata.frame_rate), t=t, period=period,
        xy=xy[:, secim], top=top, oyunda=oyunda, sahip=sahip,
        oyuncu_id=[oyuncular[i][0].player_id for i in secim],
        oyuncu_takim=np.array([oyuncular[i][1] for i in secim], dtype=np.int8),
        oyuncu_rol=[meta["rol"].get(oyuncular[i][0].player_id, "?") for i in secim],
    )
