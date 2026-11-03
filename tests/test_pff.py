"""PFF okuyucusu için sentetik .jsonl.bz2 ile birim testler (ham veri gerektirmez)."""
import bz2
import json

import numpy as np
import pytest

from ghostdef.data import pff, roller

EV, DEP = "10", "20"


def oyuncu(no, x, y, gorunur=True):
    return {"jerseyNum": str(no), "confidence": "LOW", "x": x, "y": y,
            "visibility": "VISIBLE" if gorunur else "ESTIMATED"}


def kare(fn, olay=None, top=(0.0, 0.0, 0.5), ev=None, dep=None, period=1, fps=10.0):
    return {
        "frameNum": fn, "period": period, "periodElapsedTime": fn / fps,
        "homePlayersSmoothed": ev if ev is not None else [oyuncu(1, -50.0, 0.0), oyuncu(9, 10.0, 5.0, False)],
        "awayPlayersSmoothed": dep if dep is not None else [oyuncu(1, 50.0, 0.0)],
        "homePlayers": [], "awayPlayers": [], "balls": [],
        "ballsSmoothed": None if top is None else {"visibility": "VISIBLE", "x": top[0], "y": top[1], "z": top[2]},
        "game_event": olay,
    }


def olay(tip, home_ball=None):
    return {"game_event_type": tip, "home_ball": home_ball}


@pytest.fixture
def kok(tmp_path):
    for d in ("Metadata", "Rosters", "Tracking Data"):
        (tmp_path / d).mkdir()
    meta = [{"id": "1", "fps": 10.0, "homeTeam": {"id": EV}, "awayTeam": {"id": DEP},
             "stadium": {"pitches": [{"length": 105.0, "width": 68.0}]}}]
    kadro = [
        {"player": {"id": "e1"}, "positionGroupType": "GK", "shirtNumber": "1", "team": {"id": EV}},
        {"player": {"id": "e9"}, "positionGroupType": "CF", "shirtNumber": "9", "team": {"id": EV}},
        {"player": {"id": "e5"}, "positionGroupType": "CM", "shirtNumber": "5", "team": {"id": EV}},  # yedek
        {"player": {"id": "d1"}, "positionGroupType": "GK", "shirtNumber": "1", "team": {"id": DEP}},
    ]
    kareler = [
        kare(0, olay("FIRSTKICKOFF", True)),
        kare(1),
        kare(1),                                  # yinelenen kare → atlanır
        kare(2, olay("OUT")),                     # ölü top
        kare(3, top=None),                        # kamera dışı: top yok
        kare(4, olay("OTB", False), dep=[oyuncu(1, 50.0, 0.0), oyuncu(77, 0.0, 0.0)]),  # 77 kadroda yok
    ]
    (tmp_path / "Metadata" / "1.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "Rosters" / "1.json").write_text(json.dumps(kadro), encoding="utf-8")
    with bz2.open(tmp_path / "Tracking Data" / "1.jsonl.bz2", "wt", encoding="utf-8") as fh:
        for k in kareler:
            fh.write(json.dumps(k) + "\n")
    return tmp_path


def test_pff_yukle(kok):
    with pytest.warns(UserWarning, match="kadroda olmayan"):
        ham = pff.yukle("1", kok)
    assert pff.maclar(kok) == ["1"]
    np.testing.assert_allclose(ham.t, [0.0, 0.1, 0.2, 0.3, 0.4])           # yineleme atlandı
    assert ham.oyuncu_id == ["e1", "e9", "d1"]                            # yedek (e5) atıldı
    assert ham.oyuncu_rol == ["GK", "CF", "GK"]
    np.testing.assert_allclose(ham.xy[0, 0], [2.5, 34.0])                 # merkez → sol alt
    np.testing.assert_allclose(ham.xy[0, 2], [102.5, 34.0])
    np.testing.assert_allclose(ham.top[0], [52.5, 34.0, 0.5])
    assert np.isnan(ham.top[3]).all()
    assert ham.gorunur[:, 0].all() and not ham.gorunur[:, 1].any()       # 9 numara ESTIMATED
    assert ham.top_gorunur.tolist() == [True, True, True, False, True]
    assert ham.oyunda.tolist() == [True, True, False, False, True]
    assert ham.sahip.tolist() == [0, 0, 0, 0, 1]


def test_pff_rol_kodlari_idsse_indislerini_korur():
    assert roller.ROL_KODLARI[:len(roller.DFL_GRUP) + 1] == ["?"] + sorted(roller.DFL_GRUP)
    assert roller.grup("GK") == roller.KALECI and roller.grup("RWB") == roller.DEFANS
    assert roller.grup("AM") == roller.ORTA and roller.grup("CF") == roller.FORVET
    assert len(set(roller.ROL_KODLARI)) == len(roller.ROL_KODLARI)
