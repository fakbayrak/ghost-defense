"""Ortak format dönüşümleri için sentetik veriyle birim testler (ham veri gerektirmez)."""
import numpy as np
import pytest

from ghostdef.data import roller
from ghostdef.data.ortak_format import (
    SAHA_X, SAHA_Y, TOP, HamMac, dondur_180, ortak_formata, segmentler, yeniden_ornekle,
)

ROLLER_11 = ["TW", "LV", "IVL", "IVR", "RV", "DML", "DMR", "OLM", "ZO", "ORM", "STZ"]


def sentetik_mac(T=250, fps=25.0, sahip=1, ev_saga=True) -> HamMac:
    """22 oyuncu, tek periyot. Takımlar kendi yarı sahalarında, kaleciler kale önünde."""
    t = np.arange(T) / fps
    rng = np.random.default_rng(0)
    xy = np.zeros((T, 22, 2), dtype=np.float32)
    for takim in (0, 1):
        sol = (takim == 0) == ev_saga  # bu takımın kalesi solda mı
        for j, rol in enumerate(ROLLER_11):
            x = 5.0 if rol == "TW" else 20 + 2 * j
            y = 5 + 6 * j
            xy[:, takim * 11 + j] = (x if sol else SAHA_X - x, y)
    xy += rng.normal(0, 0.01, xy.shape).astype(np.float32)
    top = np.zeros((T, 3), dtype=np.float32)
    top[:, 0], top[:, 1] = 52.5 + t, 34.0
    return HamMac(
        mac_id="test", fps=fps, t=t, period=np.ones(T, dtype=np.int8), xy=xy, top=top,
        oyunda=np.ones(T, dtype=bool), sahip=np.full(T, sahip, dtype=np.int8),
        oyuncu_id=[f"p{i}" for i in range(22)],
        oyuncu_takim=np.repeat([0, 1], 11).astype(np.int8), oyuncu_rol=ROLLER_11 * 2,
    )


def test_yeniden_ornekleme_dogrusal_hareketi_korur():
    ham = sentetik_mac()
    on = yeniden_ornekle(ham, hz=10)
    assert on.fps == 10
    np.testing.assert_allclose(np.diff(on.t), 0.1, atol=1e-6)
    np.testing.assert_allclose(on.top[:, 0], 52.5 + on.t, atol=1e-4)


def test_yeniden_ornekleme_nan_yayar_ve_uydurmaz():
    ham = sentetik_mac()
    ham.xy[50:60, 3] = np.nan  # 2,0–2,36 s arası oyuncu 3 yok
    on = yeniden_ornekle(ham, hz=10)
    eksik = np.isnan(on.xy[:, 3, 0])
    assert eksik[(on.t >= 2.0) & (on.t <= 2.36)].all()
    assert not eksik[(on.t < 1.9) | (on.t > 2.5)].any()


def test_segmentler_olu_top_ve_sahiplik_degisimi():
    ham = yeniden_ornekle(sentetik_mac(T=1000), hz=10)  # 40 s
    ham.oyunda[100:110] = False      # 10–11 s ölü top
    ham.sahip[200:] = 0              # 20. s'de sahiplik değişir
    ham.sahip[380:390] = 1           # 1 s'lik kısa parça → atılır
    seg = segmentler(ham, min_sure=2.0)
    assert (seg[100:110] == -1).all() and (seg[380:390] == -1).all()
    # [0,10) [11,20) [20,38) kalır; [39,39.9] 2 s'den kısa → atılır
    assert len(np.unique(seg[seg >= 0])) == 3
    assert (seg[390:] == -1).all()
    assert seg[0] != seg[150] != seg[250]


def test_dondur_180_tersinir_ve_aynalama_degil():
    p = np.array([[10.0, 50.0]])
    q = dondur_180(p)
    np.testing.assert_allclose(q, [[SAHA_X - 10, SAHA_Y - 50]])
    np.testing.assert_allclose(dondur_180(q), p)


@pytest.mark.parametrize("ev_saga", [True, False])
@pytest.mark.parametrize("sahip", [0, 1])
def test_hucum_hep_soldan_saga_ve_kaleci_ilk_yuvada(sahip, ev_saga):
    ham = yeniden_ornekle(sentetik_mac(sahip=sahip, ev_saga=ev_saga), hz=10)
    of = ortak_formata(ham, segmentler(ham))
    pos, rol = of["pos"], of["rol"]
    tw = roller.kod_id("TW")
    assert (rol[:, 0] == tw).all() and (rol[:, 11] == tw).all()
    assert pos[:, 0, 0].mean() < 10            # hücum kalecisi solda
    assert pos[:, 11, 0].mean() > SAHA_X - 10  # savunma kalecisi sağda
    assert of["mask"].all()
    assert of["seg_hucum_takim"][0] == sahip


def test_gorunur_yoksa_maskeye_esit():
    on = yeniden_ornekle(sentetik_mac(), hz=10)
    of = ortak_formata(on, segmentler(on))
    assert (of["gorunur"] == of["mask"]).all()


def test_gorunur_temkinli_yeniden_orneklenir_ve_yuvaya_tasinir():
    ham = sentetik_mac()  # 25 fps, top deplasmanda → savunma = ev sahibi
    ham.gorunur = np.ones((len(ham.t), 22), dtype=bool)
    ham.gorunur[50:60, 3] = False  # ev sahibi oyuncu 3 (IVL), 2,0–2,36 s tahmin
    ham.top_gorunur = np.ones(len(ham.t), dtype=bool)
    ham.top_gorunur[100] = False   # 4,0 s
    on = yeniden_ornekle(ham, hz=10)
    g = on.gorunur[:, 3]
    assert not g[(on.t >= 2.0) & (on.t <= 2.36)].any()
    assert g[(on.t < 1.9) | (on.t > 2.5)].all()
    assert on.top_gorunur.sum() == len(on.t) - 1
    of = ortak_formata(on, segmentler(on))
    yuva = int(np.flatnonzero(of["seg_oyuncu"][0] == 3)[0])
    assert yuva >= 11                      # savunma tarafında
    assert not of["gorunur"][20:24, yuva].any() and of["gorunur"][:19, yuva].all()
    assert of["gorunur"][:, :TOP].sum() == of["mask"][:, :TOP].sum() - 4


def test_eksik_oyuncu_maskelenir():
    ham = sentetik_mac()
    ham.xy[:, 21] = np.nan  # deplasmanın forveti hiç yok (ör. kırmızı kart); top deplasmanda
    on = yeniden_ornekle(ham, hz=10)
    of = ortak_formata(on, segmentler(on))
    assert of["mask"][:, :TOP].sum(1).max() == 21
    assert not of["mask"][:, 10].any()  # hücumun son yuvası boş
    assert of["mask"][:, 11:TOP].all()
    assert (of["pos"][~of["mask"]] == 0).all()
