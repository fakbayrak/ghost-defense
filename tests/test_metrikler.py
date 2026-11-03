"""Metriklerin elle hesaplanmış küçük örneklerle testi."""
import pytest
import torch

from ghostdef import metrikler as M


def test_ade_fde_sabit_kayma():
    gercek = torch.zeros(2, 4, 3, 2)
    tahmin = gercek + torch.tensor([3.0, 4.0])          # her yerde 5 m hata
    mask = torch.ones(2, 4, 3, dtype=torch.bool)
    assert M.ade(tahmin, gercek, mask).item() == pytest.approx(5.0)
    assert M.fde(tahmin, gercek, mask).item() == pytest.approx(5.0)


def test_fde_yalniz_son_kare():
    gercek = torch.zeros(1, 3, 1, 2)
    tahmin = gercek.clone()
    tahmin[:, -1, :, 0] = 2.0
    mask = torch.ones(1, 3, 1, dtype=torch.bool)
    assert M.fde(tahmin, gercek, mask).item() == pytest.approx(2.0)
    assert M.ade(tahmin, gercek, mask).item() == pytest.approx(2.0 / 3)


def test_maskeli_yuva_etkilemez():
    gercek = torch.zeros(1, 2, 2, 2)
    tahmin = gercek.clone()
    tahmin[:, :, 1] = 1000.0                             # eksik yuva: dev hata
    mask = torch.tensor([[[True, False], [True, False]]])
    for ad, deger in M.hepsi(tahmin, gercek, mask).items():
        assert deger == pytest.approx(0.0), ad


def test_min_ade_ve_min_sade_farki():
    """K=2, N=2: her örnek bir oyuncuyu tam, diğerini 2 m hatalı tutturur.
    minADE oyuncu başına en iyiyi seçer → 0; minSADE tek örnek seçmek zorunda → 1."""
    gercek = torch.zeros(1, 1, 2, 2)
    tahmin = torch.zeros(1, 2, 1, 2, 2)
    tahmin[0, 0, 0, 1, 0] = 2.0   # örnek 0: oyuncu 1 hatalı
    tahmin[0, 1, 0, 0, 0] = 2.0   # örnek 1: oyuncu 0 hatalı
    mask = torch.ones(1, 1, 2, dtype=torch.bool)
    assert M.min_ade(tahmin, gercek, mask).item() == pytest.approx(0.0)
    assert M.min_sade(tahmin, gercek, mask).item() == pytest.approx(1.0)
    assert M.ade(tahmin, gercek, mask).item() == pytest.approx(1.0)
    assert M.min_sfde(tahmin, gercek, mask).item() == pytest.approx(1.0)


def test_k1_icin_min_metrikler_ade_ile_ayni():
    torch.manual_seed(0)
    gercek = torch.randn(4, 5, 11, 2)
    tahmin = gercek + torch.randn(4, 5, 11, 2)
    mask = torch.ones(4, 5, 11, dtype=torch.bool)
    s = M.hepsi(tahmin, gercek, mask)
    assert s["minADE"] == pytest.approx(s["ADE"], rel=1e-5)
    assert s["minSADE"] == pytest.approx(s["ADE"], rel=1e-5)
    assert s["minFDE"] == pytest.approx(s["FDE"], rel=1e-5)
