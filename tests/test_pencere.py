"""Pencere, Dataset ve bölme testleri (sentetik npz; ham veri gerektirmez)."""
import numpy as np
import pytest
import torch

from ghostdef.data.bolme import mac_disari_katlar
from ghostdef.data.pencere import PencereVeriSeti, denormalize, normalize, pencere_indeksleri

SEG_UZUNLUK = [60, 40, 55]  # kare; 40'lık segment 50'lik pencereye sığmaz


def sentetik_npz(klasor, ad, seg_uzunluk=SEG_UZUNLUK, eksik_yuva=None):
    seg = np.concatenate([np.full(n, i) for i, n in enumerate(seg_uzunluk)]).astype(np.int32)
    N = len(seg)
    pos = np.zeros((N, 23, 2), dtype=np.float32)
    pos[..., 0] = np.arange(N)[:, None] * 0.1          # kare indisi geri okunabilsin
    pos[..., 1] = np.arange(23)[None, :]               # yuva indisi
    mask = np.ones((N, 23), dtype=bool)
    if eksik_yuva is not None:
        mask[:, eksik_yuva] = False
        pos[:, eksik_yuva] = 0
    np.savez(klasor / f"{ad}.npz", pos=pos, mask=mask, top_z=np.full(N, 0.5, np.float32),
             rol=np.ones((N, 22), np.int8), segment_id=seg)


def test_pencere_segment_siniri_gecmez():
    seg = np.concatenate([np.full(n, i) for i, n in enumerate(SEG_UZUNLUK)])
    b = pencere_indeksleri(seg, 50, 5)
    np.testing.assert_array_equal(b, [0, 5, 10, 100, 105])
    for s in b:
        assert len(np.unique(seg[s:s + 50])) == 1


def test_dataset_sekiller_ve_kurulumlar(tmp_path):
    sentetik_npz(tmp_path, "M1")
    sentetik_npz(tmp_path, "M2")
    a = PencereVeriSeti(["M1", "M2"], klasor=tmp_path, kurulum="A")
    b = PencereVeriSeti(["M1", "M2"], klasor=tmp_path, kurulum="B", top_z_kullan=False)
    assert len(a) == len(b) == 10
    o = a[0]
    assert o["hucum"].shape == (50, 11, 2) and o["hedef"].shape == (50, 11, 2)
    assert o["top"].shape == (50, 3) and o["mask"].shape == (50, 23) and o["rol"].shape == (22,)
    assert "savunma_gecmis" not in o
    assert torch.allclose(o["top"][:, 2], torch.tensor(0.5))
    # yuva düzeni: hucum 0–10, hedef 11–21, top 22
    assert o["hucum"][0, :, 1].tolist() == list(range(11))
    assert o["hedef"][0, :, 1].tolist() == list(range(11, 22))
    assert torch.allclose(o["top"][:, 1], torch.tensor(22.0))
    ob = b[7]  # ikinci maçın pencereleri ofsetle doğru kareden başlamalı
    assert ob["mac"] == 1 and ob["savunma_gecmis"].shape == (20, 11, 2)
    assert torch.equal(ob["savunma_gecmis"], ob["hedef"][:20])
    assert (ob["top"][:, 2] == 0).all()
    assert ob["hucum"][0, 0, 0].item() == pytest.approx(0.1 * 10)   # M2'de 3. pencere, kare 10


def test_dataset_gecersiz_kurulum(tmp_path):
    sentetik_npz(tmp_path, "M1")
    with pytest.raises(ValueError):
        PencereVeriSeti(["M1"], klasor=tmp_path, kurulum="C")


def test_katlar_mac_sizintisi_yok():
    maclar = [f"M{i}" for i in range(7)]
    katlar = mac_disari_katlar(maclar)
    assert sorted(k["test"][0] for k in katlar) == sorted(maclar)
    for k in katlar:
        e, d, t = set(k["egitim"]), set(k["dogrulama"]), set(k["test"])
        assert not (e & d or e & t or d & t)
        assert e | d | t == set(maclar) and len(e) == 5


def test_normalize_tersinir():
    xy = torch.tensor([[0.0, 0.0], [105.0, 68.0], [52.5, 34.0]])
    assert torch.allclose(normalize(xy)[1], torch.tensor([1.0, 1.0]))
    assert torch.allclose(denormalize(normalize(xy)), xy)
    np.testing.assert_allclose(denormalize(normalize(xy.numpy())), xy.numpy(), rtol=1e-6)
