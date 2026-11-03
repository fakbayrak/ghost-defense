"""Rol başına LSTM ve eğitim döngüsü testleri (CPU, sentetik veri)."""
import pytest
import torch

from ghostdef.baseline.basit import SabitHiz
from ghostdef.egitim import EgitimAyari, egit, kayip, kaydet, tahmin_et, yukle
from ghostdef.modeller.lstm import RolLSTM

H, L = 20, 50


def sentetik_yigin(B=8, kurulum="A", tohum=0):
    """Savunma = top + yuvaya özgü sabit kayma (öğrenilebilir basit ilişki)."""
    g = torch.Generator().manual_seed(tohum)
    t = torch.arange(L, dtype=torch.float32)[None, :, None]
    bas = torch.rand(B, 1, 2, generator=g) * torch.tensor([60.0, 40.0]) + torch.tensor([20.0, 14.0])
    top_xy = bas + t * torch.rand(B, 1, 2, generator=g) * 0.3
    kayma = torch.linspace(-10, 10, 11)[:, None].expand(11, 2)
    hedef = top_xy[:, :, None] + kayma
    hucum = top_xy[:, :, None] + torch.randn(B, 1, 11, 2, generator=g) * 5
    y = {"hucum": hucum, "top": torch.cat([top_xy, torch.zeros(B, L, 1)], -1), "hedef": hedef,
         "mask": torch.ones(B, L, 23, dtype=torch.bool)}
    if kurulum == "B":
        y["savunma_gecmis"] = hedef[:, :H].clone()
    return y


@pytest.mark.parametrize("kurulum", ["A", "B"])
def test_cikis_sekli(kurulum):
    y = sentetik_yigin(kurulum=kurulum)
    out = RolLSTM(kurulum, gizli=16, katman=1)(y)
    assert out.shape == (8, 1, L, 11, 2)
    assert torch.isfinite(out).all()


def test_a_savunmayi_gormez():
    torch.manual_seed(0)
    m = RolLSTM("A", gizli=16, katman=1).eval()
    y = sentetik_yigin()
    y2 = dict(y, hedef=y["hedef"] + 50, mask=y["mask"].clone())
    y2["mask"][:, :, 11:22] = False      # savunma maskesi de girdi değil
    assert torch.equal(m(y), m(y2))


def test_b_gelecegi_gormez_ve_gecmisi_kullanir():
    torch.manual_seed(0)
    m = RolLSTM("B", gizli=16, katman=1).eval()
    for p in m.cikis.parameters():       # sıfır başlatmayı boz ki çıktı girdiye bağlı olsun
        torch.nn.init.normal_(p, std=0.1)
    y = sentetik_yigin(kurulum="B")
    y_gelecek = dict(y, hedef=y["hedef"].clone())
    y_gelecek["hedef"][:, H:] += 50
    assert torch.equal(m(y), m(y_gelecek))
    y_gecmis = dict(y, savunma_gecmis=y["savunma_gecmis"] + 1.0)
    assert not torch.allclose(m(y), m(y_gecmis))
    assert torch.equal(m(y)[:, 0, :H], y["savunma_gecmis"])   # geçmiş kareler aynen döner


def test_b_baslangicta_taban_tahmini():
    y = sentetik_yigin(kurulum="B")
    out = RolLSTM("B", gizli=16, katman=1)(y)               # sıfır başlatma → düzeltme 0
    assert torch.allclose(out, SabitHiz().tahmin(y), atol=1e-5)
    out_k = RolLSTM("B", gizli=16, katman=1, taban="konum")(y)
    assert torch.allclose(out_k[:, 0, H:], y["savunma_gecmis"][:, -1:].expand(-1, L - H, -1, -1))


def test_kayip_maskeli_yuvayi_yok_sayar():
    y = sentetik_yigin()
    tahmin = y["hedef"][:, None].clone()
    tahmin[:, :, :, 3] += 1000
    y["mask"][:, :, 11 + 3] = False
    assert kayip(tahmin, y).item() == pytest.approx(0.0, abs=1e-2)


def test_egitim_kaybi_duser_ve_kayit_yuklenir(tmp_path):
    torch.manual_seed(0)
    egitim, dog = sentetik_yigin(B=64, tohum=1), sentetik_yigin(B=16, tohum=2)
    m = RolLSTM("A", gizli=32, katman=1, dropout=0.0)
    ayar = EgitimAyari(lr=1e-2, batch=16, epoch=30, sabir=30, kayip_bas=0, metrik_bas=H)
    bas = kayip(tahmin_et(m, dog), dog).item()
    m, gunluk, _ = egit(m, egitim, dog, ayar)
    assert min(g["dogrulama_ADE"] for g in gunluk) < 0.5 * bas
    kaydet(tmp_path / "m.pt", m, ayar)
    m2, _ = yukle(tmp_path / "m.pt", RolLSTM)
    assert torch.allclose(tahmin_et(m, dog), tahmin_et(m2, dog))
