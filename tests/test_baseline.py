"""Basit baseline'ların testleri (sentetik veri)."""
import numpy as np
import torch

from ghostdef.baseline.basit import RolOrtalamasi, SabitHiz, TopaGoreRolOrtalamasi
from ghostdef.data.ortak_format import TOP


class SahteVeriSeti:
    """fit() yalnız pos ve mask kullanır."""
    def __init__(self, pos, mask):
        self.pos, self.mask = pos.astype(np.float32), mask


def test_sabit_hiz_duz_cizgide_sifir_hata():
    B, H, L = 3, 20, 50
    t = torch.arange(L, dtype=torch.float32)
    hiz = torch.tensor([0.3, -0.1])
    hedef = (torch.tensor([10.0, 30.0]) + t[:, None] * hiz)[None, :, None].expand(B, L, 11, 2)
    yigin = {"savunma_gecmis": hedef[:, :H], "top": torch.zeros(B, L, 3)}
    tahmin = SabitHiz().tahmin(yigin)
    assert tahmin.shape == (B, 1, L, 11, 2)
    assert torch.allclose(tahmin[:, 0], hedef, atol=1e-4)


def test_rol_ortalamasi_maskeyi_dikkate_alir():
    pos = np.zeros((4, 23, 2))
    pos[:, 11:22, 0] = [[1], [3], [5], [100]]          # son kare maskeli
    mask = np.ones((4, 23), bool)
    mask[3] = False
    m = RolOrtalamasi().fit(SahteVeriSeti(pos, mask))
    tahmin = m.tahmin({"top": torch.zeros(2, 50, 3)})
    assert tahmin.shape == (2, 1, 50, 11, 2)
    assert torch.allclose(tahmin[..., 0], torch.tensor(3.0))


def test_topa_gore_hucre_ortalamasi_ve_geri_dusme():
    rng = np.random.default_rng(0)
    N = 400
    pos = np.zeros((N, 23, 2))
    pos[:, TOP] = np.c_[rng.uniform(0, 52, N), rng.uniform(0, 68, N)]   # top hep sol yarıda
    pos[:, 11:22] = pos[:, TOP][:, None] + 5.0                          # savunma = top + 5 m
    model = TopaGoreRolOrtalamasi(nx=2, ny=1, min_ornek=50).fit(SahteVeriSeti(pos, np.ones((N, 23), bool)))
    genel = RolOrtalamasi().fit(SahteVeriSeti(pos, np.ones((N, 23), bool))).ort
    top = torch.tensor([[[20.0, 30.0, 0.0], [90.0, 30.0, 0.0]]])        # sol hücre, boş sağ hücre
    tahmin = model.tahmin({"top": top})[0, 0]
    sol = pos[:, TOP].mean(0) + 5.0
    assert torch.allclose(tahmin[0], torch.tensor(sol, dtype=torch.float32).expand(11, 2), atol=1e-4)
    assert torch.allclose(tahmin[1], genel, atol=1e-5)                  # sağ hücrede veri yok
