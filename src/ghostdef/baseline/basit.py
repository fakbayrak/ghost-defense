"""Basit baseline'lar. Hepsi tüm pencere için [B,1,L,11,2] (metre) savunma tahmini döndürür.

`fit(veri_seti)` eğitim maçlarının tüm segment karelerini (kısa segmentler dahil) kullanır.
Rol ortalaması ve topa göre rol ortalaması savunmayı hiç görmez → kurulum A ve B'de aynıdır.
Sabit hız `savunma_gecmis` ister → yalnız kurulum B.
"""
import numpy as np
import torch

from ..data.ortak_format import SAHA_X, SAHA_Y, TOP
from ..data.pencere import SAV


class RolOrtalamasi:
    """Her savunma yuvasının eğitimdeki ortalama konumu; zamandan ve topdan bağımsız."""

    def fit(self, ds):
        pos, m = ds.pos[:, SAV], ds.mask[:, SAV, None]
        self.ort = torch.from_numpy((pos * m).sum(0) / np.maximum(m.sum(0), 1)).float()  # [11,2]
        return self

    def tahmin(self, yigin):
        B, L = yigin["top"].shape[:2]
        return self.ort.expand(B, 1, L, *self.ort.shape).clone()


class TopaGoreRolOrtalamasi:
    """Top konumu nx×ny ızgarada; her hücre ve yuva için ortalama konum.

    Hücrede bir yuva için `min_ornek`ten az kare varsa o yuvada genel rol ortalaması kullanılır.
    """

    def __init__(self, nx: int = 7, ny: int = 5, min_ornek: int = 50):
        self.nx, self.ny, self.min_ornek = nx, ny, min_ornek

    def _hucre(self, top_xy):
        ix = np.clip((top_xy[..., 0] / SAHA_X * self.nx).astype(int), 0, self.nx - 1)
        iy = np.clip((top_xy[..., 1] / SAHA_Y * self.ny).astype(int), 0, self.ny - 1)
        return ix * self.ny + iy

    def fit(self, ds):
        genel = RolOrtalamasi().fit(ds).ort.numpy()
        hucre = self._hucre(ds.pos[:, TOP])
        pos, m = ds.pos[:, SAV], ds.mask[:, SAV].astype(np.float32)
        H = self.nx * self.ny
        toplam = np.zeros((H, 11, 2), dtype=np.float64)
        sayi = np.zeros((H, 11), dtype=np.float64)
        np.add.at(toplam, hucre, pos * m[..., None])
        np.add.at(sayi, hucre, m)
        ort = toplam / np.maximum(sayi, 1)[..., None]
        az = sayi < self.min_ornek
        ort[az] = np.broadcast_to(genel, ort.shape)[az]
        self.ort = torch.from_numpy(ort).float()   # [H,11,2]
        self.sayi = sayi
        return self

    def tahmin(self, yigin):
        hucre = torch.from_numpy(self._hucre(yigin["top"][..., :2].numpy()))  # [B,L]
        return self.ort[hucre][:, None]


class SabitHiz:
    """Son `n_hiz` kare aralığının ortalama hızıyla doğrusal uzatma (kurulum B)."""

    def __init__(self, n_hiz: int = 2):
        self.n_hiz = n_hiz

    def fit(self, ds):
        return self

    def tahmin(self, yigin):
        g = yigin["savunma_gecmis"]                  # [B,H,11,2]
        H, L = g.shape[1], yigin["top"].shape[1]
        v = (g[:, -1] - g[:, -1 - self.n_hiz]) / self.n_hiz
        k = torch.arange(1, L - H + 1, dtype=g.dtype, device=g.device)[None, :, None, None]
        gelecek = g[:, -1:] + v[:, None] * k
        return torch.cat([g, gelecek], dim=1)[:, None]
