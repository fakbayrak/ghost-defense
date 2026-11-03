"""Rol başına, etkileşimsiz LSTM baseline'ı.

Her savunma yuvası ayrı bir dizidir; tüm yuvalar ortak ağırlıklı tek bir çift yönlü LSTM'den geçer.
Savunmacılar birbirini görmez; bağlam = hücum + top (tüm pencere). Kurulum B'de yuva ayrıca kendi ilk
H karesini görür ve çıktı sabit hız tahminine eklenen düzeltmedir (`taban="sabit_hiz"`; ilk sürümde
son konuma göre kaymaydı, `taban="konum"` — kısa ufukta sabit hızın gerisinde kaldığı için değiştirildi).
Çıktı şekli baseline'larla aynı: [B,1,L,11,2] metre.
"""
import torch
from torch import nn

from ..baseline.basit import SabitHiz
from ..data.ortak_format import N_TAKIM
from ..data.pencere import denormalize, normalize

TOP_Z_OLCEK = 10.0  # metre → kabaca [0,1]-[0,3]


def baglam_ozellikleri(yigin) -> torch.Tensor:
    """Kare başına, tüm savunmacılar için ortak bağlam: top (3) + hücum 11×2 + hücum maskesi 11 → [B,L,36]."""
    top = yigin["top"]
    top_n = torch.cat([normalize(top[..., :2]), top[..., 2:] / TOP_Z_OLCEK], dim=-1)
    hucum = normalize(yigin["hucum"]).flatten(2)                       # [B,L,22]
    hucum_m = yigin["mask"][..., :N_TAKIM].to(top.dtype)               # [B,L,11]
    return torch.cat([top_n, hucum * hucum_m.repeat_interleave(2, -1), hucum_m], dim=-1)


def gecmis_ozellikleri(yigin, L: int) -> torch.Tensor:
    """Kurulum B: yuvanın kendi konumu (ilk H kare, sonrası 0) + gözlendi biti → [B,L,11,3]."""
    g = yigin["savunma_gecmis"]
    B, H = g.shape[:2]
    gozlendi = yigin["mask"][:, :H, N_TAKIM:2 * N_TAKIM].to(g.dtype)  # [B,H,11]
    oz = torch.zeros(B, L, N_TAKIM, 3, dtype=g.dtype, device=g.device)
    oz[:, :H, :, :2] = normalize(g) * gozlendi[..., None]
    oz[:, :H, :, 2] = gozlendi
    return oz


class RolLSTM(nn.Module):
    BAGLAM = 36

    def __init__(self, kurulum: str = "A", gizli: int = 128, katman: int = 2, gomme: int = 16,
                 dropout: float = 0.1, taban: str = "sabit_hiz"):
        super().__init__()
        if kurulum not in ("A", "B"):
            raise ValueError(f"kurulum 'A' veya 'B' olmalı, {kurulum!r} verildi")
        if taban not in ("sabit_hiz", "konum"):
            raise ValueError(f"taban 'sabit_hiz' veya 'konum' olmalı, {taban!r} verildi")
        self.kurulum, self.taban = kurulum, taban
        self.ayar = dict(kurulum=kurulum, gizli=gizli, katman=katman, gomme=gomme, dropout=dropout,
                         taban=taban)
        self.yuva = nn.Embedding(N_TAKIM, gomme)
        girdi = self.BAGLAM + gomme + (3 if kurulum == "B" else 0)
        self.lstm = nn.LSTM(girdi, gizli, num_layers=katman, batch_first=True,
                            bidirectional=True, dropout=dropout if katman > 1 else 0.0)
        self.cikis = nn.Linear(2 * gizli, 2)
        nn.init.zeros_(self.cikis.weight)
        nn.init.constant_(self.cikis.bias, 0.5 if kurulum == "A" else 0.0)  # A: saha ortası

    def forward(self, yigin) -> torch.Tensor:
        baglam = baglam_ozellikleri(yigin)                                # [B,L,36]
        B, L = baglam.shape[:2]
        parcalar = [baglam[:, :, None].expand(B, L, N_TAKIM, self.BAGLAM),
                    self.yuva.weight[None, None].expand(B, L, N_TAKIM, -1)]
        if self.kurulum == "B":
            parcalar.append(gecmis_ozellikleri(yigin, L))
        x = torch.cat(parcalar, dim=-1)                                   # [B,L,11,D]
        x = x.permute(0, 2, 1, 3).reshape(B * N_TAKIM, L, -1)
        h, _ = self.lstm(x)
        y = self.cikis(h).reshape(B, N_TAKIM, L, 2).permute(0, 2, 1, 3)  # [B,L,11,2] normalize
        if self.kurulum == "A":
            return denormalize(y)[:, None]
        g = yigin["savunma_gecmis"]
        H = g.shape[1]
        if self.taban == "sabit_hiz":   # model sabit hız tahmininden sapmayı öğrenir
            taban = SabitHiz().tahmin(yigin)[:, 0, H:]
        else:                           # ilk sürüm (2026-11-03): son konumdan kayma
            taban = g[:, -1:]
        gelecek = taban + denormalize(y[:, H:])
        return torch.cat([g, gelecek], dim=1)[:, None]
