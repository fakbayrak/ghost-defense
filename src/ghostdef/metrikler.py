"""Yörünge metrikleri (metre). Maskeli: eksik yuva hiçbir ortalamaya girmez.

Şekiller: tahmin [B,K,T,N,2] (deterministik model için K=1; [B,T,N,2] de kabul edilir),
gercek [B,T,N,2], mask [B,T,N] bool.

- ADE / FDE: tüm geçerli (oyuncu, kare) üzerinden ortalama hata; K>1 ise örnekler üzerinden de
  ortalama (beklenen hata). FDE yalnız son kare.
- minADE_K / minFDE_K: her oyuncu için en iyi örnek ayrı seçilir (oyuncular farklı örneklerden
  gelebilir → sahne tutarlılığını ölçmez).
- minSADE_K / minSFDE_K: her pencere için takımın tamamına en iyi TEK örnek (sahne düzeyi).
"""
import torch


def _hazirla(tahmin, gercek, mask):
    if tahmin.dim() == 4:
        tahmin = tahmin[:, None]
    hata = torch.linalg.norm(tahmin - gercek[:, None], dim=-1)  # [B,K,T,N]
    w = mask[:, None].to(hata.dtype).expand_as(hata)
    return torch.where(w > 0, hata, torch.zeros_like(hata)), w


def _bol(pay, payda):
    return pay / payda.clamp_min(1)


def ade(tahmin, gercek, mask):
    h, w = _hazirla(tahmin, gercek, mask)
    return _bol(h.sum(), w.sum())


def fde(tahmin, gercek, mask):
    return ade(_son(tahmin), gercek[:, -1:], mask[:, -1:])


def min_ade(tahmin, gercek, mask):
    h, w = _hazirla(tahmin, gercek, mask)
    oyuncu = _bol(h.sum(2), w.sum(2))            # [B,K,N]
    gecerli = w[:, 0].sum(1) > 0                 # [B,N]
    return oyuncu.min(1).values[gecerli].mean()


def min_fde(tahmin, gercek, mask):
    return min_ade(_son(tahmin), gercek[:, -1:], mask[:, -1:])


def min_sade(tahmin, gercek, mask):
    h, w = _hazirla(tahmin, gercek, mask)
    sahne = _bol(h.sum((2, 3)), w.sum((2, 3)))   # [B,K]
    gecerli = w[:, 0].sum((1, 2)) > 0            # [B]
    return sahne.min(1).values[gecerli].mean()


def min_sfde(tahmin, gercek, mask):
    return min_sade(_son(tahmin), gercek[:, -1:], mask[:, -1:])


def _son(tahmin):
    return tahmin[..., -1:, :, :]


def hepsi(tahmin, gercek, mask) -> dict:
    """Tüm metrikler, Python float olarak."""
    fonks = dict(ADE=ade, FDE=fde, minADE=min_ade, minFDE=min_fde, minSADE=min_sade, minSFDE=min_sfde)
    return {ad: float(f(tahmin, gercek, mask)) for ad, f in fonks.items()}
