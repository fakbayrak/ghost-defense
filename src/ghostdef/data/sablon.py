"""Şablon eşleştirmeyle rol yuvaları (formasyondan bağımsız yuva anlamı).

Kaleci her tarafın 0. yuvasında kalır. 10 saha oyuncusu, takımın segment ortalama şekline göre
(oyuncu segment ortalaması − saha oyuncularının ağırlık merkezi) 10 rollük şablona Macar algoritmasıyla
atanır. Şablon, kısıtlı k-ortalamalarla (her segmentte her rol tek oyuncu) yalnız verilen maçlardan
öğrenilir; hücum ve savunma için ayrı şablon. Atama segment düzeyinde: pencere içinde kimlik sabit.

Literatür: Le vd. 2017 (rol hizalama), Kim vd. 2022 SoccerCPD (Macar ile rol atama).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import roller
from .ortak_format import N_TAKIM

N_SAHA = N_TAKIM - 1
TW = roller.kod_id("TW")


def segment_sekilleri(pos, mask, rol, segment_id, taraf: int):
    """Bir tarafın (0 hücum, 1 savunma) segment başına şekli.

    Dönüş: sekil [S,10,2] (merkeze göre), gecerli [S,10], saha_yuva [S,10] (eski yuva indisi, -1 boş),
    kaleci [S] (eski yuva indisi, -1 yok).
    """
    b = taraf * N_TAKIM
    p, m = pos[:, b:b + N_TAKIM], mask[:, b:b + N_TAKIM].astype(np.float64)
    S = int(segment_id.max()) + 1
    toplam = np.zeros((S, N_TAKIM, 2))
    sayi = np.zeros((S, N_TAKIM))
    np.add.at(toplam, segment_id, p * m[..., None])
    np.add.at(sayi, segment_id, m)
    ort = toplam / np.maximum(sayi, 1)[..., None]
    var = sayi > 0
    _, ilk = np.unique(segment_id, return_index=True)
    rol_seg = rol[ilk, b:b + N_TAKIM]

    sekil = np.zeros((S, N_SAHA, 2))
    gecerli = np.zeros((S, N_SAHA), dtype=bool)
    saha_yuva = np.full((S, N_SAHA), -1, dtype=np.int64)
    kaleci = np.full(S, -1, dtype=np.int64)
    for s in range(S):
        aday = np.flatnonzero(var[s])
        if len(aday) == 0:
            continue
        tw = aday[rol_seg[s, aday] == TW]
        if len(tw):
            k = tw[0]
        else:  # yedek yol: kendi kalesine en yakın (hücum −x, savunma +x tarafını savunur)
            x = ort[s, aday, 0]
            k = aday[np.argmin(x) if taraf == 0 else np.argmax(x)]
        kaleci[s] = k
        saha = aday[aday != k][:N_SAHA]
        merkez = ort[s, saha].mean(0)
        n = len(saha)
        sekil[s, :n] = ort[s, saha] - merkez
        gecerli[s, :n] = True
        saha_yuva[s, :n] = saha
    return sekil, gecerli, saha_yuva, kaleci


def sablona_ata(sekil, gecerli, sablon) -> np.ndarray:
    """[S,10] → her rol için `sekil` içindeki oyuncu sırası (-1 = rol boş). Kare uzaklık maliyeti."""
    S = len(sekil)
    atama = np.full((S, N_SAHA), -1, dtype=np.int64)
    for s in range(S):
        v = np.flatnonzero(gecerli[s])
        if len(v) == 0:
            continue
        maliyet = ((sekil[s, v][:, None] - sablon[None]) ** 2).sum(-1)   # [n,10]
        satir, sutun = linear_sum_assignment(maliyet)
        atama[s, sutun] = v[satir]
    return atama


def _rol_ortalamasi(sekil, gecerli, atama):
    al = np.take_along_axis(sekil, np.maximum(atama, 0)[..., None], axis=1)  # [S,10,2]
    w = (atama >= 0).astype(np.float64)
    return (al * w[..., None]).sum(0) / np.maximum(w.sum(0), 1)[:, None]


def sablon_ogren(sekil, gecerli, maks_tur: int = 50):
    """Kısıtlı k-ortalamalar. Başlangıç = mevcut (rol grubu) sıralamanın rol ortalamaları."""
    baslangic = np.where(gecerli, np.arange(N_SAHA)[None], -1)
    sablon = _rol_ortalamasi(sekil, gecerli, baslangic)
    onceki = None
    for tur in range(maks_tur):
        atama = sablona_ata(sekil, gecerli, sablon)
        sablon = _rol_ortalamasi(sekil, gecerli, atama)
        if onceki is not None and np.array_equal(atama, onceki):
            break
        onceki = atama
    return sablon, tur + 1


def yuva_haritasi(pos, mask, rol, segment_id, taraf: int, sablon) -> np.ndarray:
    """[S,11] yeni yuva → eski yuva (taraf içinde, -1 boş). 0 = kaleci, 1–10 = şablon rolleri."""
    sekil, gecerli, saha_yuva, kaleci = segment_sekilleri(pos, mask, rol, segment_id, taraf)
    atama = sablona_ata(sekil, gecerli, sablon)
    saha = np.where(atama >= 0, np.take_along_axis(saha_yuva, np.maximum(atama, 0), axis=1), -1)
    return np.concatenate([kaleci[:, None], saha], axis=1)


@dataclass
class Sablon:
    hucum: np.ndarray     # [10,2] metre, takım merkezine göre
    savunma: np.ndarray   # [10,2]
    tur: tuple = (0, 0)   # yakınsama tur sayıları (bilgi)

    @classmethod
    def ogren(cls, maclar, klasor: Path):
        sablonlar, turlar = [], []
        for taraf in (0, 1):
            parcalar = [segment_sekilleri(d["pos"], d["mask"], d["rol"], d["segment_id"], taraf)[:2]
                        for d in (np.load(Path(klasor) / f"{m}.npz") for m in maclar)]
            sekil = np.concatenate([p[0] for p in parcalar])
            gecerli = np.concatenate([p[1] for p in parcalar])
            s, t = sablon_ogren(sekil, gecerli)
            sablonlar.append(s)
            turlar.append(t)
        return cls(sablonlar[0], sablonlar[1], tuple(turlar))

    def yeniden_sirala(self, pos, mask, rol, segment_id):
        """Ortak format dizilerini şablon yuvalarına göre yeniden sıralar (yeni kopyalar döner)."""
        pos2, mask2, rol2 = pos.copy(), mask.copy(), rol.copy()
        for taraf, sab in ((0, self.hucum), (1, self.savunma)):
            b = taraf * N_TAKIM
            harita = yuva_haritasi(pos, mask, rol, segment_id, taraf, sab)[segment_id]  # [N,11]
            bos = harita < 0
            eski = b + np.maximum(harita, 0)
            satir = np.arange(len(pos))[:, None]
            pos2[:, b:b + N_TAKIM] = np.where(bos[..., None], 0, pos[satir, eski])
            mask2[:, b:b + N_TAKIM] = np.where(bos, False, mask[satir, eski])
            rol2[:, b:b + N_TAKIM] = np.where(bos, 0, rol[satir, eski])
        return pos2, mask2, rol2
