"""Segment içi sabit uzunlukta pencereler ve PyTorch Dataset.

Pencere = geçmiş H + gelecek F kare (varsayılan 20 + 30 = 5 s, 10 Hz); segment sınırını geçmez.
İki kurulum aynı pencereleri kullanır:
  A — hayalet: koşul = tüm pencere boyunca hücum + top; savunma hiç verilmez.
  B — tahmin: A'ya ek olarak savunmanın ilk H karesi (`savunma_gecmis`) verilir.
`hedef` (tüm pencere savunma konumu) her iki kurulumda da kayıp/metrik içindir, modele girmez.
Konumlar metre; [0,1] normalizasyonu `normalize` ile modelin girişinde yapılır.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .ortak_format import N_TAKIM, SAHA_X, SAHA_Y, TOP

VERI_KOK = Path(__file__).resolve().parents[3] / "data" / "processed"
GECMIS, GELECEK = 20, 30
SAV = slice(N_TAKIM, TOP)  # savunma yuvaları 11–21


def pencere_indeksleri(segment_id: np.ndarray, uzunluk: int, adim: int) -> np.ndarray:
    """Segment sınırını geçmeyen pencerelerin başlangıç indisleri (segmentler ardışık kareler)."""
    sinir = np.flatnonzero(np.diff(segment_id) != 0) + 1
    baslar, sonlar = np.r_[0, sinir], np.r_[sinir, len(segment_id)]
    parcalar = [np.arange(b, s - uzunluk + 1, adim) for b, s in zip(baslar, sonlar) if s - b >= uzunluk]
    return np.concatenate(parcalar).astype(np.int64) if parcalar else np.zeros(0, dtype=np.int64)


def _olcek(xy):
    if torch.is_tensor(xy):
        return torch.tensor([SAHA_X, SAHA_Y], dtype=xy.dtype, device=xy.device)
    return np.array([SAHA_X, SAHA_Y], dtype=xy.dtype)


def normalize(xy):
    """Metre → [0,1]. Son eksen (x, y) olmalı."""
    return xy / _olcek(xy)


def denormalize(xy):
    return xy * _olcek(xy)


class PencereVeriSeti(Dataset):
    """Maç listesindeki npz'leri belleğe alır (IDSSE 7 maç ≈ 40 MB) ve pencere örnekleri verir.

    Örnek alanları: `hucum` [L,11,2], `top` [L,3] (x, y, z; z kapalıysa 0), `hedef` [L,11,2],
    `mask` [L,23] bool, `rol` [22], `mac` (maç listesindeki indis), `segment`;
    kurulum B'de ayrıca `savunma_gecmis` [H,11,2].
    """

    def __init__(self, maclar, klasor: Path = VERI_KOK / "idsse", kurulum: str = "A",
                 gecmis: int = GECMIS, gelecek: int = GELECEK, adim: int = 5,
                 top_z_kullan: bool = True, sablon=None):
        """`sablon` (data.sablon.Sablon) verilirse yuvalar şablon rollerine göre yeniden sıralanır;
        None ise ortak formattaki rol grubu sırası kullanılır."""
        if kurulum not in ("A", "B"):
            raise ValueError(f"kurulum 'A' veya 'B' olmalı, {kurulum!r} verildi")
        self.maclar, self.kurulum = list(maclar), kurulum
        self.gecmis, self.gelecek, self.uzunluk = gecmis, gelecek, gecmis + gelecek
        alanlar = {k: [] for k in ("pos", "mask", "top_z", "rol", "segment_id", "mac")}
        baslar, ofset = [], 0
        for mi, m in enumerate(self.maclar):
            d = dict(np.load(Path(klasor) / f"{m}.npz"))
            if sablon is not None:
                d["pos"], d["mask"], d["rol"] = sablon.yeniden_sirala(
                    d["pos"], d["mask"], d["rol"], d["segment_id"])
            n = len(d["segment_id"])
            for k in ("pos", "mask", "top_z", "rol", "segment_id"):
                alanlar[k].append(d[k])
            alanlar["mac"].append(np.full(n, mi, dtype=np.int64))
            baslar.append(pencere_indeksleri(d["segment_id"], self.uzunluk, adim) + ofset)
            ofset += n
        # Tüm segment kareleri (kısa segmentler dahil): baseline'lar bunlardan "fit" eder.
        self.pos = np.concatenate(alanlar["pos"]).astype(np.float32)
        self.mask = np.concatenate(alanlar["mask"]).astype(bool)
        self.top_z = np.concatenate(alanlar["top_z"]).astype(np.float32)
        if not top_z_kullan:
            self.top_z = np.zeros_like(self.top_z)
        self.rol = np.concatenate(alanlar["rol"]).astype(np.int64)
        self.segment_id = np.concatenate(alanlar["segment_id"]).astype(np.int64)
        self.mac = np.concatenate(alanlar["mac"])
        self.baslar = np.concatenate(baslar) if baslar else np.zeros(0, dtype=np.int64)

    def __len__(self):
        return len(self.baslar)

    def yigin(self, indisler=None) -> dict:
        """Verilen pencereleri (varsayılan: hepsi) tek seferde tensör sözlüğü olarak döndürür."""
        b = self.baslar if indisler is None else self.baslar[np.asarray(indisler)]
        k = b[:, None] + np.arange(self.uzunluk)  # [B, L]
        pos = self.pos[k]
        ornek = {
            "hucum": pos[:, :, :N_TAKIM],
            "top": np.concatenate([pos[:, :, TOP], self.top_z[k][..., None]], axis=-1),
            "hedef": pos[:, :, SAV],
            "mask": self.mask[k],
            "rol": self.rol[b],
            "mac": self.mac[b],
            "segment": self.segment_id[b],
        }
        if self.kurulum == "B":
            ornek["savunma_gecmis"] = ornek["hedef"][:, :self.gecmis]
        return {a: torch.from_numpy(np.ascontiguousarray(v)) for a, v in ornek.items()}

    def __getitem__(self, i):
        return {a: v[0] for a, v in self.yigin([i]).items()}
