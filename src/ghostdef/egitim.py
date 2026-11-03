"""Genel eğitim döngüsü: maskeli Öklid kaybı, AdamW + kosinüs, doğrulama ADE'sinde erken durdurma.

Veri `PencereVeriSeti.yigin()` sözlükleri olarak bir kez cihaza alınır; batch'ler indisle kesilir
(DataLoader işçisi yok). Model arayüzü: `model(yigin) → [B,K,L,11,2]` metre.
"""
import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from . import metrikler
from .data.ortak_format import N_TAKIM, TOP


@dataclass
class EgitimAyari:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 256
    epoch: int = 40
    sabir: int = 6
    tohum: int = 0
    kayip_bas: int = 0          # kaybın başladığı kare (A: 0, B: geçmiş uzunluğu)
    metrik_bas: int = 20        # doğrulama ADE'sinin başladığı kare (son 3 s)


def cihaza(yigin: dict, cihaz) -> dict:
    return {k: v.to(cihaz) for k, v in yigin.items()}


def dilim(yigin: dict, idx) -> dict:
    return {k: v[idx] for k, v in yigin.items()}


def boyut(yigin: dict) -> int:
    return len(yigin["top"])


def kayip(tahmin, yigin, bas: int = 0, eps: float = 1e-6) -> torch.Tensor:
    """Maskeli ortalama Öklid uzaklığı (metre). Sıfır uzaklıkta türev NaN olmasın diye eps."""
    gercek = yigin["hedef"][:, bas:]
    mask = yigin["mask"][:, bas:, N_TAKIM:TOP].to(gercek.dtype)
    fark = tahmin[:, :, bas:] - gercek[:, None]
    d = torch.sqrt((fark ** 2).sum(-1) + eps) * mask[:, None]
    return d.sum() / (mask.sum() * tahmin.shape[1]).clamp_min(1)


@torch.no_grad()
def tahmin_et(model, yigin, batch: int = 1024) -> torch.Tensor:
    model.eval()
    n = boyut(yigin)
    return torch.cat([model(dilim(yigin, slice(i, i + batch))) for i in range(0, n, batch)])


def degerlendir(model, yigin, bas: int, son: int | None = None) -> dict:
    t = tahmin_et(model, yigin)
    son = son or t.shape[2]
    mask = yigin["mask"][:, bas:son, N_TAKIM:TOP]
    return metrikler.hepsi(t[:, :, bas:son], yigin["hedef"][:, bas:son], mask)


def egit(model, egitim: dict, dogrulama: dict, ayar: EgitimAyari, gunluk_yolu: Path | None = None):
    """En iyi doğrulama ADE'sine sahip ağırlıklarla modeli ve epoch günlüğünü döndürür."""
    torch.manual_seed(ayar.tohum)
    opt = torch.optim.AdamW(model.parameters(), lr=ayar.lr, weight_decay=ayar.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ayar.epoch)
    gen = torch.Generator().manual_seed(ayar.tohum)
    n = boyut(egitim)
    en_iyi, en_iyi_durum, en_iyi_epoch, gunluk = float("inf"), None, -1, []
    for ep in range(ayar.epoch):
        t0 = time.time()
        model.train()
        sira = torch.randperm(n, generator=gen).to(egitim["top"].device)
        toplam, adim = 0.0, 0
        for i in range(0, n, ayar.batch):
            b = dilim(egitim, sira[i:i + ayar.batch])
            k = kayip(model(b), b, ayar.kayip_bas)
            opt.zero_grad(set_to_none=True)
            k.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            toplam, adim = toplam + k.item(), adim + 1
        sched.step()
        dog = degerlendir(model, dogrulama, ayar.metrik_bas)["ADE"]
        gunluk.append(dict(epoch=ep, egitim_kayip=toplam / adim, dogrulama_ADE=dog,
                           lr=opt.param_groups[0]["lr"], sure_s=round(time.time() - t0, 1)))
        if dog < en_iyi:
            en_iyi, en_iyi_durum, en_iyi_epoch = dog, copy.deepcopy(model.state_dict()), ep
        elif ep - en_iyi_epoch >= ayar.sabir:
            break
    model.load_state_dict(en_iyi_durum)
    if gunluk_yolu is not None:
        _csv_yaz(gunluk_yolu, gunluk)
    return model, gunluk, en_iyi_epoch


def kaydet(yol: Path, model, ayar: EgitimAyari, ek: dict | None = None):
    yol.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"durum": model.state_dict(), "model_ayar": model.ayar,
                "egitim_ayar": asdict(ayar), **(ek or {})}, yol)


def yukle(yol: Path, sinif, cihaz="cpu"):
    k = torch.load(yol, map_location=cihaz, weights_only=True)
    model = sinif(**k["model_ayar"]).to(cihaz)
    model.load_state_dict(k["durum"])
    return model, k


def _csv_yaz(yol: Path, satirlar: list):
    import csv
    yol.parent.mkdir(parents=True, exist_ok=True)
    with open(yol, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)
