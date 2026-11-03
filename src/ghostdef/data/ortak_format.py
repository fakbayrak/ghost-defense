"""Sağlayıcıdan bağımsız dönüşümler: HamMac → yeniden örnekleme → segment → yön → rol yuvaları.

Koordinat kuralı: metre, saha 105×68, orijin sol alt köşe. Yeniden ölçekleme ve
[0,1] normalizasyonu modelin girişinde yapılır, burada değil.
Ortak format her karede 23 nesne tutar: 0–10 hücum (top sahibi), 11–21 savunma, 22 top.
"""
from dataclasses import dataclass

import numpy as np

from . import roller

SAHA_X, SAHA_Y = 105.0, 68.0
N_TAKIM = 11
TOP = 2 * N_TAKIM  # top yuvasının indisi (22)


@dataclass
class HamMac:
    """Bir maçın sağlayıcı formatından çıkarılmış, henüz işlenmemiş hali.

    Oyuncu ekseni P maçta görünen tüm oyunculardır (yedekler dahil); yokken NaN.
    """
    mac_id: str
    fps: float
    t: np.ndarray          # [T] periyot başından saniye
    period: np.ndarray     # [T] int
    xy: np.ndarray         # [T, P, 2] float32, NaN = veri yok
    top: np.ndarray        # [T, 3] float32 (x, y, z)
    oyunda: np.ndarray     # [T] bool
    sahip: np.ndarray      # [T] int8: 0 ev, 1 deplasman, -1 belirsiz
    oyuncu_id: list
    oyuncu_takim: np.ndarray  # [P] 0/1
    oyuncu_rol: list          # [P] sağlayıcı pozisyon kodu
    # Yayın tracking'i (PFF): konum kamerada görüldü mü, yoksa bağlamdan tahmin mi?
    # None = sağlayıcı ayırmıyor (IDSSE) → veri olan her konum görünür sayılır.
    gorunur: np.ndarray | None = None      # [T, P] bool
    top_gorunur: np.ndarray | None = None  # [T] bool


# ---------------------------------------------------------------- yeniden örnekleme
def _ara_deger_indis(ts: np.ndarray, tg: np.ndarray, maks_bosluk: float):
    """Hedef zamanlar için (i0, i1, w, gecerli). Boşluk aşılırsa geçersiz."""
    i1 = np.clip(np.searchsorted(ts, tg, side="left"), 1, len(ts) - 1)
    i0 = i1 - 1
    tam = ts[i1] == tg            # hedef zaman bir kareye denk geliyorsa
    i0 = np.where(tam, i1, i0)
    dt = ts[i1] - ts[i0]
    w = np.where(dt > 0, (tg - ts[i0]) / np.where(dt > 0, dt, 1), 0.0)
    gecerli = (dt <= maks_bosluk) & (tg >= ts[0]) & (tg <= ts[-1])
    return i0, i1, w, gecerli


def _dogrusal(a: np.ndarray, i0, i1, w, gecerli) -> np.ndarray:
    """NaN'ı koruyan doğrusal ara değer: komşulardan biri NaN ise sonuç NaN."""
    wb = w.reshape((-1,) + (1,) * (a.ndim - 1))
    sonuc = np.where(wb == 0, a[i0], (1 - wb) * a[i0] + wb * a[i1])
    sonuc = sonuc.astype(np.float32)
    sonuc[~gecerli] = np.nan
    return sonuc


def yeniden_ornekle(ham: HamMac, hz: float = 10.0) -> HamMac:
    """Her periyodu ortak hz ızgarasına taşır. Konumlar doğrusal, kategorikler en yakın kare.

    Görünürlük temkinli: ara değerin iki komşusu da görünürse görünür.
    """
    parcalar = []
    for p in np.unique(ham.period):
        s = ham.period == p
        ts = ham.t[s]
        tg = np.arange(np.ceil(ts[0] * hz) / hz, ts[-1] + 1e-9, 1 / hz)
        tg = np.round(tg, 6)
        i0, i1, w, ok = _ara_deger_indis(ts, tg, maks_bosluk=1.5 / ham.fps)
        yakin = np.where(w < 0.5, i0, i1)
        parca = dict(
            t=tg, period=np.full(len(tg), p, dtype=np.int8),
            xy=_dogrusal(ham.xy[s], i0, i1, w, ok),
            top=_dogrusal(ham.top[s], i0, i1, w, ok),
            oyunda=ham.oyunda[s][yakin] & ok,
            sahip=np.where(ok, ham.sahip[s][yakin], -1).astype(np.int8),
        )
        if ham.gorunur is not None:
            g = ham.gorunur[s]
            parca["gorunur"] = g[i0] & g[i1] & ok[:, None]
        if ham.top_gorunur is not None:
            g = ham.top_gorunur[s]
            parca["top_gorunur"] = g[i0] & g[i1] & ok
        parcalar.append(parca)
    birlesik = {k: np.concatenate([d[k] for d in parcalar]) for k in parcalar[0]}
    return HamMac(ham.mac_id, hz, oyuncu_id=ham.oyuncu_id, oyuncu_takim=ham.oyuncu_takim,
                  oyuncu_rol=ham.oyuncu_rol, **birlesik)


# ---------------------------------------------------------------- ölü top ayıklama
def segmentler(ham: HamMac, min_sure: float = 2.0) -> np.ndarray:
    """Top oyunda + sahiplik değişmeden süren kesintisiz parçalar. [T] segment no, -1 = dışarıda."""
    iyi = ham.oyunda & (ham.sahip >= 0) & ~np.isnan(ham.top[:, 0])
    yeni = np.ones(len(iyi), dtype=bool)
    yeni[1:] = (~iyi[:-1] | (ham.sahip[1:] != ham.sahip[:-1])
                | (ham.period[1:] != ham.period[:-1])
                | (np.diff(ham.t) > 1.5 / ham.fps))
    grup_no = np.cumsum(yeni) - 1
    seg = np.full(len(iyi), -1, dtype=np.int32)
    min_kare = int(round(min_sure * ham.fps))
    sayac = 0
    for g in np.unique(grup_no[iyi]):
        k = (grup_no == g) & iyi
        if k.sum() >= min_kare:
            seg[k] = sayac
            sayac += 1
    return seg


# ---------------------------------------------------------------- hücum yönü
def ev_saga_hucum(ham: HamMac) -> dict:
    """Her periyotta ev sahibi +x yönüne mi hücum ediyor? Kalecinin medyan x'inden."""
    kaleci = [i for i, r in enumerate(ham.oyuncu_rol)
              if roller.grup(r) == roller.KALECI and ham.oyuncu_takim[i] == 0]
    sonuc = {}
    for p in np.unique(ham.period):
        s = ham.period == p
        if kaleci:
            x = np.nanmedian(ham.xy[s][:, kaleci, 0])
        else:  # yedek yol: ev sahibinin ağırlık merkezi
            x = np.nanmedian(ham.xy[s][:, ham.oyuncu_takim == 0, 0])
        sonuc[int(p)] = bool(x < SAHA_X / 2)  # kendi kalesi solda → sağa hücum
    return sonuc


def dondur_180(xy: np.ndarray) -> np.ndarray:
    """Sahayı merkez etrafında 180° döndürür. Aynalama değil: sol bek solda kalır."""
    out = xy.copy()
    out[..., 0] = SAHA_X - xy[..., 0]
    out[..., 1] = SAHA_Y - xy[..., 1]
    return out


# ---------------------------------------------------------------- rol yuvaları
def yuvalari_sec(xy_seg: np.ndarray, takim_idx: np.ndarray, roller_: list) -> list:
    """Bir segmentte bir takımın oyuncularını 11 yuvaya dizer.

    Sıra: rol grubu (GK→DEF→MID→FWD), grup içinde segment ortalama y artan.
    11'den fazla oyuncu görünürse en çok geçerli karesi olan 11'i alınır.
    Dönüş: oyuncu indisleri listesi (≤11).
    """
    gecerli = ~np.isnan(xy_seg[:, takim_idx, 0])
    sayim = gecerli.sum(0)
    var = takim_idx[sayim > 0]
    if len(var) > N_TAKIM:
        var = takim_idx[np.argsort(-sayim)[:N_TAKIM]]
    ort_y = np.nanmean(xy_seg[:, var, 1], axis=0)
    anahtar = sorted(range(len(var)), key=lambda j: (roller.grup(roller_[var[j]]), ort_y[j]))
    return [int(var[j]) for j in anahtar]


def ortak_formata(ham: HamMac, seg: np.ndarray) -> dict:
    """Yeniden örneklenmiş HamMac + segmentler → model için ortak format sözlüğü (yalnız segment kareleri)."""
    yon = ev_saga_hucum(ham)
    secili = np.flatnonzero(seg >= 0)
    N = len(secili)
    pos = np.full((N, TOP + 1, 2), np.nan, dtype=np.float32)
    gor = np.zeros((N, TOP + 1), dtype=bool)
    top_z = np.full(N, np.nan, dtype=np.float32)
    rol = np.zeros((N, 2 * N_TAKIM), dtype=np.int8)
    seg_sayisi = int(seg.max()) + 1
    seg_hucum = np.zeros(seg_sayisi, dtype=np.int8)
    seg_oyuncu = np.full((seg_sayisi, 2 * N_TAKIM), -1, dtype=np.int16)

    cikis_seg = seg[secili]
    for s in range(seg_sayisi):
        k = secili[cikis_seg == s]           # HamMac indisleri
        j = np.flatnonzero(cikis_seg == s)   # çıktı indisleri
        hucum = int(ham.sahip[k[0]])
        p = int(ham.period[k[0]])
        sagdan_sola = yon[p] != (hucum == 0)  # hücum eden takım -x'e gidiyorsa döndür
        xy = ham.xy[k]
        top = ham.top[k, :2]
        if sagdan_sola:
            xy, top = dondur_180(xy), dondur_180(top)
        for taraf, takim in enumerate([hucum, 1 - hucum]):
            idx = np.flatnonzero(ham.oyuncu_takim == takim)
            yuvalar = yuvalari_sec(xy, idx, ham.oyuncu_rol)
            for y, oi in enumerate(yuvalar):
                yuva = taraf * N_TAKIM + y
                pos[j, yuva] = xy[:, oi]
                rol[j, yuva] = roller.kod_id(ham.oyuncu_rol[oi])
                seg_oyuncu[s, yuva] = oi
                if ham.gorunur is not None:
                    gor[j, yuva] = ham.gorunur[k, oi]
        pos[j, TOP] = top
        if ham.top_gorunur is not None:
            gor[j, TOP] = ham.top_gorunur[k]
        top_z[j] = ham.top[k, 2]
        seg_hucum[s] = hucum

    mask = ~np.isnan(pos[..., 0])
    if ham.gorunur is None:
        gor[:, :TOP] = True
    if ham.top_gorunur is None:
        gor[:, TOP] = True
    gor &= mask
    return dict(
        pos=np.nan_to_num(pos), mask=mask, gorunur=gor, top_z=np.nan_to_num(top_z), rol=rol,
        segment_id=cikis_seg, period=ham.period[secili], t=ham.t[secili].astype(np.float32),
        seg_hucum_takim=seg_hucum, seg_oyuncu=seg_oyuncu,
        oyuncu_id=np.array(ham.oyuncu_id), oyuncu_takim=ham.oyuncu_takim,
        oyuncu_rol=np.array(ham.oyuncu_rol), rol_kodlari=np.array(roller.ROL_KODLARI),
        mac_id=np.array(ham.mac_id), hz=np.array(ham.fps),
    )


# ---------------------------------------------------------------- akıl sağlığı
def kontrol(of: dict, maks_oyuncu_hiz: float = 12.0, maks_top_hiz: float = 45.0) -> dict:
    """Ortak format üzerinde basit kontroller; rapor sözlüğü döner."""
    pos, mask, seg, hz = of["pos"], of["mask"], of["segment_id"], float(of["hz"])
    ayni = (seg[1:] == seg[:-1])[:, None] & mask[1:] & mask[:-1]
    hiz = np.linalg.norm(np.diff(pos, axis=0), axis=-1) * hz
    hiz = np.where(ayni, hiz, 0)
    sure = len(seg) / hz
    rapor = {
        "segment_sayisi": int(seg.max()) + 1,
        "oyunda_sure_dk": round(sure / 60, 1),
        "ort_segment_s": round(sure / (int(seg.max()) + 1), 1),
        "saha_disi_orani": float(((pos[..., 0] < -3) | (pos[..., 0] > SAHA_X + 3)
                                  | (pos[..., 1] < -3) | (pos[..., 1] > SAHA_Y + 3))[mask].mean()),
        "oyuncu_hiz_asim_orani": float((hiz[:, :TOP] > maks_oyuncu_hiz).sum() / max(ayni[:, :TOP].sum(), 1)),
        "oyuncu_hiz_p999": float(np.percentile(hiz[:, :TOP][ayni[:, :TOP]], 99.9)),
        "top_hiz_asim_orani": float((hiz[:, TOP] > maks_top_hiz).sum() / max(ayni[:, TOP].sum(), 1)),
        "eksik_yuva_orani": float(1 - mask[:, :TOP].mean()),
        "hucum_gorunur_orani": float(of["gorunur"][:, :N_TAKIM][mask[:, :N_TAKIM]].mean()),
        "savunma_gorunur_orani": float(of["gorunur"][:, N_TAKIM:TOP][mask[:, N_TAKIM:TOP]].mean()),
        "top_gorunur_orani": float(of["gorunur"][:, TOP][mask[:, TOP]].mean()),
        "hucum_ort_x": float(pos[:, :N_TAKIM, 0][mask[:, :N_TAKIM]].mean()),
        "savunma_ort_x": float(pos[:, N_TAKIM:TOP, 0][mask[:, N_TAKIM:TOP]].mean()),
    }
    return rapor
