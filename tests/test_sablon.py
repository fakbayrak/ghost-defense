"""Şablon eşleştirme testleri (sentetik dizilişler)."""
import numpy as np

from ghostdef.data import roller
from ghostdef.data.pencere import PencereVeriSeti
from ghostdef.data.sablon import Sablon, sablona_ata, segment_sekilleri

TW = roller.kod_id("TW")
# Takım merkezine göre şekiller (x hücum yönü). 4-4-2: bek, stoper, kanat, orta, forvet.
D442 = np.array([[-15, -25], [-15, -7], [-15, 7], [-15, 25],
                 [5, -25], [5, -7], [5, 7], [5, 25], [15, -6], [15, 6]], float)
# 3-5-2: 3 stoper, 2 kanat beki (stoperlerin biraz önünde), 3 orta, 2 forvet.
D352 = np.array([[-15, -8], [-15, 0], [-15, 8], [-10, -25], [-10, 25],
                 [0, -8], [0, 0], [0, 8], [15, -6], [15, 6]], float)
# Modül şekli takım merkezine göre ölçer → şablonlar da merkezlenmiş olmalı.
D442, D352 = D442 - D442.mean(0), D352 - D352.mean(0)
KANAT = [0, 3, 4, 7]   # 4-4-2 şablonunda bek ve kanat orta rolleri (|y| = 25)


def mac(dizilisler, kare=30, karistir=False, eksik=None, tohum=0):
    """Her segment bir diziliş; iki taraf aynı dizilişte. Kaleci her tarafta TW kodlu."""
    rng = np.random.default_rng(tohum)
    S = len(dizilisler)
    pos = np.zeros((S * kare, 23, 2), np.float32)
    mask = np.ones((S * kare, 23), bool)
    rol = np.full((S * kare, 22), roller.kod_id("ZO"), np.int8)
    seg = np.repeat(np.arange(S), kare).astype(np.int32)
    for s, d in enumerate(dizilisler):
        k = slice(s * kare, (s + 1) * kare)
        for taraf in (0, 1):
            b = taraf * 11
            sira = rng.permutation(10) if karistir else np.arange(10)
            pos[k, b] = [10 if taraf == 0 else 95, 34]
            rol[k, b] = TW
            pos[k, b + 1:b + 11] = d[sira] + [52.5, 34] + rng.normal(0, 0.3, (kare, 10, 2))
        if eksik is not None:
            mask[k, eksik] = False
            pos[k, eksik] = 0
    return dict(pos=pos, mask=mask, rol=rol, segment_id=seg)


def kaydet(klasor, ad, d):
    np.savez(klasor / f"{ad}.npz", top_z=np.zeros(len(d["pos"]), np.float32), **d)


def test_kanat_bekleri_bek_rolune_atanir():
    d = mac([D352])
    sekil, gecerli, _, kaleci = segment_sekilleri(d["pos"], d["mask"], d["rol"], d["segment_id"], 1)
    assert kaleci[0] == 0
    atama = sablona_ata(sekil, gecerli, D442)[0]
    # D352'de kanat bekleri 3. ve 4. oyuncu → ikisi de kanat rollerine (bek veya kanat orta, |y| = 25).
    # Hangi kanat rolü olduğu toplam maliyete bağlı (3 stoper bir tarafı kaydırabilir).
    roller_ = {int(np.flatnonzero(atama == o)[0]) for o in (3, 4)}
    assert roller_ <= set(KANAT)
    assert {int(np.flatnonzero(atama == o)[0]) for o in (8, 9)} == {8, 9}   # forvetler forvette


def test_oyuncu_sirasi_karisinca_sonuc_degismez(tmp_path):
    kaydet(tmp_path, "E", mac([D442] * 5 + [D352] * 3))
    sab = Sablon.ogren(["E"], tmp_path)
    a = mac([D442, D352], tohum=1)
    b = mac([D442, D352], tohum=1, karistir=True)
    pa = sab.yeniden_sirala(a["pos"], a["mask"], a["rol"], a["segment_id"])[0]
    pb = sab.yeniden_sirala(b["pos"], b["mask"], b["rol"], b["segment_id"])[0]
    # Gürültü aynı tohumla aynı oyuncuya değil aynı yuvaya eklendiği için konumlar 1 m içinde aynı olmalı
    np.testing.assert_allclose(pa, pb, atol=2.0)
    assert (pa[:, 0] == [10, 34]).all() and (pa[:, 11] == [95, 34]).all()   # kaleciler 0. yuvada


def test_eksik_oyuncu_rolu_maskelenir(tmp_path):
    kaydet(tmp_path, "E", mac([D442] * 4))
    sab = Sablon.ogren(["E"], tmp_path)
    d = mac([D442], eksik=11 + 4)          # savunmada 4. eski yuva (bek) yok
    pos2, mask2, _ = sab.yeniden_sirala(d["pos"], d["mask"], d["rol"], d["segment_id"])
    assert mask2[:, 11:22].sum(1).tolist() == [10] * 30
    assert mask2[:, :11].all()
    assert (pos2[~mask2] == 0).all()


def test_sablon_ogrenir_ve_yalniz_verilen_maclari_okur(tmp_path):
    kaydet(tmp_path, "E", mac([D442] * 6))
    (tmp_path / "BOZUK.npz").write_bytes(b"bozuk")
    sab = Sablon.ogren(["E"], tmp_path)
    np.testing.assert_allclose(np.sort(sab.savunma, axis=0), np.sort(D442, axis=0), atol=0.5)


def test_dataset_sablonlu_ve_sablonsuz(tmp_path):
    kaydet(tmp_path, "E", mac([D442] * 4 + [D352] * 4, kare=60))
    sab = Sablon.ogren(["E"], tmp_path)
    eski = PencereVeriSeti(["E"], klasor=tmp_path)
    yeni = PencereVeriSeti(["E"], klasor=tmp_path, sablon=sab)
    d = np.load(tmp_path / "E.npz")
    np.testing.assert_array_equal(eski.pos, d["pos"])          # sablon=None: dokunulmaz
    assert len(eski) == len(yeni)
    # aynı oyuncular, farklı sıra: kare başına konum kümesi aynı
    np.testing.assert_allclose(np.sort(yeni.pos[:, 11:22, 0], 1), np.sort(eski.pos[:, 11:22, 0], 1))
