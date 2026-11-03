# PFF okuyucusu planı

Tarih: 2026-11-03 · Durum: onaylandı

## Kararlar (3840/3850/3841 ölçümlerine dayanarak)
1. **Yumuşatılmış konum** (oyuncu + top). Ham izler %50 yinelenen kare (etkin 15 Hz), oyuncu hızı
   p99,9 42 m/s; yumuşatılmış 8,4 m/s (IDSSE 8,1–8,4). Top: ham p90 10,8 m sahipten uzak, yumuşatılmış
   olaya sabit (olaylar arası doğrusal → havadaki top eğrisi kaybolur; alan farkı olarak not).
   Görünürlük yumuşatılmış kayıtlardaki `visibility`'den. Ham iz sürümü 3. araştırma sorusuna kalır.
2. **Ölü top = olay durum makinesi + sahiplik + top var.** OUT/END/VID → ölü; OTB/başlama → oyunda;
   sahiplik `home_ball` (sonraki olaya kadar). Topsuz kareler kamera dışı (oyuncuların %100'ü ESTIMATED),
   3840'ta 45 dk. Segment kuralı IDSSE ile aynı (≥2 s, sahiplik değişmeden).
3. **Görünürlük eşiği okuyucuda yok.** npz'ye `gorunur` [N,23] yazılır; eşik, maskeli kayıp ve görünürlük
   giriş kanalı Dataset'te. Ön eğitimde {0; 0,5; 0,7} ablasyonu, seçim IDSSE doğrulamasına göre.

## Adımlar
1. `src/ghostdef/data/pff.py` → `yukle(mac_id) -> HamMac`: akışla okuma, yinelenen kare ayıklama,
   `periodElapsedTime`, dönem 3–4 dahil, takım + forma → kadro (`player.id`, `positionGroupType`),
   bilinmeyen forma uyarı; orijin merkez → sol alt, saha ölçeğine göre.
2. `roller.py`: PFF kodları → grup (GK; LCB/MCB/RCB/LB/RB/LWB/RWB; DM/CM/AM/LM/RM; LW/RW/CF).
   Yeni kodlar `ROL_KODLARI` sonuna (IDSSE indisleri korunur).
3. `ortak_format.py`: `HamMac.gorunur` [T,P] isteğe bağlı (yoksa = NaN değil); yeniden örneklemede
   iki komşu da görünürse görünür; npz'ye `gorunur` [N,23].
4. `scripts/veri_isle.py`: `--kaynak idsse|pff`; çıktı `data/processed/<kaynak>/`; PFF 6 süreç paralel;
   rapora görünürlük oranları.
5. Testler `tests/test_pff.py` (sentetik `.jsonl.bz2`): yineleme, koordinat, forma eşleme, görünürlük,
   OUT → ölü, sahiplik; `gorunur` yeniden örnekleme testi; mevcut testler geçmeli.
6. Doğrulama: 64 maç kontrol raporu (hız p99,9, saha dışı, segment dk); bir PFF maçı mini harita
   (ESTIMATED içi boş).
7. Kapanış: `docs/veri_notlari.md`, `CLAUDE.md`.

## Kapsam dışı
Dataset/kayıp değişiklikleri (ön eğitim planı), PFF olay verisi (11. hafta), ham iz sürümü (AS3).

## Riskler
- Forma güveni %76 LOW → kimlik takası rol yuvasını kaydırabilir; yumuşatılmış hız sıçramaları raporda.
- PFF ve DFL rol kodları ayrı gömme → transferde ortak rol sözlüğü gerek (model planında karar).
