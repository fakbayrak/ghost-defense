# Dataset + metrikler + basit baseline'lar — hafta 4–5 (IDSSE)

Tarih: 2026-11-03 · Durum: onaylandı, tamamlandı (2026-11-03) — sonuçlar `docs/veri_notlari.md`

Takvim satırları: 21 (Dataset + bölme), 22 (metrikler), 23 (rol ortalaması baseline'ları),
24 (sabit hız; tek satırlık iş olduğu için buraya alındı). LSTM ayrı plan.

## Ön bulgular (2026-11-03)
- 7 npz toplam ≈211 bin kare (10 Hz), NaN yok; eksik yuva yalnız J03WN1 (%4,1) ve J03WQQ (%1,2).
- Segment süresi (s) yüzdelikleri 10/25/50/75/90: 2,4 / 3,4 / 5,7 / 11,3 / 22,9.
  Pencere uzadıkça kullanılabilen segment hızla düşüyor:

  | Pencere | Kullanılan segment | Pencere sayısı (1 s adım) |
  |---|---|---|
  | 4 s | %67 | ≈14,9 bin |
  | 5 s | %56 | ≈13,3 bin |
  | 6 s | %49 | ≈12,0 bin |

## Kararlar (önerilen)
1. **Pencere:** 5 s = 50 kare; geçmiş H = 2 s (20 kare), gelecek F = 3 s (30 kare).
   Pencere segment sınırını geçmez; segment < 5 s ise atılır. Adım: eğitimde 0,5 s, değerlendirmede 1 s.
   H/F parametre; sayılar netleşince değiştirilebilir.
2. **İki kurulum** (aynı pencereler, aynı metrik aralığı):
   - **A — hayalet (ana):** koşul = 50 kare boyunca hücum (11) + top (x, y, z); savunma **hiç verilmez**.
     Model tüm pencerede savunmayı üretir; metrik gelecek 30 kare üzerinden (B ile kıyas için),
     ayrıca tam 50 kare için de raporlanır.
   - **B — tahmin:** A'ya ek olarak savunmanın ilk 20 karesi verilir; son 30 kare tahmin edilir.
     Sabit hız ve (sonraki planda) LSTM yalnız burada anlamlı.
   - Hücum ve top gelecekte de verilir: hayalet = "gerçek hücuma tipik savunma tepkisi".
3. **Bölme:** bir-maçı-dışarıda-bırak, 7 kat. Kalan 6 maçın biri doğrulama (listede test maçından
   sonraki maç, döngüsel), 5'i eğitim. Kat başına + ortalama ± std raporlanır.
   Not: 7 maçın 5'inde Düsseldorf var → takım bazında ayrıca bakılacak (10. hafta).
4. **Koordinat:** Dataset metre döndürür; [0,1] normalizasyon (`/ [105, 68]`) ayrı fonksiyon,
   modelin girişinde. Metrikler her zaman metre.
5. **Top z:** girdiye dahil, ama `top_z_kullan` bayrağıyla kapatılabilir (PFF'de olmayabilir).
6. **Maske:** eksik savunma yuvası metriğe girmez; eksik hücum/top yuvası girdide 0 + maske kanalı.

## Adımlar
1. **Pencere + Dataset** (`src/ghostdef/data/pencere.py`):
   - `pencere_indeksleri(segment_id, uzunluk, adim)` → başlangıç indisleri.
   - `PencereVeriSeti(mac_listesi, kurulum="A"|"B", ...)` (torch `Dataset`): npz'leri belleğe alır
     (toplam ≈25 MB), örnek başına `hucum [50,11,2]`, `top [50,3]`, `savunma [50,11,2]`,
     `mask [50,23]`, `rol [22]`, `mac_id`, `segment_id`.
   - `normalize / denormalize`.
2. **Bölme** (`src/ghostdef/data/bolme.py`): `mac_disari_katlar(maclar)` → 7 × (eğitim, doğrulama, test).
3. **Metrikler** (`src/ghostdef/metrikler.py`, torch): tahmin `[B,K,T,11,2]`, deterministik K=1.
   - ADE, FDE (oyuncu ve kare ortalaması, maskeli).
   - minADE_K / minFDE_K: oyuncu başına en iyi örnek.
   - minSADE_K / minSFDE_K: takım düzeyinde en iyi tek örnek (sahne tutarlılığı; asıl üretken kıyas).
4. **Baseline'lar** (`src/ghostdef/baseline/basit.py`), eğitim maçlarından "fit":
   - **Rol ortalaması:** her savunma yuvasının ortalama konumu (sabit).
   - **Topa göre rol ortalaması:** top konumu 7×5 ızgaraya (≈15×13,6 m) bölünür; her hücre ve yuva için
     ortalama konum. Boş/az örnekli hücre (<50 kare) → genel rol ortalaması. Her karede topa göre seçilir.
   - **Sabit hız (yalnız B):** son 2 karenin (0,2 s) ortalama hızı ile doğrusal uzatma.
5. **Değerlendirme betiği** (`scripts/baseline_degerlendir.py`): 7 kat × 3 baseline × 2 kurulum →
   `outputs/sonuclar/baseline_idsse.csv` + konsola Markdown tablo (ADE/FDE, metre).
6. **Testler** (`tests/test_pencere.py`, `test_metrikler.py`, `test_baseline.py`):
   pencere segment sınırını geçmiyor; katlarda maç sızıntısı yok; elle hesaplanmış küçük örnekte
   ADE/FDE/minADE/minSADE; maskeli yuva metriği etkilemiyor; sabit hız düz çizgide 0 hata;
   normalize∘denormalize = birim.
7. **Kapanış:** `docs/veri_notlari.md`'ye pencere/kurulum tanımı; takvimde satır 21–24 Durum + tarih;
   satır 19 notu (StatsBomb → 11. hafta, Metrica = PFF B planı); CLAUDE.md "Şu anki adım" güncellemesi.

## Başarı ölçütü
- Testler geçer; betik 7 katta hatasız çalışır.
- Beklenen sıra: topa göre rol ort. < rol ort. (A'da); B'de sabit hız kısa ufukta en iyisi olmalı.
  Bu sıra çıkmazsa önce veri/metrik hatası aranır.

## Kapsam dışı
LSTM, Transformer, eğitim döngüsü, PFF/Metrica/StatsBomb okuyucuları, veri artırma (y aynalama).
