# Veri notları

## IDSSE (Bassek vd., 2025) — işlendi 2026-11-01

**Ham format (DFL/Sportec XML):**
- 25 Hz (40 ms), metre, orijin saha merkezi, y ekseni yukarı (sol bek hücum yönüne göre +y).
- Her 7 maçta saha 105×68 (meta `PitchX/PitchY`).
- **Top z koordinatı VAR** (`Z`, metre): medyan ≈0,14 m, p99 ≈11 m, maks. 17–27 m.
- Topta `BallPossession` (1 ev, 2 deplasman) ve `BallStatus` (1 oyunda, 0 ölü) var.
- Pozisyon kodları meta XML'de (`PlayingPosition`): TW, IVL/IVR/IVZ, LV/RV, DML/DMR/DMZ/DLM/DRM,
  ZO, OLM/ORM, LM/RM, HL/HR, LA/RA, STL/STR/STZ. HL/HR anlamı doğrulanmadı (orta sahaya atandı).
- Maç başına konum XML'i ≈400 MB; kloppy ile yükleme ≈30–40 s, ≈1,7 GB RAM.
- kloppy 3.19 hatası: `PositionType.parent` WingBack için KeyError → rol grubu kendi eşlememizle.

**İşlenmiş veri (`data/processed/idsse/<maç>.npz`, 10 Hz, yalnız segment kareleri):**

| Maç | Segment | Oyunda süre (dk) | Ort. segment (s) | Eksik yuva % | Not |
|---|---|---|---|---|---|
| J03WMX Köln–Bayern | 320 | 53,5 | 10,0 | 0,0 | 1. Bundesliga |
| J03WN1 Bochum–Leverkusen | 333 | 42,6 | 7,7 | 4,3 | kırmızı kart |
| J03WOH Düsseldorf–Regensburg | 295 | 46,0 | 9,4 | 0,0 | |
| J03WOY Düsseldorf–Rostock | 329 | 48,4 | 8,8 | 0,0 | |
| J03WPY Düsseldorf–Nürnberg | 301 | 52,3 | 10,4 | 0,0 | |
| J03WQQ Düsseldorf–St. Pauli | 303 | 54,3 | 10,8 | 1,3 | |
| J03WR9 Düsseldorf–Kaiserslautern | 277 | 55,3 | 12,0 | 0,0 | |
| **Toplam** | **2158** | **352,4** | | | |

Segment = top oyunda + sahiplik değişmeden ≥2 s. Akıl sağlığı: oyuncu hızı p99,9 ≈8,1–8,4 m/s,
12 m/s aşımı <0,01 %; saha dışı (±3 m) <0,01 %; karelerin ≈%80'inde topa en yakın oyuncu hücumcu.
Ayrıntı: `data/processed/idsse/kontrol_raporu.json`.

**npz alanları:** `pos` [N,23,2] (0–10 hücum, 11–21 savunma, 22 top; metre, orijin sol alt,
hücum hep +x), `mask` [N,23], `top_z` [N], `rol` [N,22] (`rol_kodlari` indisi; 0 = boş/bilinmiyor),
`segment_id`, `period`, `t`, `seg_hucum_takim` [S] (0 ev / 1 dep), `seg_oyuncu` [S,22]
(`oyuncu_id` indisi), `oyuncu_id/oyuncu_takim/oyuncu_rol`, `mac_id`, `hz`.

**Yöntem kararları:**
- Yön: kalecinin periyot medyan x'inden; hücum eden takım −x'e gidiyorsa 180° döndürme
  (aynalama değil → sol/sağ korunur).
- 10 Hz: zaman ızgarasına doğrusal ara değer; komşulardan biri yoksa NaN (uydurma yok);
  kategorikler en yakın kare.
- Rol sırası: segment içinde sabit; GK → DEF → MID → FWD, grup içinde ortalama y artan
  (hücum için sağdan sola). Formasyon değişince yuvanın anlamı kayabilir → ileride şablon eşleştirme.

## Pencere ve kurulumlar — 2026-11-03

- Pencere 5 s = 50 kare (geçmiş H = 20, gelecek F = 30); segment sınırını geçmez, < 5 s segment atılır
  (segmentlerin %56'sı kalır). Adım: eğitim 0,5 s, değerlendirme 1 s → IDSSE'de ≈12,7 bin test penceresi.
- **A — hayalet:** koşul = tüm pencere hücum + top (x, y, z); savunma verilmez.
  **B — tahmin:** ek olarak savunmanın ilk 20 karesi. Metrik her ikisinde son 30 kare (A'da ayrıca tam pencere).
- Bölme: 7 kat bir-maçı-dışarıda-bırak; doğrulama = listede test maçından sonraki maç.
- Kod: `ghostdef.data.pencere.PencereVeriSeti`, `ghostdef.data.bolme`, `ghostdef.metrikler`,
  `ghostdef.baseline.basit`. Model sonuçları: `docs/deney_notlari.md`.
- Yuva sıralaması tanısı (J03WMX): pencerelerin %8'i 3'lü veya 5'li defans → DEF/MID sınırındaki yuvanın
  anlamı kayıyor; sıralamanın hataya payı ≈1,8 m (ayrıntı deney notlarında).
- **Şablon yuvaları (2026-11-03, önerilen varsayılan):** `PencereVeriSeti(..., sablon=Sablon.ogren(egitim_maclari, klasor))`.
  Her taraf: yuva 0 = kaleci (TW kodu; yoksa kendi kalesine en yakın), yuva 1–10 = şablon rolleri
  (segment ortalama şekli − takım merkezi, Macar ataması). Şablon yalnız eğitim maçlarından → sızıntı yok.
  npz'ler değişmez; `sablon=None` eski rol grubu sırası. `rol` (pozisyon kodu) yuvayla birlikte taşınır.

## PFF FC / Gradient Sports — WC 2022 (indirildi 2026-11-03)

**Kaynak:** PFF'nin herkese açık Google Drive klasörü
(`1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa`; bağlantı J. Bekkers'ın AIMS 2025 sunumunda). Mail yanıtı beklenmeden
erişildi. İndirme: `scripts/veri_pff_indir.py` (gdown) → `data/raw/pff/`. Dosya listesi önbelleği
`data/raw/pff/_drive_liste.tsv`.
- Klasör: `Tracking Data/<id>.jsonl.bz2` (≈42 MB/maç), `Event Data/<id>.json` (≈16 MB/maç; güncel sürüm,
  tarihli alt klasörler eski sürüm), `Metadata/<id>.json`, `Rosters/<id>.json`, `players.csv`,
  `competitions.csv`, spesifikasyonlar (tracking v2.2, olay v2.5), `PFF FC Change Log.docx`
  (son değişiklik 2025-06-03). 64 maç, kimlikler 3812–3859 ve 10502–10517. Toplam ≈3,7 GB.
- **Lisans:** klasörde, spesifikasyonda ve değişiklik günlüğünde lisans/koşul yok; PDF'lerde yalnız
  "PFFA Acquisition LLC proprietary information". **2026-11-03: PFF 2026-11-01 mailine yanıt verdi →
  akademik kullanım serbest, makalede kullanım için izin var.** Veri yine yeniden dağıtılmaz.
  Veri yeniden dağıtılmaz.

**Ham format (spesifikasyon v2.2 + 3840 Kamerun–Sırbistan incelemesi):**
- Sportlogiq yayın tracking'i + PFF olayları. 29,97 fps (meta `fps`), metre, orijin saha merkezi,
  x touch çizgisi boyunca, y yukarı; saha 105×68 (meta `stadium.pitches`). `homeTeamStartLeft` meta'da.
- Kare alanları: `homePlayers/awayPlayers` (ham) ve `*Smoothed` (Kalman), `balls` (ham) ve
  `ballsSmoothed` (olaylara göre: top taşıyıcıya konur, olaylar arası doğrusal). Oyuncu:
  `jerseyNum`, `confidence` (LOW/MEDIUM/HIGH), `visibility`, `x`, `y`. Top: `visibility`, `x`, `y`, `z`.
- **Kadraj dışı oyuncular tahminle doldurulmuş:** `visibility` = VISIBLE / ESTIMATED → maske kanalı
  doğrudan bundan. 3840'ta oyuncu konumlarının **%71'i ESTIMATED** (1,19 M görünür / 2,93 M tahmin).
- **Top z VAR:** 3840'ta medyan 0,06 m, p95 5,6 m, p99 11,4 m, maks. 21,8 m; min. −6,2 m (gürültü →
  kırpılmalı). Karelerin %43'ünde top yok; topun olduğu karelerin %41'inde top ESTIMATED.
- Karelerin %99,8'inde 11+11 oyuncu; %0,2'de bir takımda 12 (değişiklik anları).
- **Pas yüksekliği VAR:** olay verisinde `possessionEvents.ballHeightType`, ayrıca
  `initialTouch.initialHeightType`; `passType`, `passOutcomeType`, `bodyType` da var.
  2025-05 sürümünde topu alma anları ayrı olay (`possessionEventType = IT`); 2025-06'da
  `teamAttackingDirection` ve oyuncu `positionGroupType` eklendi (rol bilgisi için kullanılabilir).
- kloppy 3.19 okuyor: `pff.load_tracking(raw_data, meta_data, roster_meta_data, coordinates=...)`.

**64 maç bütünlük kontrolü (2026-11-03):** her tracking dosyası sonuna kadar açıldı, her satır JSON
olarak ayrıştırıldı; olay/meta/kadro dosyaları yüklendi. Hata 0, bozuk satır 0, geri giden `frameNum` 0.
Toplam 11,89 M kare (29,97 fps ≈ 110 saat video), 144,5 bin olay. Her maçta fps 29,97, saha 105×68,
karelerin çoğunda 22 oyuncu. Maç başına sayılar: `data/raw/pff/_kontrol_raporu.json`.

| Ölçüt (maç ortalaması) | Ort. | Min | Maks |
|---|---|---|---|
| Oyuncu konumu ESTIMATED (tüm kareler) | %62,6 | %48,7 (3850) | %74,8 (3841) |
| Oyuncu konumu ESTIMATED (topun olduğu kareler) | %44,8 | %30,8 | %61,3 |
| Top olan kare | %67,7 | %53,2 (3844) | %78,8 |
| Top ESTIMATED (top olan karelerde) | %38,6 | %24,4 | %52,8 |
| Top z < −0,5 m | %0,4 | %0,0 | %0,8 |

- Top z maks. 13,8–26,3 m; negatif z az ama var → [0, ~30] kırpma.
- **Uzatma eksik:** uzatmaya giden 5 maçtan 3'ünde dönem 3–4 var (10506 Japonya–Hırvatistan,
  10508 Fas–İspanya, 10517 Arjantin–Fransa). **10510 Hırvatistan–Brezilya ve 10511 Hollanda–Arjantin'de
  tracking yalnız dönem 1–2.** Penaltı atışları hiçbir maçta beklenmez.

**Sonuçları:** Ön eğitimde kayıp yalnız VISIBLE yuvalarda (maskeli kayıp); ESTIMATED oranı 3. araştırma
sorusu (gürültüye sağlamlık) için gerçek gürültü kaynağı. Top olan karelerde bile savunmacıların yaklaşık
yarısı tahmin → yalnız VISIBLE hedefle eğitilirse örnek sayısı ciddi düşer; pencere başına asgari görünür
oyuncu eşiği okuyucu planında belirlenecek.

**Okuyucu kararları (plan `docs/plans/2026-11-03-pff-okuyucu.md`, ölçümler 3840/3850/3841):**
- **Yumuşatılmış konum.** Ham izlerde karelerin %50'si önceki kareyle aynı (etkin 15 Hz); ham oyuncu hızı
  p99,9 42 m/s, yumuşatılmış 8,4 m/s. Ham top olay anında sahipten p50 2,3 m / p90 10,8 m uzak;
  yumuşatılmış top olaya sabit ama olaylar arası doğrusal (havadaki top eğrisi yok → IDSSE ile alan farkı).
- **Ölü top:** olay durum makinesi (OUT/END/VID ölü; OTB/başlama oyunda) + `home_ball` sahipliği + top var.
  Topsuz kareler ana kamera dışı: oyuncuların %100'ü ESTIMATED (3840'ta 152 parça, 45 dk).
- **Görünürlük eşiği Dataset'te.** Savunmacı görünürlüğü topa uzaklıkla düşüyor (<15 m %79–94,
  >30 m %16–43). 5 s pencerede ort. ≥0,7 görünür savunmacı: 3850 %73, 3840 %31, 3841 %14 → sert eşik
  maç/kamera tarzına göre seçer. Ön eğitimde {0; 0,5; 0,7} ablasyonu.
- kloppy'nin PFF okuyucusu kullanılmadı: görünürlüğü atıyor, dosyayı tümüyle belleğe okuyor, yinelenen
  kareleri ayıklamıyor. Forma numarası `jerseyNum` metin (str).

**İşlenmiş veri (`data/processed/pff/<id>.npz`, 2026-11-03):** `scripts/veri_isle.py --kaynak pff`
(6 süreç, maç başına ≈30 s). npz alanları IDSSE ile aynı + **`gorunur` [N,23]** (IDSSE'de = `mask`).
- 64 maç, 17 253 segment, **3567 dk oyunda (≈59,5 saat, IDSSE'nin ≈10 katı)**; maç başına 43–75 dk,
  ort. segment 12,6 s. Toplam 338 MB.
- Akıl sağlığı: oyuncu hızı p99,9 7,7–9,6 m/s (ort. 8,6); 12 m/s aşımı <0,04 %; saha dışı <0,01 %;
  eksik yuva ≈0 (PFF her oyuncuyu dolduruyor → eksiklik `gorunur`'da).
- Segment içi görünürlük (maç ort.): hücum %57 (40–69), savunma %60 (41–76), top %69 (54–80).
- Maçta görünen oyuncu sayısı 26–35. Kadroda olmayan forma uyarısı çıkmadı.
- Rol kodları: PFF `positionGroupType` (GK, LCB/MCB/RCB, LB/RB, LWB/RWB, DM/CM/AM, LM/RM, LW/RW, CF)
  `ROL_KODLARI` sonuna eklendi; IDSSE indisleri değişmedi. PFF ve DFL kodları ayrı gömme satırı →
  transfer için ortak rol sözlüğü kararı model planında.
- Mini harita: `scripts/mini_harita.py <id> --kaynak pff` (tahmin konumlar içi boş daire).

## Açık kontroller
- PFF: 35 oyunculu maçlar (6. değişiklik + olası forma kimliği hataları) incelenebilir.
- PFF: Dataset'e `gorunur` desteği (eşik, maskeli kayıp, giriş kanalı) → ön eğitim planı.
- SoccerNet-GSR: top konumu ve kapısız indirme.
