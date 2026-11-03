# Deney notları

Pencere/kurulum tanımları: `docs/veri_notlari.md`. Tüm sayılar IDSSE, 7 kat bir-maçı-dışarıda-bırak,
ortalama ± std (katlar arası), metre, son 3 s (30 kare). Güncel birleşik tablo:
`.venv/Scripts/python scripts/sonuc_tablosu.py` (kaynak: `outputs/sonuclar/*_idsse.csv`).

## Baseline'lar — 2026-11-03

Plan: `docs/plans/2026-11-03-dataset-metrik-baseline.md` · betik `scripts/baseline_degerlendir.py`.

| Model | Kurulum | ADE | FDE |
|---|---|---|---|
| Rol ortalaması | A/B | 16,79 ± 1,23 | 16,96 ± 1,22 |
| Topa göre rol ort. (7×5 ızgara) | A/B | 12,33 ± 0,68 | 12,29 ± 0,67 |
| Sabit hız | B | 1,80 ± 0,08 | 4,50 ± 0,22 |

Tanı (test J03WMX, topa göre rol ort.): kare başına Macar eşleştirmesiyle ADE 11,7 → 9,9 m; yani yuva
sıralamasının payı ≈1,8 m. En kötü yuvalar DEF/MID sınırı (yuva 4: 15,5 m) ve forvet hattı; pencerelerin
%8'i (J03WMX) 3'lü veya 5'li defans → yuva anlamı kayıyor. Şablon eşleştirme notu geçerliliğini koruyor.

## LSTM (rol başına, etkileşimsiz) — 2026-11-03

Plan: `docs/plans/2026-11-03-lstm-baseline.md` · betik `scripts/lstm_egit.py` · modeller `outputs/modeller/lstm/`.
Çift yönlü, 2 katman, gizli 128; tek tohum; en çok 40 epoch, doğrulama ADE'sinde 6 epoch sabır.

| Model | Kurulum | ADE | FDE |
|---|---|---|---|
| Topa göre rol ort. | A/B | 12,33 ± 0,68 | 12,29 ± 0,67 |
| **LSTM** | **A** | **9,94 ± 2,00** | **11,19 ± 1,49** |
| Sabit hız | B | 1,80 ± 0,08 | 4,50 ± 0,22 |
| **LSTM** | **B** | **1,90 ± 0,12** | **3,45 ± 0,23** |

Kat başına ADE (A): J03WMX 8,75 · J03WN1 8,97 · J03WOH 8,27 · **J03WOY 13,00** · J03WPY 8,98 ·
**J03WQQ 13,15** · J03WR9 8,45. Kat başına ADE (B): 1,75–2,08.

Başarı ölçütü: A geçti (5 katta ≈3–3,5 m iyileşme). B'de FDE geçti (−1,05 m), **ADE geçmedi** (+0,10 m).

**Tanı 1 — B'de kısa ufuk (J03WOH, ufuk boyunca ADE, m):**

| Ufuk (s) | 0,1 | 0,4 | 0,7 | 1,0 | 1,3 | 1,6 | 1,9 | 2,2 | 2,5 | 2,8 |
|---|---|---|---|---|---|---|---|---|---|---|
| LSTM | 0,17 | 0,50 | 0,86 | 1,20 | 1,54 | 1,86 | 2,16 | 2,46 | 2,74 | 3,02 |
| Sabit hız | 0,02 | 0,17 | 0,43 | 0,78 | 1,21 | 1,71 | 2,27 | 2,89 | 3,55 | 4,26 |

LSTM ≈1,7 s'den sonra üstün, kısa ufukta anlık hızı yakalayamıyor (0,1 s'de 0,17 m hata). Sızıntı değil
(testler: gelecek görülmüyor). Neden: kayma son *konuma* göre; hız bilgisini normalize girdiden
(1 m ≈ 0,01) yeniden üretmek zor. Ayrıca B koşularının çoğunda en iyi epoch 37–40 → eğitim yetersiz.

**Tanı 2 — A'daki iki kötü kat = 3'lü defans (segment başına savunan takımın DEF sayısı):**
J03WOY deplasman 164/164 segment 3'lü; J03WQQ iki takım da 3'lü (ev 156/158, dep 130/145 + 2'li 15).
Diğer maçlarda 4'lü baskın. Eğitimde 4'lü çoğunlukta olduğundan 3'lü takımlarda yuvanın anlamı kayıyor
(ör. 4. yuva 4'lüde son defans oyuncusu, 3'lüde ilk orta saha oyuncusu). Model hatası değil, temsil (yuva sıralaması) sorunu;
J03WOH katındaki erken durmanın nedeni de bu (doğrulama maçı J03WOY).

## Şablon eşleştirme — 2026-11-03

Plan: `docs/plans/2026-11-03-sablon-eslestirme.md` · modül `ghostdef.data.sablon` · tanı `scripts/sablon_tani.py`
(şekil `outputs/sablon/sablonlar.png`). Yuva = kaleci + 10 şablon rolü; şablon her katta yalnız eğitim maçlarından
(kısıtlı k-ortalamalar), atama segment başına Macar. Hücum ve savunmaya ayrı şablon.

**Şablon kararlılığı:** 7 katın şablonları neredeyse üst üste (4'lü defans + 4'lü orta saha + 2 forvet şekli);
yalnız merkez orta saha rolleri (6, 7) katlar arasında birkaç metre oynuyor.

**Şekil hatası** (oyuncunun segment şeklinin atandığı rol ortalamasına uzaklığı, m; test maçı, kat şablonu):

| Test maçı | Hücum eski | Hücum şablon | Savunma eski | Savunma şablon |
|---|---|---|---|---|
| J03WMX | 10,72 | 6,93 | 10,00 | 6,11 |
| J03WN1 | 10,31 | 6,71 | 9,75 | 6,15 |
| J03WOH | 9,64 | 6,88 | 8,96 | 5,86 |
| J03WOY | 10,48 | 6,71 | 9,42 | 6,12 |
| J03WPY | 10,09 | 6,92 | 9,54 | 6,07 |
| J03WQQ | 12,81 | 7,14 | 12,00 | 6,47 |
| J03WR9 | 10,49 | 6,93 | 9,08 | 5,98 |

**Baseline'lar, şablonla** (`outputs/sonuclar/baseline_sablon_idsse.csv`):

| Model | ADE eski | ADE şablon | Kat başına ADE (şablon) |
|---|---|---|---|
| Rol ortalaması | 16,79 ± 1,23 | 14,97 ± 0,95 | 13,87–16,67 |
| Topa göre rol ort. | 12,33 ± 0,68 | **9,85 ± 0,32** | 9,23–10,24 (J03WQQ 13,88 → 9,73) |
| Sabit hız | 1,80 ± 0,08 | 1,80 ± 0,08 | yuva sırasından bağımsız (beklenen) |

Şablonla yeni A çıtası: topa göre rol ortalaması **9,85 m** (eski LSTM-A'nın 9,94'ü bu çıtanın üstünde kalıyor).

## LSTM, şablon yuvalarıyla — 2026-11-03 (güncel karşılaştırma)

Modeller `outputs/modeller/lstm_sablon/`, sonuçlar `outputs/sonuclar/lstm_sablon_idsse.csv`.
B: sabit hız tabanı, en çok 60 epoch (onaylı sapma, LSTM planına bakın).

| Model | Kurulum | ADE | FDE |
|---|---|---|---|
| Topa göre rol ort. (şablon) — çıta | A | 9,85 ± 0,32 | 9,81 ± 0,31 |
| LSTM, eski sıra | A | 9,94 ± 2,00 | 11,19 ± 1,49 |
| **LSTM, şablon** | **A** | **5,84 ± 0,38** | **6,03 ± 0,37** |
| Sabit hız — çıta | B | 1,80 ± 0,08 | 4,50 ± 0,22 |
| LSTM ilk sürüm (konum tabanı, 40 epoch), eski sıra | B | 1,90 ± 0,12 | 3,45 ± 0,23 |
| LSTM (sabit hız tabanı, 60 epoch), eski sıra | B | 1,18 ± 0,04 | 2,75 ± 0,09 |
| **LSTM (sabit hız tabanı, 60 epoch), şablon** | **B** | **1,15 ± 0,05** | **2,68 ± 0,14** |

Kat başına ADE, LSTM-A şablon: J03WMX 5,69 · J03WN1 6,07 · J03WOH 5,27 · J03WOY 6,22 · J03WPY 5,77 ·
J03WQQ 6,42 · J03WR9 5,45 (3'lü defans maçları J03WOY/J03WQQ artık diğerleriyle aynı düzeyde).
Kat başına ADE, LSTM-B şablon: 1,09–1,25.

**Yorum:**
- Şablon planının başarı ölçütleri karşılandı: 3'lü defans katları 13,0/13,2 → 6,2/6,4 m (< 10 m); 4'lü defans
  katları kötüleşmedi, ≈3 m iyileşti; katlar arası std 2,00 → 0,38. A'da erken durma artık 8–21. epoch.
- B'de şablonun etkisi küçük (−0,03 m ADE): yuva kendi geçmişini gördüğü için yuva anlamı daha az önemli.
- B'de en iyi epoch'lar hâlâ 48–59/60 → daha uzun eğitim biraz daha kazandırabilir (7. hafta ayarına not).
- **Varsayılan bundan sonra şablon yuvaları.** Transformer ve sonrası şablonla eğitilir; eski sıra yalnız
  karşılaştırma için.

**Bilinen zararsız sorun:** GPU eğitimi yapan `lstm_egit.py` süreçleri işi bitirip CSV'yi yazdıktan sonra çıkış kodu
127 ile kapanıyor (basit CUDA betiği 0 ile kapanıyor; neden bilinmiyor). Betikleri zincirlerken `&&` yerine `;` kullan.
Arka plandaki iki koşu, oturum boştayken Claude Code'un bellek baskısı korumasıyla durduruldu; `--devam` ile sürdürüldü.
