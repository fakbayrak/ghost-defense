# LSTM baseline + eğitim altyapısı — hafta 5 (IDSSE)

Tarih: 2026-11-03 · Durum: onaylandı, tamamlandı (2026-11-03) — sonuçlar `docs/deney_notlari.md`

**Onaylı sapma (2026-11-03, ilk koşudan sonra):** B'de çıktı tabanı son konum yerine sabit hız tahmini
(`RolLSTM(taban="sabit_hiz")`), B en çok 60 epoch. Neden: ilk sürüm kısa ufukta sabit hızın gerisinde kaldı
(ADE 1,90 vs 1,80) ve en iyi epoch'lar 37–40'tı. İlk sürüm sonuçları `docs/deney_notlari.md`'de.
Yuva sıralaması sorunu ayrı plan: `docs/plans/2026-11-03-sablon-eslestirme.md`.

Takvim satırı 25: "LSTM baseline'ı (rol başına, etkileşimsiz)". Bu planın eğitim altyapısı
(eğitim döngüsü, kayıt, sonuç tablosu) 6. haftada Transformer'da aynen kullanılacak.

## Çıta (önceki plan, 7 kat, son 3 s)
- A: topa göre rol ortalaması ADE 12,33 m.
- B: sabit hız ADE 1,80 m, FDE 4,50 m.

## Kararlar (önerilen)
1. **"Rol başına, etkileşimsiz" tanımı:** her savunma yuvası ayrı bir dizi olarak, **ortak ağırlıklı** tek bir LSTM'den
   geçer. Savunmacılar birbirini görmez (etkileşim Transformer'ın katkısı olarak ölçülecek); bağlam olarak
   hücum + top verilir.
2. **Tek mimari, iki kurulum.** Kare başına girdi (yuva başına):
   - top (x, y, z/10), hücum 11×(x, y) + 11 maske biti, yuva gömmesi (11 → 16 boyut);
   - yalnız B'de: yuvanın kendi konumu (ilk 20 kare; sonrası 0) + "gözlendi" biti.
   Konumlar [0,1] normalize. **Çift yönlü LSTM** (2 katman, gizli 128): hayalet tüm pencerenin hücum/top
   hareketine koşulludur, gelecek dahil (iki kurulumda da bu bilgi zaten veriliyor).
3. **Çıktı:** A'da kare başına mutlak konum; B'de son gözlenen konuma göre kayma (residual) →
   B'de model sabit konumdan başlar, sabit hızı geçmesi için kolaylık. Yalnız tek bir tahmin (deterministik).
4. **Kayıp:** maskeli ortalama Öklid uzaklığı, metre cinsinden (metrikle aynı ölçek).
   A'da tüm 50 kare, B'de son 30 kare.
5. **Eğitim:** AdamW (lr 1e-3, weight decay 1e-4), kosinüs azaltma, batch 256 pencere, en çok 40 epoch,
   doğrulama maçı ADE'sinde (son 3 s) 6 epoch erken durdurma; en iyi epoch kaydedilir. Tek tohum (seed 0);
   tohum varyansı 7. hafta ayarında. Hiperparametre araması yok (sabit ayar, belgelenir).
6. **Veri GPU'da:** tüm pencereler `yigin()` ile bir kez GPU'ya alınır (≈40 MB), batch'ler indisle kesilir;
   DataLoader işçisi yok (Windows çoklu süreç sorunlarından kaçınmak için).

## Adımlar
1. **Model** (`src/ghostdef/modeller/lstm.py`): `RolLSTM(kurulum, gizli=128, katman=2)`;
   `forward(yigin) → [B,1,50,11,2]` metre (metrik/baseline arayüzüyle aynı şekil).
   Girdi hazırlığı ayrı fonksiyon (`girdi_ozellikleri`), Transformer'da tekrar kullanılabilsin.
2. **Eğitim döngüsü** (`src/ghostdef/egitim.py`): `egit(model, egitim_ds, dogrulama_ds, ayar) → en iyi model + epoch günlüğü`;
   maskeli kayıp, erken durdurma, kontrol noktası `outputs/modeller/<model>/<kurulum>/<test_mac>.pt`,
   epoch günlüğü CSV.
3. **Betik** (`scripts/lstm_egit.py`): `--kurulum A|B|hepsi`, `--kat <maç>|hepsi`, `--epoch`.
   7 kat × 2 kurulum → `outputs/sonuclar/lstm_idsse.csv` (baseline CSV'siyle aynı sütunlar).
4. **Sonuç tablosu** (`scripts/sonuc_tablosu.py`): `outputs/sonuclar/*_idsse.csv` dosyalarını birleştirip
   Markdown özet (ortalama ± std) + kat başına ADE tablosu basar. Sonraki modeller de buraya düşer.
5. **Testler** (`tests/test_lstm.py`): A/B çıktı şekli; A'da model savunma konumunu görmüyor
   (savunma değiştirilince çıktı değişmez); B'de gelecek savunma konumunu görmüyor; maskeli yuva kayba
   girmiyor; küçük sentetik veride birkaç yüz adımda kayıp belirgin düşüyor (CPU, aşırı öğrenme testi);
   kontrol noktası kaydet/yükle aynı çıktıyı veriyor.
6. **Kapanış:** model sonuçları yeni `docs/deney_notlari.md`'ye (veri notlarından ayrı; baseline tablosu
   da oraya taşınır, veri notlarında yalnız pencere/kurulum tanımı kalır); takvim satırı 25;
   CLAUDE.md "Şu anki adım".

## Başarı ölçütü
- A: LSTM ADE < 12,33 m (topa göre rol ortalaması). B: LSTM ADE < 1,80 m ve FDE < 4,50 m (sabit hız).
- B'de sabit hızı geçemezse: önce sızıntı/maske/residual hatası aranır; sonra raporlanır (bu da bir bulgu).
- 7 kat × 2 kurulum eğitimi RTX 5070'te makul sürede biter (tahmin: toplam < 1 saat).

## Kapsam dışı
Transformer, savunmacılar arası etkileşim, üretken modeller, veri artırma (y aynalama rol eşlemesi ister),
hiperparametre araması, çoklu tohum.
