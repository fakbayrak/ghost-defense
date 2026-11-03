# Proje devri: Futbol Yayınları için Derin Öğrenme Tabanlı Hayalet Savunma Sistemi

## Bağlam
Bilgisayar Mühendisliği 4. sınıf öğrencisiyim, Derin Öğrenme dersi için ~16 haftalık
(Ekim 2026 – Şubat 2027) bir dönem projesi yapıyorum. Danışmanımla kapsamı YALNIZCA
FUTBOL olarak belirledik (F1 ve basketbol çıkarıldı). Hedef üst düzey bir proje;
sonuçlar iyi olursa dergiye/çalıştaya makale olarak gönderebiliriz. Benimle Türkçe konuş.

## Klasördeki belgeler (önce bunları oku)
- docs/Proje_Taslagi_Hayalet_Savunma.docx → proje konusu ve kapsam taslağı (v0.1).
  Okumak için: `pandoc -t markdown docs/Proje_Taslagi_Hayalet_Savunma.docx`
- docs/Literatur_Tablosu.xlsx → boş literatür karşılaştırma tablosu.
  "Literatür Taraması" sayfası: başlıklar 3. satırda, veri 4. satırdan başlar.
  Sütunlar: Makale İsmi | Yazarların İsmi | Makale Nerede Yayınlandı? | Yayın Yılı |
  Kullanılan Yöntemler | Kullanılan Metrikler | Başarı Oranı | Makale Linki
  "Açıklamalar" sayfası doldurma kurallarını ve bir örnek satırı içerir; onu değiştirme.

## Projenin özü
Hücum oyuncularının ve topun hareketinden, savunma oyuncularının nerede olmasının
BEKLENDİĞİNİ üreten bir "hayalet savunma" (ghost defense) modeli. Gerçek ve hayalet
konum farkı yayın grafiğine dönüştürülür. Model ideal değil TİPİK savunmayı öğrenir;
bu ayrım her yerde korunmalı.

Araştırma soruları:
1. Üretken model (difüzyon / koşullu VAE), deterministik Transformer'dan daha gerçekçi
   hayalet üretir mi? (deterministik model ortalamaya çöker)
2. Transfer: büyük ama gürültülü yayın tracking verisiyle (PFF, 64 maç) ön eğitim +
   küçük temiz veride (IDSSE, 7 maç) ince ayar işe yarıyor mu? İnce ayar 1/3/6 maçla
   tekrarlanacak.
3. Sağlamlık: videodan çıkarılan gürültülü konumlarla model ne kadar kötüleşiyor?

## Kapsam (öncelik sırasıyla; zaman daralırsa sondan kırpılır)
Çekirdek: 1) veri hattı 2) hayalet savunma modeli (deterministik baseline + üretken ana
model) 3) değerlendirme ve model karşılaştırma 4) transfer deneyi (MUTLAKA yapılacak)
5) 2D mini harita görselleştirme
Ek hedefler: 6) sadeleştirilmiş oyuncu gözü 3D görünüm (Three.js) 7) pas isabet modeli
(StatsBomb 360 + PFF) 8) yayın görüntüsünden giriş (SoccerNet sn-gamestate hazır kodu)
Kapsam dışı: fotogerçekçi görüntü, takıma özel taktik model, RL ile "optimal" savunma,
video modellerini sıfırdan eğitmek, gerçek zamanlı sistem, telifli yayın görüntüsü.

## Veri kaynakları
- PFF FC / Gradient Sports – 2022 Dünya Kupası: 64 maç yayın tracking (~30 Hz) + olay
  verisi; kayıt formuyla ücretsiz. Kadraj dışı oyuncular tahminle doldurulmuş olabilir,
  spesifikasyondan doğrulanmalı; maske kanalı tutulacak.
- IDSSE (Bassek vd., 2025, Scientific Data): 7 Bundesliga maçı, resmi DFL çok kameralı,
  25 Hz, CC-BY 4.0. floodlight veya kloppy ile yüklenir. Nihai test verisi.
- Metrica Sports örnek verisi (GitHub), StatsBomb Open Data (WC 2022 + 360).
- SoccerNet-GSR (2024/2025): Hugging Face'te kapısız, GPL-3.0; NDA GEREKMİYOR (01.11.2026 doğrulandı).
  200 klip × 30 s, 25 fps; etiketlerde top var (saha konumu zemine izdüşüm, z yok).
Veriyi yeniden dağıtma, ham verileri git'e ekleme.

## Teknik kurallar
- Eğitim/test bölmesi KARELERE GÖRE DEĞİL MAÇLARA GÖRE (IDSSE'de bir-maçı-dışarıda-bırak).
- Saha 105×68 m, modele [0,1] normalize; hücum yönü hep soldan sağa; ölü toplar ayıklanır;
  ortak örnekleme (~10 Hz); değişken oyuncu sayısı için maskeleme; savunmacı sıralama
  (permütasyon) problemi rol bazlı sabit sırayla çözülecek.
- Metrikler: ADE, FDE, minADE/minFDE; pas modeli için log-loss, Brier, ROC-AUC, ECE.
- Baseline'lar: sabit hız, LSTM, deterministik Transformer.
- Donanım: RTX 5070 Laptop (8 GB VRAM), 24 GB RAM, Windows. Ortam: proje içi `.venv`
  (torch 2.11+cu128, kloppy 3.19, mplsoccer); paket `pip install -e .` (src/ghostdef). floodlight kurulu
  değil (pandas<3 ister). Testler: `.venv/Scripts/python -m pytest -q`. Büyük JSON'ları toplu yükleme;
  maç başına NumPy/Parquet'e işle.
- Yığın: Python, PyTorch, kloppy, floodlight, statsbombpy, mplsoccer, Three.js.
- Ürettiğin tüm belgelerde Türkçe karakterler (ö, ü, ş, ç, ğ, ı, İ) doğru görünmeli.

## Tamamlananlar (Kasım 2026 başı)
- Literatür taraması bitti: tabloda 21 makale (en yeniden en eskiye), yedekte 18.
  Tek doğru kaynak `docs/literatur_kayit.json`; tabloyu `python -I scripts/literatur_tablosu_guncelle.py`
  yazar (biçimi korur). Yeni makale yalnız onayımla eklenir; uydurma bilgi yok,
  emin olunmayan "doğrulanmadı" diye işaretlenir. Pas modeli makaleleri 11. haftada eklenecek.
- Taslak v0.3 danışman onaylı; güncel dosya v0.4 (`docs/Proje_Taslagi_Hayalet_Savunma.docx`,
  yalnız SoccerNet NDA düzeltmesi). Eski sürümler `docs/arsiv/`.
- Danışmanın isteğiyle yeni Ek hedef 1: tepki veren hücum (difüzyonla savunma + top dışı hücum
  ortak üretimi, top yolu sabit). Ek hedef sırası: 1) tepki veren hücum, 2) oyuncu gözü 3D,
  3) pas modeli, 4) video girişi, 5) korner. Kapalı döngü simülasyon ve tam simülatör opsiyonel.
- Model zinciri: sabit hız / rol ortalaması → LSTM → Transformer → CVAE → difüzyon.
- Planlar: `docs/plans/`. Ayrıntılı kararlar proje belleğinde.
- Haftalık takvim ve ilerleme: `docs/Proje_Takvimi.xlsx` (1. hafta = 5 Ekim 2026, 16 hafta kesintisiz).
  Tek doğru kaynak bu dosya; iş bitince Durum + Tamamlanma tarihi elle/openpyxl ile işlenir.

## Şu anki adım: veri hattı (hafta 1–3)
Plan: `docs/plans/2026-11-01-veri-hatti.md`. Bulgular ve npz alanları: `docs/veri_notlari.md`.
- Bitti (2026-11-01): IDSSE 7 maç indirildi (`scripts/veri_idsse_indir.py`) ve ortak formata
  işlendi (`scripts/veri_isle.py` → `data/processed/idsse/*.npz`, 2158 segment, 352 dk).
  IDSSE'de top z VAR. İlk mini harita: `scripts/mini_harita.py <maç>` → `outputs/mini_harita/`.
- Bitti (2026-11-03): Dataset + 7 kat bölme + metrikler + rol ortalaması / topa göre / sabit hız
  baseline'ları (plan `docs/plans/2026-11-03-dataset-metrik-baseline.md`). Pencere 5 s (2+3),
  kurulum A = hayalet (savunma verilmez), B = tahmin (savunmanın 2 s geçmişi). İlk sayılar veri notlarında.
- Bitti (2026-11-03): PFF 64 maç PFF'nin herkese açık Drive klasöründen indirildi
  (`scripts/veri_pff_indir.py` → `data/raw/pff/`, 3,7 GB). Top z ve pas yüksekliği alanı VAR; oyuncu
  konumlarının çoğu ESTIMATED (maske kanalı `visibility`'den). Ayrıntı veri notlarında.
- Bitti (2026-11-03): PFF okuyucusu (`ghostdef.data.pff`, plan `docs/plans/2026-11-03-pff-okuyucu.md`)
  → `veri_isle.py --kaynak pff` → `data/processed/pff/*.npz` (64 maç, 3567 dk ≈ IDSSE×10). Yumuşatılmış
  konum; npz'de yeni `gorunur` [N,23] maskesi; görünürlük eşiği Dataset'te (ön eğitimde ablasyon).
- Bitti (2026-11-03): LSTM baseline + eğitim altyapısı (`ghostdef.egitim`, `scripts/lstm_egit.py`,
  `scripts/sonuc_tablosu.py`) ve plan dışı **şablon eşleştirme** (`ghostdef.data.sablon`; yuva = kaleci + 10
  mekânsal rol, şablon kat başına yalnız eğitim maçlarından). Şablon artık varsayılan (`--sablon`).
  ADE (son 3 s): A topa göre rol ort. 9,85 → LSTM 5,84; B sabit hız 1,80 → LSTM 1,15. Ayrıntı `docs/deney_notlari.md`.
  Not: GPU eğitim süreçleri bitince çıkış kodu 127 veriyor (sonuç etkilenmiyor) → zincirde `;` kullan.
- Sırada: Transformer (6. hafta, şablon yuvalarıyla, aynı eğitim altyapısı). PFF ön eğitimi için Dataset'e
  `gorunur` (şablon PFF'ye de uygulanacak). StatsBomb 360 → 11. hafta; Metrica = PFF'de sorun çıkarsa B planı.
  GSR kontrolü bitti (NDA yok, top etiketli) → taslak v0.4'te düzeltildi (v0.3 arşivde).

## Bekleyen işler
- Taslak belgedeki [Ad Soyad] ve [Danışman] yer tutucuları doldurulacak.