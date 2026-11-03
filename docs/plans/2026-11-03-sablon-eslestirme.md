# Şablon eşleştirmeyle rol yuvaları — hafta 5–6 (IDSSE)

Tarih: 2026-11-03 · Durum: onaylandı, tamamlandı (2026-11-03) — sonuçlar `docs/deney_notlari.md`

## Neden
Şu anki yuva sırası (rol grubu GK→DEF→MID→FWD, grup içinde ortalama y) formasyona bağlı: 3'lü defanslı
takımda 4. yuva orta saha oyuncusuna düşüyor. Sonuç: LSTM-A'da 3'lü defans maçları J03WOY ve J03WQQ
≈13 m ADE, diğerleri ≈8,5 m (`docs/deney_notlari.md`, Tanı 2). Sorun Transformer ve difüzyonu da etkiler.

Literatürde (tablodaki makaleler): Le vd. 2017 (Data-Driven Ghosting) rol hizalaması; SoccerCPD (Kim vd. 2022)
Macar algoritmasıyla rol atama. Yaklaşım ikisindeki fikrin sadeleştirilmiş, segment düzeyindeki hâli.

## Kararlar (önerilen)
1. **Yuva = mekânsal rol, pozisyon kodu değil.** Kaleci 0. yuvada sabit (pozisyon kodundan). Kalan 10 saha
   oyuncusu, takımın segment ortalama şekline göre 10 rollük bir **şablona** Macar algoritmasıyla atanır.
   Şekil = oyuncunun segment ortalama konumu − takımın saha oyuncuları ağırlık merkezi (ölçekleme yok).
2. **Şablon öğrenme:** kısıtlı k-ortalamalar (her segmentte her rol tam bir oyuncuya): mevcut sıralamanın rol
   ortalamalarıyla başlat → ata → rol ortalamalarını güncelle → değişim bitene kadar (en çok 50 tur).
   Hücum ve savunma için **ayrı şablon** (top sahipliğinde şekil farklı).
3. **Sızıntı yok:** şablon her katta **yalnız eğitim maçlarından** öğrenilir; doğrulama/test maçları bu
   şablona atanır. Atama ucuz (≈2,2 bin segment × 2 takım × 10×10 Macar) → Dataset yüklerken yapılır;
   npz dosyaları ve veri hattı değişmez.
4. **Atama birimi segment** (pencere içinde oyuncu kimliği sabit kalsın; kişi başı hayalet için gerekli).
   Eksik oyuncu (kırmızı kart): dikdörtgen Macar ataması, boş kalan rol maskelenir.
5. **Rol gömmesi:** model girdisindeki yuva gömmesi artık şablon rolünü temsil eder; pozisyon kodu (`rol`)
   yuvayla birlikte yeniden sıralanır, ek girdi olarak şimdilik kullanılmaz.
6. **Eski sıralama korunur:** `PencereVeriSeti(..., sablon=None)` eski davranış; karşılaştırma ve yeniden
   üretilebilirlik için.

## Adımlar
1. **Modül** (`src/ghostdef/data/sablon.py`):
   - `segment_sekilleri(pos, mask, segment_id, taraf)` → [S,10,2] şekil + [S,10] geçerlilik + kaleci indisi.
   - `sablon_ogren(sekiller, gecerli)` → [10,2] şablon (kısıtlı k-ortalamalar).
   - `sablona_ata(sekiller, gecerli, sablon)` → [S,10] permütasyon (scipy `linear_sum_assignment`).
   - `Sablon` veri sınıfı: hücum + savunma şablonları; `ogren(maclar)`, `kaydet/yukle` (npz).
2. **Dataset:** `PencereVeriSeti(..., sablon=Sablon | None)`: yüklemede segment başına permütasyonu
   `pos`, `mask`, `rol`'e uygular. Kat başına: `Sablon.ogren(kat["egitim"])`.
3. **Tanı betiği** (`scripts/sablon_tani.py`):
   - Şablon çizimi (mplsoccer; hücum + savunma, 7 katın şablonları üst üste → katlar arası kararlılık)
     → `outputs/sablon/`.
   - Eski vs yeni sıralamada yuva başına "şekil hatası" (oyuncunun rol ortalamasına uzaklığı) ve 3'lü/4'lü
     defans maçlarında yuva anlamının tutarlılığı.
4. **Yeniden değerlendirme:** `baseline_degerlendir.py` ve `lstm_egit.py`'ye `--sablon` bayrağı; sonuçlar ayrı
   model adıyla (ör. "LSTM (rol başına, şablon)") CSV'ye → eski/yeni yan yana. Önce rol ortalaması baseline'ları
   (dakikalar), sonra LSTM A ve B (≈2–3 saat, arka planda).
5. **Testler** (`tests/test_sablon.py`): sentetik 4'lü ve 3'lü (kanat beki) takımlar → kanat bekleri bek
   rollerine atanır; oyuncu sırası karıştırılınca atama değişmez; eksik oyuncuda boş rol maskelenir;
   şablon öğrenme yalnız verilen maçları kullanır; `sablon=None` eski sonucu birebir verir.
6. **Kapanış:** deney notları (eski/yeni tablo + şablon şekli), veri notlarına yuva tanımı, takvim (satır 25
   + yeni satır "Şablon eşleştirme"), CLAUDE.md.

## Başarı ölçütü
- LSTM-A'da J03WOY ve J03WQQ katları belirgin iyileşir (hedef: diğer katlara yaklaşır, < 10 m) ve 4'lü maçlar
  kötüleşmez (±0,3 m). Kat ortalaması 9,94 m'nin altına iner, katlar arası std 2,00 m'den belirgin düşer.
- Topa göre rol ortalaması da iyileşmeli (yuva anlamı tutarlı hâle geldiği için).
- Hiçbiri olmazsa: yuva sorunu tek açıklama değil demektir → 3'lü defans maçlarının kendisi (az örnek) incelenir.

## Kapsam dışı
Kare başına (dinamik) rol atama, formasyon değişim noktası tespiti (SoccerCPD), takıma özel şablon,
permütasyondan bağımsız kayıp, PFF'ye uygulama (PFF okuyucusundan sonra aynı modül kullanılır).
