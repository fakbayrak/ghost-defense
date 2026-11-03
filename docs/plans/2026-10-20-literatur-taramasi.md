# Literatür Taraması Planı (2026-10-20)

## Amaç
`docs/Literatur_Tablosu.xlsx` tablosunu, projenin araştırma sorularını (AS1–AS3) ve modüllerini
kapsayan ~45 makaleyle doldurmak. Öncelik güncel çalışmalar (2023–2026); temel eserler istisna.

## Gruplar
| # | Grup | Kapsam | Durum |
|---|---|---|---|
| 1 | Hayalet savunma | Ghosting, karşı-olgusal savunma, futbolda üretken taktik modelleri | Tamamlandı (9 makale) |
| 2 | Üretken yörünge modelleri (AS1) | Difüzyon, VAE, flow matching; spor çok-ajanlı tahmin; deterministik baseline'lar | Tamamlandı (5 makale + 5 yedek) |
| 3 | Ön eğitim ve aktarım (AS2) | Ön eğitim, alan uyarlama, az veriyle ince ayar, veri setleri arası genelleme | Tamamlandı (5 makale + 2 yedek) |
| 4 | Görüntü işleme | Yayın videosundan oyun durumu: GSR, saha kalibrasyonu, tespit/takip, forma/takım tanıma, top | Tamamlandı (5 makale + 1 yedek) |
| 5 | Veri ve sağlamlık (AS3 + veri hattı) | Kadraj dışı oyuncu tahmini, IDSSE veri makalesi, rol/diziliş atama, gürültüye dayanıklı yörünge tahmini, tracking doğruluğu | Tamamlandı (5 makale + 3 yedek) |
| 6 | Pas isabet modeli (ek hedef) | Pas olasılığı, pitch control, EPV, StatsBomb 360 — pas modeline başlarken (11. hafta) yapılacak | Ertelendi |

Not (2026-10-20): Eski 5. ve 6. gruplar birleştirildi; veri seti makalelerinin (PFF, Metrica,
StatsBomb) çoğu tabloya değil raporun "Veri" bölümüne atıf olarak girer.

## Akış (her grup için)
1. Araştır; künye ve sayıları makalenin kendisinden doğrula, emin olunamayanı "doğrulanmadı" işaretle.
2. Her makale için özet, projeye katkı, zayıf yön ve öneri sun; satır taslaklarını göster.
3. Onay gelince `docs/literatur_kayit.json`'a ekle ve `python -I scripts/literatur_tablosu_guncelle.py` çalıştır.

## Tablo kuralları
- Grup başına 5 makale (1. grup istisna: 9). Elenenler kayıtta `"durum": "yedek"` ile tutulur,
  tabloya yazılmaz; rapor/makale kaynakçasında kullanılabilir.
- Sıralama: en güncelden en eskiye (kayıttaki `tarih` alanına göre).
- Yazarlar: "Soyad, A., Soyad, B., Soyad, C. vd." (Açıklamalar sayfasındaki örnek gibi).
- Yöntem ve metrik sütunları kısa: yalnız yöntem ve metrik adları.
- Başarı oranı: metrik + değer + veri seti birlikte.
- Link: DOI veya arXiv tercih edilir.
- Satır yüksekliği metne göre betik tarafından ayarlanır.
- Tabloda 50 biçimli satır var; aşılırsa biçim genişletilir.
