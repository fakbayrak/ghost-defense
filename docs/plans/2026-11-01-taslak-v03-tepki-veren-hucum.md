# Taslak v0.3: Danışman onayı ve tepki veren hücum (2026-11-01)

## Bağlam
Danışman, literatür sunumunda konuşulan planı onayladı ve hücum oyuncularının hayalet savunmaya
dinamik tepki vermesinin eklenmesini istedi.

## Kararlar
- **Kademe 1 – Ortak üretim (top yolu sabit):** Difüzyon modeli savunmayı ve top dışındaki hücum
  oyuncularını birlikte üretir; top yolu ve topa sahip oyuncu gerçek veriden sabit tutulur.
  **Yeni Ek hedef 1** (ek hedeflerin başına alındı).
- **Kademe 2 – Kayan ufuklu kapalı döngü simülasyon:** opsiyonel, zaman kalırsa.
- **Kademe 3 – Top da tepki verir (tam simülatör):** opsiyonel / gelecek çalışma.
- Yeni ek hedef sırası: 1) Tepki veren hücum, 2) Oyuncu gözü 3D, 3) Pas isabet modeli,
  4) Yayın görüntüsünden giriş, 5) Korner hayalet savunması. Kırpma sondan başlar.

## Taslağa işlenecekler (v0.3)
1. Kapak: sürüm 0.3, durum "danışman onayı alındı".
2. Özet: tepki veren hücum cümlesi.
3. 6. Kapsam tablosu ve 6.x bölümleri: yeni ek hedef sırası, 6.6 Tepki veren hücum (kademeler +
   değerlendirme), 6.10 Korner hayalet savunması; diğer "Ek hedef" numaraları güncellenir
   (AS3 metni, risk tablosu dahil).
4. 6.2: kademeli karşılaştırma (Transformer → CVAE → difüzyon).
5. 6.3: baseline'lara rol ortalaması ve topa göre rol ortalaması; sabit hızın yalnız tahmin
   kurulumunda geçerli olduğu.
6. 7. bölüm: "Kapsam Dışı Bırakılanlar, Varsayımlar ve Sınırlılıklar" — 7.1 kapsam dışı (mevcut),
   7.2 varsayımlar ve sınırlılıklar, 7.3 gelecek çalışma.
7. Metrik tablosu: tepki veren hücum için tutarlılık / gerçekçilik / duyarlılık satırı.
8. Takvim: 11–14. haftalar yeni ek hedef sırasına göre.
9. Risk tablosu: tepki veren hücumun değerlendirilmesi riski.
10. Kaynaklar: TacEleven (Zhao vd., 2025).

v0.2 `docs/arsiv/` altına saklanır.
