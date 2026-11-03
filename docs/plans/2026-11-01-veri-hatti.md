# Veri hattı planı — hafta 1 (IDSSE)

Tarih: 2026-11-01 · Durum: onaylandı

## Ön bulgular
- PyTorch yalnız CPU (`2.13.0+cpu`); RTX 5070 için CUDA (cu128+) build gerekli.
- GPU: RTX 5070 Laptop, **8 GB** VRAM (CLAUDE.md'de 12 GB yazıyordu → düzeltilecek).
- kloppy, floodlight, mplsoccer, pyarrow kurulu değildi. Disk: 93 GB boş.

## Adımlar
0. **Ortam:** proje içinde `.venv`; CUDA'lı torch, kloppy 3.19, mplsoccer, pytest.
   floodlight bağımlılıkları çakışırsa yalnız kloppy (IDSSE = sportec açık veri; PFF de destekleniyor).
1. **İndir + keşif:** önce 1 maç. Kare hızı, koordinat sistemi, saha ölçüleri, top z,
   ölü top / sahiplik alanları, oyuncu pozisyon etiketleri, oyuncu değişiklikleri, eksik kareler
   → `docs/veri_notlari.md`. Sonra 7 maç. Ham veri `data/raw/` (`.gitignore`).
2. **Ortak format** (`src/ghostdef/data/`) → `data/processed/idsse/<maç>.npz`:
   - `pos` [T, 23, 2] = 11 hücum + 11 savunma + top; metre, 105×68, orijin sol alt.
     [0,1] normalizasyon modelin girişinde.
   - `ball_z` (varsa), `mask` [T, 23], `role_id`, `segment_id`, `period`, `t`.
   - Hücum yönü: top sahibi takım hep soldan sağa (aynalama).
   - 10 Hz: ortak zaman ızgarasına doğrusal ara değer; kategorik alanlar en yakın kare.
   - Ölü top: oyunda + aynı sahiplik kesintisiz segmentler; <2 s atılır; pencere kesilmez.
   - Rol sırası: rol grubu (GK→DEF→MID→FWD), grup içinde segment ortalama y.
     Şablon eşleştirme (Bialkowski / Macar) not olarak ileriye.
   - Maske: eksik oyuncu (kırmızı kart, veri yok) yuvası 0.
3. **Doğrulama** (`tests/`): aynalama, yeniden örnekleme, maske, rol sırası birim testleri;
   maç başına akıl sağlığı (hız ≤12 m/s, saha içi, segment sayısı/süresi).
4. **Mini harita** (`scripts/mini_harita.py`): mplsoccer ile 15–20 s segment MP4/GIF;
   hücum/savunma/top renkli, hayalet katmanı için yer hazır.
5. **Kapanış:** CLAUDE.md GPU düzeltmesi, memory'ye veri bulguları.

## Kapsam dışı (bu tur)
PFF (form bekliyor), StatsBomb, Metrica, model kodu.

## Paralel iş (kullanıcı)
PFF erişim formu bu hafta gönderilecek; koşullar birlikte okunacak
(yayın/şekil izni, türetilmiş veri paylaşımı, akademik kullanım).
