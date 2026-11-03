"""docs/literatur_kayit.json'daki makaleleri en güncelden en eskiye sıralayıp
docs/Literatur_Tablosu.xlsx dosyasının "Literatür Taraması" sayfasına yazar.

Biçim korunur: yalnız hücre değerleri, linkler ve satır yükseklikleri değişir.
Kayıt dosyası tek doğru kaynaktır; Excel'de olup kayıtta olmayan bir makale
varsa betik durur.
"""
import json
import sys
import textwrap
from pathlib import Path

import openpyxl
from openpyxl.utils import get_column_letter

KOK = Path(__file__).resolve().parent.parent
KAYIT = KOK / "docs" / "literatur_kayit.json"
TABLO = KOK / "docs" / "Literatur_Tablosu.xlsx"
SAYFA = "Literatür Taraması"
ILK_SATIR = 4
ALANLAR = ["makale", "yazarlar", "yayin_yeri", "yil", "yontemler", "metrikler", "basari", "link"]

# Arial 10 için yaklaşık değerler: sütun genişliği birimi başına karakter, satır başına punto.
KARAKTER_CARPANI = 1.2
CIZGI_YUKSEKLIGI = 12.75
EN_AZ_YUKSEKLIK = 45.0


def satir_yuksekligi(ws, satir):
    """Kaydırılan metne göre satır yüksekliğini tahmin eder (şablondaki 45 pt'nin altına inmez)."""
    en_cok_cizgi = 1
    for sutun in range(1, len(ALANLAR) + 1):
        deger = ws.cell(satir, sutun).value
        if deger is None:
            continue
        genislik = ws.column_dimensions[get_column_letter(sutun)].width
        harf = max(1, int(genislik * KARAKTER_CARPANI))
        en_cok_cizgi = max(en_cok_cizgi, len(textwrap.wrap(str(deger), harf)))
    return max(EN_AZ_YUKSEKLIK, en_cok_cizgi * CIZGI_YUKSEKLIGI + 6)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    # "durum": "yedek" olan makaleler kayıtta tutulur ama tabloya yazılmaz.
    tum_kayitlar = json.loads(KAYIT.read_text(encoding="utf-8"))
    kayitlar = [k for k in tum_kayitlar if k.get("durum") != "yedek"]
    kayitlar.sort(key=lambda k: k["tarih"], reverse=True)

    # İsteğe bağlı argüman: aynı biçimde başka bir dosyaya yaz (ör. Google Sheets'e yüklenecek kopya).
    hedef = Path(sys.argv[1]) if len(sys.argv) > 1 else TABLO
    wb = openpyxl.load_workbook(TABLO)
    ws = wb[SAYFA]
    son_satir = ws.max_row
    kapasite = son_satir - ILK_SATIR + 1
    if len(kayitlar) > kapasite:
        sys.exit(f"Tabloda {kapasite} biçimli satır var, {len(kayitlar)} kayıt sığmıyor.")

    kayittaki = {k["makale"] for k in tum_kayitlar}
    elle_eklenen = [
        ws.cell(r, 1).value
        for r in range(ILK_SATIR, son_satir + 1)
        if ws.cell(r, 1).value and ws.cell(r, 1).value not in kayittaki
    ]
    if elle_eklenen:
        sys.exit("Excel'de olup kayıtta olmayan makaleler var, önce kayda ekleyin:\n" + "\n".join(elle_eklenen))

    for i in range(kapasite):
        satir = ILK_SATIR + i
        kayit = kayitlar[i] if i < len(kayitlar) else None
        for sutun, alan in enumerate(ALANLAR, start=1):
            ws.cell(satir, sutun).value = kayit[alan] if kayit else None
        ws.cell(satir, len(ALANLAR)).hyperlink = kayit["link"] if kayit else None
        ws.row_dimensions[satir].height = satir_yuksekligi(ws, satir)

    wb.save(hedef)
    print(f"{len(kayitlar)} makale yazıldı ({kapasite - len(kayitlar)} boş satır kaldı).")


if __name__ == "__main__":
    main()
