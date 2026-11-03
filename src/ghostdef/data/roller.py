"""Pozisyon kodları → rol grubu. Rol sıralaması (permütasyon çözümü) bu gruplara dayanır.

Grup sırası: 0 kaleci, 1 defans, 2 orta saha, 3 forvet.
PFF kodları (kadro `positionGroupType`): PFF_GRUP'a bakın.
DFL kodları (IDSSE meta XML, PlayingPosition):
  TW kaleci · IV* stoper · LV/RV bek · D*M/DM* defansif orta · ZO 10 numara
  OLM/ORM kanat orta · LA/RA kanat · ST* forvet
"""
import warnings

KALECI, DEFANS, ORTA, FORVET = 0, 1, 2, 3
GRUP_ADLARI = ["GK", "DEF", "MID", "FWD"]

DFL_GRUP = {
    "TW": KALECI,
    "IVL": DEFANS, "IVR": DEFANS, "IVZ": DEFANS, "LV": DEFANS, "RV": DEFANS,
    "DML": ORTA, "DMR": ORTA, "DMZ": ORTA, "DLM": ORTA, "DRM": ORTA,
    "ZD": ORTA, "ZO": ORTA, "OLM": ORTA, "ORM": ORTA, "LM": ORTA, "RM": ORTA,
    # HL/HR: muhtemelen "Halblinks/Halbrechts" (iç orta saha) — DFL belgesinden doğrulanmadı;
    # IDSSE'de ortalama hücum x'i 53,8 m (orta saha bandı) olduğu için ORTA'ya atandı.
    "HL": ORTA, "HR": ORTA,
    "LA": FORVET, "RA": FORVET, "STL": FORVET, "STR": FORVET, "STZ": FORVET, "HST": FORVET,
}

# PFF kadro dosyası (positionGroupType). LM/RM DFL'de de var, ikisinde de orta saha.
PFF_GRUP = {
    "GK": KALECI,
    "LCB": DEFANS, "MCB": DEFANS, "RCB": DEFANS, "LB": DEFANS, "RB": DEFANS,
    "LWB": DEFANS, "RWB": DEFANS,
    "DM": ORTA, "CM": ORTA, "AM": ORTA, "LM": ORTA, "RM": ORTA,
    "LW": FORVET, "RW": FORVET, "CF": FORVET,
}
KOD_GRUP = {**DFL_GRUP, **PFF_GRUP}

# Modele gömme (embedding) için sabit kod sözlüğü; 0 = boş yuva / bilinmiyor.
# PFF kodları sona eklenir → IDSSE npz'lerindeki rol indisleri değişmez.
ROL_KODLARI = ["?"] + sorted(DFL_GRUP) + sorted(set(PFF_GRUP) - set(DFL_GRUP))


def grup(kod: str) -> int:
    """Bilinmeyen kodu orta sahaya atar ve uyarır (sessizce yutmaz)."""
    if kod in KOD_GRUP:
        return KOD_GRUP[kod]
    warnings.warn(f"Bilinmeyen pozisyon kodu {kod!r} → orta saha varsayıldı")
    return ORTA


def kod_id(kod: str) -> int:
    return ROL_KODLARI.index(kod) if kod in ROL_KODLARI else 0
