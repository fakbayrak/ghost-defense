"""Maçlara göre bölme (karelere göre değil): bir-maçı-dışarıda-bırak."""


def mac_disari_katlar(maclar) -> list:
    """Her maç bir kez test olur; listede ondan sonraki maç (döngüsel) doğrulama, kalanlar eğitim."""
    maclar = list(maclar)
    if len(maclar) < 3:
        raise ValueError("En az 3 maç gerekli (eğitim, doğrulama, test)")
    katlar = []
    for i, test in enumerate(maclar):
        dogrulama = maclar[(i + 1) % len(maclar)]
        katlar.append({
            "egitim": [m for m in maclar if m not in (test, dogrulama)],
            "dogrulama": [dogrulama],
            "test": [test],
        })
    return katlar
