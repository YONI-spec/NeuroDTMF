#definition d'un dictionnaire des touches avec leurs frequences de groetzel respectifs associés
DTMF_CHARACTERS:dict[str, tuple[int, int]] =  {
    #les clé chaque charactères associé à son couple de frequences basses et hautes
    # keys: (l_f(hz),h_f(hz))
    '1': (697, 1209), '2': (697, 1336), '3': (697, 1477), 'A': (697, 1633),
    '4': (770, 1209), '5': (770, 1336), '6': (770, 1477), 'B': (770, 1633),
    '7': (852, 1209), '8': (852, 1336), '9': (852, 1477), 'C': (852, 1633),
    '*': (941, 1209), '0': (941, 1336), '#': (941, 1477), 'D': (941, 1633)
}


LOW_FREQS: list[int] = [697, 770, 852, 941]
HIGH_FREQS: list[int] = [1209, 1336, 1477,1633]


def get_freqs(key: str) -> tuple[int, int] :
    if key not in DTMF_CHARACTERS:
        raise ValueError(f"Invalid key '{key}'. Expected one of: {list(DTMF_CHARACTERS.keys())}")
    
    freqs = DTMF_CHARACTERS[key]
    return freqs 
        