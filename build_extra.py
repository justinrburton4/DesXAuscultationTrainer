"""Add extra clips from CirCor DigiScope (heart) and ICBHI 2017 (lung) to sounds/ and sounds.json.

Usage: python -I build_extra.py <CirCor training_data dir> <ICBHI wav/txt dir>
Run build_sounds.py first. Safe to re-run: earlier extra clips (those with a "src" field) are replaced.

Heart sites: the four CirCor valve sites are mapped onto the nearest existing site
  AV -> RUSB, PV -> LUSB, TV -> LLSB, MV -> A.
Lung sites: ICBHI only records anterior left/right, with no upper/mid/lower zone, so each clip's zone is
  assigned round-robin within its side. Treat the zone as arbitrary.
"""
import csv, json, sys, wave
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

RATE = 4000
here = Path(__file__).parent
out = here / "sounds"

# ---------- helpers ----------
def read_wav(path):
    sr, x = wavfile.read(str(path))          # handles 8/16/24/32-bit PCM and float
    if x.dtype == np.uint8: x = (x.astype(np.float64) - 128) / 128
    elif np.issubdtype(x.dtype, np.integer): x = x.astype(np.float64) / np.iinfo(x.dtype).max
    else: x = x.astype(np.float64)
    if x.ndim > 1: x = x.mean(axis=1)
    return sr, x

def to_rate(x, sr):
    if sr == RATE:
        return x
    g = np.gcd(sr, RATE)
    return resample_poly(x, RATE // g, sr // g)

def noise_score(seg):
    """Lower is cleaner: share of energy outside the 20-800 Hz band, plus spikiness (handling noise)."""
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / RATE)
    tot = spec.sum() or 1
    outside = spec[(f < 20) | (f > 800)].sum() / tot
    rms = np.sqrt(np.mean(seg ** 2)) or 1e-9
    return outside + 0.05 * (np.abs(seg).max() / rms)

def write_wav(path, seg):
    seg = seg - seg.mean()
    seg = seg / (np.abs(seg).max() or 1) * 0.89          # -1 dBFS, same as normalize_sounds.py
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes(np.round(seg * 32767).astype("<i2").tobytes())

def edge_fade(seg, ms=8):
    n = int(RATE * ms / 1000)
    seg = seg.copy()
    seg[:n] *= np.linspace(0, 1, n); seg[-n:] *= np.linspace(1, 0, n)
    return seg

# ---------- heart: CirCor ----------
HEART_SITE = {"AV": "RUSB", "PV": "LUSB", "TV": "LLSB", "MV": "A"}
HEART_COND = {  # (systolic timing) -> (code, label)
    "Early-systolic": ("ESM", "Early systolic murmur"),
    "Mid-systolic": ("MSM", "Mid systolic murmur"),
    "Holosystolic": ("HSM", "Holosystolic murmur"),
}
GRADE_MIN = {"I/VI": 1, "II/VI": 2, "III/VI": 3, "IV/VI": 4}

def s1_windows(tsv, total, lo=6.0, hi=11.0):
    """Candidate loop windows that start at S1 and end at the next S1, fully annotated (no state 0)."""
    rows = [(float(a), float(b), int(c)) for a, b, c in (l.split() for l in open(tsv) if l.strip())]
    starts = [a for (a, b, s), prev in zip(rows[1:], rows[:-1]) if s == 1 and prev[2] == 4]
    state0 = [(a, b) for a, b, s in rows if s == 0]
    wins = []
    for i, a in enumerate(starts):
        for b in starts[i + 2:]:
            if b - a > hi: break
            if b - a >= lo and b <= total and not any(x < b and y > a for x, y in state0):
                wins.append((a, b))
    return wins

def best_window(x, tsv):
    wins = s1_windows(tsv, len(x) / RATE)
    if not wins: return None
    scored = [(noise_score(x[int(a * RATE):int(b * RATE)]), a, b) for a, b in wins]
    s, a, b = min(scored)
    return s, a, b

def heart_clips(d):
    rows = list(csv.DictReader(open(d / "training_data.csv")))
    cands = defaultdict(list)           # (code, site, gender) -> [(score, patient, wavpath, a, b, label)]
    for r in rows:
        pid, sex = r["Patient ID"], {"Male": "M", "Female": "F"}.get(r["Sex"])
        if not sex or r["Diastolic murmur timing"] != "nan": continue
        if r["Murmur"] == "Absent" and r["Outcome"] == "Normal":
            code, label, locs = "N", "Normal", set(HEART_SITE)
        elif r["Murmur"] == "Present" and r["Systolic murmur timing"] in HEART_COND:
            if GRADE_MIN.get(r["Systolic murmur grading"], 0) < 2: continue   # grade I is barely audible
            code, label = HEART_COND[r["Systolic murmur timing"]]
            locs = set(r["Murmur locations"].split("+")) & set(HEART_SITE)
        else:
            continue
        for loc in locs:
            wav = d / "training_data" / f"{pid}_{loc}.wav"
            tsv = wav.with_suffix(".tsv")
            if not wav.exists() or not tsv.exists(): continue
            sr, x = read_wav(wav); x = to_rate(x, sr)
            bw = best_window(x, tsv)
            if bw: cands[(code, HEART_SITE[loc], sex)].append((bw[0], pid, wav, bw[1], bw[2], label))
    return cands

# ---------- lung: ICBHI ----------
LUNG_SIDE = {"Al": "L", "Ar": "R"}
ZONES = ["U", "M", "L"]
LUNG_LABEL = {"N": "Normal", "C": "Crackles", "W": "Wheezing", "CW": "Crackles and wheezes"}

def lung_label(cycles):
    n = len(cycles)
    c = sum(1 for _, _, cr, wh in cycles if cr) / n
    w = sum(1 for _, _, cr, wh in cycles if wh) / n
    if c == 0 and w == 0: return "N"
    if c >= .5 and w >= .5: return "CW"
    if c >= .5 and w == 0: return "C"
    if w >= .5 and c == 0: return "W"
    return None                          # too mixed to label honestly

def lung_clips(d):
    cands = defaultdict(list)           # (code, side) -> [(score, patient, wavpath, a, b)]
    for txt in sorted(d.glob("*.txt")):
        parts = txt.stem.split("_")
        if len(parts) < 5 or parts[2] not in LUNG_SIDE: continue
        cycles = [(float(a), float(b), int(c), int(w)) for a, b, c, w in (l.split() for l in open(txt) if l.strip())]
        if len(cycles) < 4: continue
        code = lung_label(cycles)
        if not code: continue
        sr, x = read_wav(txt.with_suffix(".wav")); x = to_rate(x, sr)
        best = None
        for i in range(len(cycles)):
            a = cycles[i][0]
            j = i
            while j + 1 < len(cycles) and cycles[j][1] - a < 8: j += 1
            b = cycles[j][1]
            if b - a < 5 or b * RATE > len(x): continue
            if lung_label(cycles[i:j + 1]) != code: continue
            s = noise_score(x[int(a * RATE):int(b * RATE)])
            if best is None or s < best[0]: best = (s, a, b)
        if best: cands[(code, LUNG_SIDE[parts[2]])].append((best[0], parts[0], txt.with_suffix(".wav"), best[1], best[2]))
    return cands

# ---------- main ----------
def main(circor, icbhi, per_heart=2, per_lung=6):
    out.mkdir(exist_ok=True)
    meta = json.loads((here / "sounds.json").read_text())
    meta["clips"] = [c for c in meta["clips"] if "src" not in c]
    for f in out.glob("X_*.wav"): f.unlink()

    n = 0
    for (code, site, sex), lst in sorted(heart_clips(Path(circor)).items()):
        seen = set()
        for score, pid, wav, a, b, label in sorted(lst, key=lambda t: t[0]):
            if pid in seen: continue
            seen.add(pid)
            sr, x = read_wav(wav); x = to_rate(x, sr)
            name = f"X_{sex}_{code}_{site}_{len(seen)}.wav"
            write_wav(out / name, edge_fade(x[int(a * RATE):int(b * RATE)]))
            meta["clips"].append({"file": f"sounds/{name}", "kind": "heart", "site": site, "code": code,
                                  "condition": label, "gender": sex, "src": "circor"})
            n += 1
            if len(seen) == per_heart: break

    for (code, side), lst in sorted(lung_clips(Path(icbhi)).items()):
        seen, k = set(), 0
        for score, pid, wav, a, b in sorted(lst, key=lambda t: t[0]):
            if pid in seen: continue
            seen.add(pid)
            zone = ZONES[k % 3]; k += 1
            site = f"{side}{zone}A"
            sr, x = read_wav(wav); x = to_rate(x, sr)
            name = f"X_{code}_{site}_{k}.wav"
            write_wav(out / name, edge_fade(x[int(a * RATE):int(b * RATE)]))
            meta["clips"].append({"file": f"sounds/{name}", "kind": "lung", "site": site, "code": code,
                                  "condition": LUNG_LABEL[code], "gender": "", "src": "icbhi"})
            n += 1
            if k == per_lung: break

    meta["sites"] = meta["sites"]
    (here / "sounds.json").write_text(json.dumps(meta, indent=1))
    print(n, "extra clips;", len(meta["clips"]), "total")

if __name__ == "__main__":
    main(*sys.argv[1:3])
