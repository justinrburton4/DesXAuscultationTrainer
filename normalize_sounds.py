"""Peak-normalize every clip in sounds/ to -1 dBFS (source clips peak at only ~1-3% of full scale,
which is near-inaudible on phone speakers). Safe to re-run."""
import array, wave
from pathlib import Path

TARGET = int(32767 * 0.89)
for f in sorted(Path(__file__).parent.glob("sounds/*.wav")):
    with wave.open(str(f)) as w:
        params, frames = w.getparams(), w.readframes(w.getnframes())
    s = array.array("h"); s.frombytes(frames)
    peak = max(max(s), -min(s)) or 1
    s = array.array("h", (max(-32768, min(32767, round(x * TARGET / peak))) for x in s))
    with wave.open(str(f), "wb") as w:
        w.setparams(params); w.writeframes(s.tobytes())
print("done")
