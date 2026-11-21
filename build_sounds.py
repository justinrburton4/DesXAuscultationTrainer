"""Copy HLS-CMDS v2 heart/lung wavs into sounds/ and write sounds.json.

Usage: python -I build_sounds.py <unzipped HS dir> <unzipped LS dir>
Filenames are <Gender>_<ConditionCode>_<SiteCode>.wav (the CSVs disagree with
the lung filenames, e.g. CSV says G for coarse crackles, files say CC).
"""
import json, shutil, sys
from pathlib import Path

HEART = {"N": "Normal", "AF": "Atrial fibrillation", "AVB": "AV block", "ESM": "Early systolic murmur",
         "MSM": "Mid systolic murmur", "LSM": "Late systolic murmur", "LDM": "Late diastolic murmur",
         "S3": "Third heart sound (S3)", "S4": "Fourth heart sound (S4)", "T": "Tachycardia"}
LUNG = {"N": "Normal", "W": "Wheezing", "R": "Rhonchi", "FC": "Fine crackles",
        "CC": "Coarse crackles", "PR": "Pleural rub"}
SITES = {"RUSB": "Right upper sternal border", "LUSB": "Left upper sternal border",
         "LLSB": "Lower left sternal border", "RC": "Right costal margin", "LC": "Left costal margin",
         "A": "Apex",
         "RUA": "Right upper anterior", "LUA": "Left upper anterior", "RMA": "Right mid anterior",
         "LMA": "Left mid anterior", "RLA": "Right lower anterior", "LLA": "Left lower anterior"}

here = Path(__file__).parent
out = here / "sounds"
out.mkdir(exist_ok=True)
clips = []
for kind, src, names in (("heart", Path(sys.argv[1]), HEART), ("lung", Path(sys.argv[2]), LUNG)):
    for f in sorted(src.glob("*.wav")):
        g, code, site = f.stem.split("_")
        assert code in names and site in SITES, f.name
        shutil.copy(f, out / f.name)
        clips.append({"file": f"sounds/{f.name}", "kind": kind, "site": site, "code": code,
                      "condition": names[code], "gender": g})
(here / "sounds.json").write_text(json.dumps({"sites": SITES, "clips": clips}, indent=1))
print(len(clips), "clips")
