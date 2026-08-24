from __future__ import annotations
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
from pathlib import Path
from ml.data.loader import load_splits
from ml.data.validation import validate_no_text_overlap, frame_fingerprint

root=Path("data/splits");train,val,test=load_splits(root);overlap=validate_no_text_overlap(train,val,test)
assert not any(overlap.values()), f"Normalized text leakage detected: {overlap}"
manifest_path=root/"split_manifest.json"
if manifest_path.exists():
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    for name,df in [("train",train),("val",val),("test",test)]:
        actual=frame_fingerprint(df); expected=manifest["splits"][name]["fingerprint"]
        assert actual==expected, f"{name} fingerprint drift: {actual} != {expected}"
print({"rows":{"train":len(train),"val":len(val),"test":len(test)},"text_overlap":overlap,"manifest":"ok"})
