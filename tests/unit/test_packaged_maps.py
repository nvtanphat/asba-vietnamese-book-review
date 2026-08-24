import json
from pathlib import Path

def test_maps_are_packaged_with_absa_core():
    root=Path(__file__).resolve().parents[2]
    maps=root/'packages/absa_core/absa_core/data/maps'
    emoji=json.loads((maps/'emoji_map.json').read_text(encoding='utf-8'))
    vocab=json.loads((maps/'vocab_map.json').read_text(encoding='utf-8'))
    assert emoji.get('thumbs_up')
    assert isinstance(vocab,dict) and len(vocab)>0
