from pathlib import Path

def test_no_ipynb_in_unified_repo():
    root=Path(__file__).resolve().parents[2]
    assert not list(root.rglob('*.ipynb'))
    migrated=list((root/'scripts/migrated_notebooks').glob('*.py'))
    assert len(migrated)==10
