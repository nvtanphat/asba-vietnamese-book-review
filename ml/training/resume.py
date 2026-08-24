from pathlib import Path
def can_resume(output_dir): return (Path(output_dir)/"last.pt").exists()
