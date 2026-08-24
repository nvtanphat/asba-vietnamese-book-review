from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = ROOT / ".kaggle_work"
MODELS = ("logistic", "linear_svm", "textcnn", "bilstm", "phobert", "xlmr", "mdeberta", "vit5")
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
# Kaggle auto-extracts any dataset file ending in .zip/.tar on upload, so the source bundle
# uses a non-archive extension to reach the mounted dataset intact as a single file.
SOURCE_BUNDLE_NAME = "sentenai_src_bundle.dat"


class KaggleToolError(RuntimeError):
    pass


def _print_cmd(cmd: list[str]) -> None:
    print("+ " + " ".join(str(x) for x in cmd), flush=True)


def get_kaggle_env() -> dict[str, str]:
    env = dict(os.environ)
    local_kaggle_dir = ROOT / ".kaggle"
    local_key = local_kaggle_dir / "kaggle.json"
    if local_key.exists():
        env["KAGGLE_CONFIG_DIR"] = str(local_kaggle_dir)
        try:
            if hasattr(os, "chmod") and os.name != "nt":
                local_key.chmod(0o600)
        except Exception:
            pass
    return env


def run_cmd(cmd: list[str], *, cwd: Path | None = None, check: bool = True, capture: bool = False, env: dict | None = None) -> subprocess.CompletedProcess:
    _print_cmd(cmd)
    merged_env = get_kaggle_env()
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        text=True,
        check=check,
        capture_output=capture,
    )



def require_kaggle() -> str:
    exe = shutil.which("kaggle")
    if not exe:
        raise KaggleToolError(
            "Kaggle CLI not found. Install the official CLI with Python 3.11+: `pip install -U kaggle`."
        )
    return exe


def slug_from_handle(handle: str) -> str:
    if "/" not in handle:
        raise KaggleToolError(f"Expected Kaggle handle owner/slug, got: {handle}")
    return handle.split("/", 1)[1]


def validate_handle(handle: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", handle):
        raise KaggleToolError(f"Invalid Kaggle handle: {handle}. Expected owner/slug.")
    return handle


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


@dataclass
class KernelSpec:
    model: str
    kernel_handle: str
    data_handle: str
    accelerator: str = DEFAULT_ACCELERATOR
    resume_handle: str | None = None
    use_tuned: bool = False
    run_test: bool = False
    internet: bool = True

    @property
    def kernel_slug(self) -> str:
        return slug_from_handle(self.kernel_handle)


SOURCE_EXCLUDES = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    ".kaggle_work",
    ".kaggle",
    "node_modules",
    ".next",
    "artifacts",
    "experiments",
    "mlruns",
    ".dvc",
}


def should_include(rel: Path) -> bool:
    if any(part in SOURCE_EXCLUDES or part == "__pycache__" for part in rel.parts):
        return False
    if rel.parts and rel.parts[0] == "data":
        # Kaggle Dataset is the source of data; only code-side maps are packaged elsewhere.
        return False
    if rel.name in {"kaggle.json", ".kaggle.json", "dataset-metadata.json", "kernel-metadata.json"}:
        return False
    if rel.name.startswith(".env"):
        return False
    if rel.suffix in {".pyc", ".pyo", ".zip", ".pem", ".key"}:
        return False
    return True


def build_source_zip(out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if should_include(rel):
                zf.write(path, rel.as_posix())
    return out_path


def stage_data_dataset(handle: str, *, title: str | None = None) -> Path:
    handle = validate_handle(handle)
    stage = WORK_ROOT / "datasets" / "main"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    required = ROOT / "data/raw/tiki-book-review_merged_fixed_v3.json"
    if not required.exists():
        raise KaggleToolError(f"Missing raw dataset: {required}")
    shutil.copy2(required, stage / required.name)

    for name in ("train.json", "val.json", "test.json", "split_manifest.json"):
        copy_if_exists(ROOT / "data/splits" / name, stage / name)
    for name in ("emoji_map.json", "vocab_map.json"):
        copy_if_exists(ROOT / "data/maps" / name, stage / name)
    # Kaggle silently auto-extracts any uploaded *.zip into individual files, so the archive
    # is named with a non-archive extension to survive the upload intact.
    build_source_zip(stage / SOURCE_BUNDLE_NAME)


    write_json(
        stage / "dataset-metadata.json",
        {
            "title": title or "SentenAI Vietnamese Tiki Book Review ABSA",
            "id": handle,
            "licenses": [{"name": "unknown"}],
        },
    )
    print(f"Prepared Kaggle data dataset at {stage}")
    return stage


def dataset_exists(handle: str) -> bool:
    kaggle = require_kaggle()
    proc = run_cmd([kaggle, "datasets", "status", handle], check=False, capture=True)
    return proc.returncode == 0


def sync_dataset(stage: Path, handle: str, *, message: str, public: bool = False) -> None:
    kaggle = require_kaggle()
    if dataset_exists(handle):
        run_cmd([kaggle, "datasets", "version", "-p", str(stage), "-m", message, "-q", "-t", "-r", "skip"])
    else:
        cmd = [kaggle, "datasets", "create", "-p", str(stage), "-q", "-t", "-r", "skip"]
        if public:
            cmd.append("--public")
        run_cmd(cmd)


def stage_resume_dataset(model: str, handle: str, source_dir: Path) -> Path:
    handle = validate_handle(handle)
    stage = WORK_ROOT / "datasets" / f"resume-{model}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    last_candidates = list(source_dir.rglob("last.pt"))
    if not last_candidates:
        raise KaggleToolError(f"No last.pt found under {source_dir}")
    shutil.copy2(last_candidates[0], stage / "last.pt")
    best_candidates = list(source_dir.rglob("best.pt"))
    if best_candidates:
        shutil.copy2(best_candidates[0], stage / "best.pt")

    write_json(
        stage / "dataset-metadata.json",
        {
            "title": f"SentenAI {model} resume checkpoint",
            "id": handle,
            "licenses": [{"name": "unknown"}],
        },
    )
    return stage


def kaggle_runner_source(spec: KernelSpec) -> str:
    data_slug = slug_from_handle(spec.data_handle)
    resume_slug = slug_from_handle(spec.resume_handle) if spec.resume_handle else None
    flags = ["--model", spec.model]
    if spec.resume_handle:
        flags.append("--resume")
    if spec.use_tuned:
        flags.append("--use-tuned")
    if not spec.run_test:
        flags.append("--no-test")
    flag_literal = repr(flags)
    resume_slug_literal = repr(resume_slug)

    return textwrap.dedent(
        f"""
        from __future__ import annotations

        import json
        import os
        import shutil
        import subprocess
        import sys
        import zipfile
        from pathlib import Path

        WORK = Path('/kaggle/working')
        PROJECT = WORK / 'sentenai'
        INPUT = Path('/kaggle/input')
        DATA_SLUG = {data_slug!r}
        RESUME_SLUG = {resume_slug_literal}
        MODEL = {spec.model!r}
        TRAIN_FLAGS = {flag_literal}
        BUNDLE_NAME = {SOURCE_BUNDLE_NAME!r}

        def run(cmd, cwd=None):
            print('+ ' + ' '.join(map(str, cmd)), flush=True)
            subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

        def find_dataset_dir(slug):
            direct = INPUT / slug
            if direct.exists():
                return direct
            # Recursive check in case Kaggle nests inside /kaggle/input/datasets/ or similar
            for path in INPUT.rglob("*"):
                if path.is_dir() and (path.name == slug or (path / "train.json").exists() or (path / BUNDLE_NAME).exists()):
                    return path
            matches = [p for p in INPUT.iterdir() if p.is_dir() and (p.name == slug or p.name.endswith(slug))]
            if matches:
                return matches[0]
            raise FileNotFoundError(f"Kaggle input dataset not found; inputs={{list(INPUT.rglob('*'))}}")



        data_src = find_dataset_dir(DATA_SLUG)
        if PROJECT.exists():
            shutil.rmtree(PROJECT)
        PROJECT.mkdir(parents=True)
        bundle_path = data_src / BUNDLE_NAME
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"{{bundle_path}} not found. The mounted dataset {{DATA_SLUG!r}} is missing {{BUNDLE_NAME}}; "
                "re-run `python -m tools.kaggle_cli run --sync-data ...` so the source bundle is "
                "included in the dataset version this kernel mounts."
            )
        with zipfile.ZipFile(bundle_path) as zf:
            zf.extractall(PROJECT)

        # Install only dependencies Kaggle images commonly miss. Core torch/sklearn/transformers
        # are left to the image unless an import check fails, reducing startup time.
        required = ['emoji', 'ftfy', 'pyvi', 'sentencepiece', 'peft', 'accelerate']
        missing = []
        for pkg in required:
            module = 'pyvi' if pkg == 'pyvi' else pkg
            try:
                __import__(module)
            except Exception:
                missing.append(pkg)
        if missing:
            run([sys.executable, '-m', 'pip', 'install', '-q', *missing])
        if MODEL == 'vit5':
            # transformers v5's rewritten tokenizer backend raises KeyError(0) in
            # convert_to_native_format() for this checkpoint's legacy sentencepiece vocab
            # (not yet re-saved in the v5 native format); pin the last 4.x line to avoid it.
            run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers<5'])
            # peft's LoRA dispatcher probes torchao even for plain (non-quantized) LoRA and
            # raises if a too-old torchao is present (Kaggle ships 0.10.0, peft wants >=0.16.0).
            # We don't use torchao at all, so remove it and let peft's check report "unavailable".
            subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-q', '-y', 'torchao'], check=False)
        run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT / 'packages/absa_core')])

        (PROJECT / 'data/raw').mkdir(parents=True, exist_ok=True)
        (PROJECT / 'data/splits').mkdir(parents=True, exist_ok=True)
        (PROJECT / 'data/maps').mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_src / 'tiki-book-review_merged_fixed_v3.json', PROJECT / 'data/raw/tiki-book-review_merged_fixed_v3.json')
        for name in ('train.json', 'val.json', 'test.json', 'split_manifest.json'):
            src = data_src / name
            if src.exists():
                shutil.copy2(src, PROJECT / 'data/splits' / name)
        for name in ('emoji_map.json', 'vocab_map.json'):

            src = data_src / name
            if src.exists():
                shutil.copy2(src, PROJECT / 'data/maps' / name)

        exp_dir = PROJECT / 'experiments' / MODEL
        exp_dir.mkdir(parents=True, exist_ok=True)
        if RESUME_SLUG:
            resume_src = find_dataset_dir(RESUME_SLUG)
            if not (resume_src / 'last.pt').exists():
                raise FileNotFoundError(f'Resume dataset {{RESUME_SLUG}} has no last.pt')
            shutil.copy2(resume_src / 'last.pt', exp_dir / 'last.pt')
            if (resume_src / 'best.pt').exists():
                shutil.copy2(resume_src / 'best.pt', exp_dir / 'best.pt')
            print(f'Resuming {{MODEL}} from {{resume_src}}')

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        print('GPU:', subprocess.check_output([sys.executable, '-c', 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'], text=True).strip())

        train_error = None
        try:
            run([sys.executable, '-m', 'ml.train', *TRAIN_FLAGS], cwd=PROJECT)
        except Exception as exc:
            # Export the latest checkpoint even if training exits with a normal Python error.
            # A hard Kaggle VM termination can still prevent this finally block from running.
            train_error = exc
        finally:
            # Only /kaggle/working is persisted as Kernel output. Export compact artifacts there.
            export = WORK / 'sentenai-output' / MODEL
            export.mkdir(parents=True, exist_ok=True)
            for name in ('last.pt', 'best.pt', 'model.pt', 'metadata.json', 'metrics.json', 'lineage.json', 'run_manifest.json', 'MODEL_CARD.md', 'test_predictions.npy'):
                src = exp_dir / name
                if src.exists():
                    shutil.copy2(src, export / name)
            for dirname in ('model', 'tokenizer', 'encoder'):
                src = exp_dir / dirname
                if src.exists():
                    shutil.copytree(src, export / dirname, dirs_exist_ok=True)
            manifest = {{'model': MODEL, 'flags': TRAIN_FLAGS, 'resume_source': RESUME_SLUG, 'error': None if train_error is None else repr(train_error)}}
            (export / 'kaggle_run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            print('Exported:', export)

        if train_error is not None:
            raise train_error
        """
    ).strip() + "\n"


def combined_kernel_source(specs: list[KernelSpec]) -> str:
    """One run.py that trains every model in `specs` sequentially inside a single Kaggle
    session, so the whole batch only ever occupies one concurrent-GPU-session slot instead
    of one slot per model."""
    data_slug = slug_from_handle(specs[0].data_handle)
    # vit5 needs a downgraded transformers + torchao removed; run it last so every other
    # model still trains under the base image's normal (newer) transformers.
    ordered = sorted(specs, key=lambda s: s.model == "vit5")
    jobs = []
    for spec in ordered:
        flags = ["--model", spec.model]
        if spec.resume_handle:
            flags.append("--resume")
        if spec.use_tuned:
            flags.append("--use-tuned")
        if not spec.run_test:
            flags.append("--no-test")
        jobs.append({"model": spec.model, "flags": flags, "resume_slug": slug_from_handle(spec.resume_handle) if spec.resume_handle else None})
    jobs_literal = repr(jobs)

    return textwrap.dedent(
        f"""
        from __future__ import annotations

        import json
        import os
        import shutil
        import subprocess
        import sys
        import zipfile
        from pathlib import Path

        WORK = Path('/kaggle/working')
        PROJECT = WORK / 'sentenai'
        INPUT = Path('/kaggle/input')
        DATA_SLUG = {data_slug!r}
        JOBS = {jobs_literal}
        BUNDLE_NAME = {SOURCE_BUNDLE_NAME!r}

        def run(cmd, cwd=None):
            print('+ ' + ' '.join(map(str, cmd)), flush=True)
            subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

        def find_dataset_dir(slug):
            direct = INPUT / slug
            if direct.exists():
                return direct
            for path in INPUT.rglob("*"):
                if path.is_dir() and (path.name == slug or (path / "train.json").exists() or (path / BUNDLE_NAME).exists()):
                    return path
            matches = [p for p in INPUT.iterdir() if p.is_dir() and (p.name == slug or p.name.endswith(slug))]
            if matches:
                return matches[0]
            raise FileNotFoundError(f"Kaggle input dataset not found; inputs={{list(INPUT.rglob('*'))}}")

        data_src = find_dataset_dir(DATA_SLUG)
        if PROJECT.exists():
            shutil.rmtree(PROJECT)
        PROJECT.mkdir(parents=True)
        bundle_path = data_src / BUNDLE_NAME
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"{{bundle_path}} not found. The mounted dataset {{DATA_SLUG!r}} is missing {{BUNDLE_NAME}}; "
                "re-run `python -m tools.kaggle_cli run-all --sync-data ...` so the source bundle is "
                "included in the dataset version this kernel mounts."
            )
        with zipfile.ZipFile(bundle_path) as zf:
            zf.extractall(PROJECT)

        required = ['emoji', 'ftfy', 'pyvi', 'sentencepiece', 'peft', 'accelerate']
        missing = []
        for pkg in required:
            module = 'pyvi' if pkg == 'pyvi' else pkg
            try:
                __import__(module)
            except Exception:
                missing.append(pkg)
        if missing:
            run([sys.executable, '-m', 'pip', 'install', '-q', *missing])
        run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT / 'packages/absa_core')])

        (PROJECT / 'data/raw').mkdir(parents=True, exist_ok=True)
        (PROJECT / 'data/splits').mkdir(parents=True, exist_ok=True)
        (PROJECT / 'data/maps').mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_src / 'tiki-book-review_merged_fixed_v3.json', PROJECT / 'data/raw/tiki-book-review_merged_fixed_v3.json')
        for name in ('train.json', 'val.json', 'test.json', 'split_manifest.json'):
            src = data_src / name
            if src.exists():
                shutil.copy2(src, PROJECT / 'data/splits' / name)
        for name in ('emoji_map.json', 'vocab_map.json'):
            src = data_src / name
            if src.exists():
                shutil.copy2(src, PROJECT / 'data/maps' / name)

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        print('GPU:', subprocess.check_output([sys.executable, '-c', 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'], text=True).strip())

        results = {{}}
        for job in JOBS:
            model = job['model']
            flags = job['flags']
            resume_slug = job['resume_slug']
            print()
            print('===== ' + model + ' =====')

            if model == 'vit5':
                # transformers v5's rewritten tokenizer backend raises KeyError(0) for this
                # checkpoint's legacy sentencepiece vocab; pin the last 4.x line. Run last
                # in the batch since this downgrade isn't reverted afterward.
                run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers<5'])
                # peft's LoRA dispatcher probes torchao even for plain (non-quantized) LoRA
                # and raises if a too-old torchao is present (Kaggle ships 0.10.0, peft
                # wants >=0.16.0). Not used at all here, so remove it.
                subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-q', '-y', 'torchao'], check=False)

            exp_dir = PROJECT / 'experiments' / model
            exp_dir.mkdir(parents=True, exist_ok=True)
            if resume_slug:
                resume_src = find_dataset_dir(resume_slug)
                if (resume_src / 'last.pt').exists():
                    shutil.copy2(resume_src / 'last.pt', exp_dir / 'last.pt')
                    if (resume_src / 'best.pt').exists():
                        shutil.copy2(resume_src / 'best.pt', exp_dir / 'best.pt')
                    print(f'Resuming {{model}} from {{resume_src}}')

            train_error = None
            try:
                run([sys.executable, '-m', 'ml.train', *flags], cwd=PROJECT)
            except Exception as exc:
                # Keep going with the remaining models even if this one fails.
                train_error = exc
                print(f'{{model}} failed: {{exc}}')
            finally:
                export = WORK / 'sentenai-output' / model
                export.mkdir(parents=True, exist_ok=True)
                for name in ('last.pt', 'best.pt', 'model.pt', 'metadata.json', 'metrics.json', 'lineage.json', 'run_manifest.json', 'MODEL_CARD.md', 'test_predictions.npy'):
                    src = exp_dir / name
                    if src.exists():
                        shutil.copy2(src, export / name)
                for dirname in ('model', 'tokenizer', 'encoder'):
                    src = exp_dir / dirname
                    if src.exists():
                        shutil.copytree(src, export / dirname, dirs_exist_ok=True)
                manifest = {{'model': model, 'flags': flags, 'resume_source': resume_slug, 'error': None if train_error is None else repr(train_error)}}
                (export / 'kaggle_run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
                print('Exported:', export)
                results[model] = 'ok' if train_error is None else 'error'

                # Everything worth keeping is already copied into `export` above, so the
                # exp_dir checkpoint/model files and this model's HF pretrained-weight cache
                # are pure duplication from here on. A combined run trains several models
                # back to back on one disk, and leaving these in place accumulates several
                # GB per model until the kernel runs out of space mid-checkpoint-save (this
                # crashed a real run: mdeberta hit `OSError: No space left on device` while
                # writing last.pt, after phobert+xlmr had already left their copies behind).
                for name in ('last.pt', 'best.pt', 'model.pt'):
                    f = exp_dir / name
                    if f.exists():
                        f.unlink()
                for dirname in ('model', 'tokenizer', 'encoder'):
                    d = exp_dir / dirname
                    if d.exists():
                        shutil.rmtree(d, ignore_errors=True)
                shutil.rmtree(Path.home() / '.cache' / 'huggingface', ignore_errors=True)

        print()
        print('===== SUMMARY =====')
        for model, status in results.items():
            print(f'{{model}}: {{status}}')
        if any(status == 'error' for status in results.values()):
            raise SystemExit('One or more models failed; see SUMMARY above and per-model kaggle_run_manifest.json.')
        """
    ).strip() + "\n"


def prepare_combined_kernel(specs: list[KernelSpec], kernel_handle: str, accelerator: str, *, internet: bool = True) -> Path:
    for spec in specs:
        validate_handle(spec.data_handle)
        if spec.resume_handle:
            validate_handle(spec.resume_handle)
        if spec.model not in MODELS:
            raise KaggleToolError(f"Unknown model {spec.model}; choose one of {', '.join(MODELS)}")
    validate_handle(kernel_handle)

    stage = WORK_ROOT / "kernels" / "combined"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    build_source_zip(stage / "sentenai_src.zip")
    (stage / "run.py").write_text(combined_kernel_source(specs), encoding="utf-8")

    sources = [specs[0].data_handle]
    for spec in specs:
        if spec.resume_handle and spec.resume_handle not in sources:
            sources.append(spec.resume_handle)
    metadata = {
        "id": kernel_handle,
        "title": slug_from_handle(kernel_handle),
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": bool(internet),
        "machine_shape": accelerator,
        "dataset_sources": sources,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    write_json(stage / "kernel-metadata.json", metadata)
    print(f"Prepared combined Kaggle kernel at {stage} ({len(specs)} models: {', '.join(s.model for s in specs)})")
    return stage


def prepare_kernel(spec: KernelSpec) -> Path:
    validate_handle(spec.kernel_handle)
    validate_handle(spec.data_handle)
    if spec.resume_handle:
        validate_handle(spec.resume_handle)
    if spec.model not in MODELS:
        raise KaggleToolError(f"Unknown model {spec.model}; choose one of {', '.join(MODELS)}")

    stage = WORK_ROOT / "kernels" / spec.model
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    build_source_zip(stage / "sentenai_src.zip")
    (stage / "run.py").write_text(kaggle_runner_source(spec), encoding="utf-8")

    sources = [spec.data_handle]
    if spec.resume_handle:
        sources.append(spec.resume_handle)
    metadata = {
        "id": spec.kernel_handle,
        # Title must match the id's slug (Kaggle derives its own slug from the title and
        # rejects/mismatches it otherwise) so it can't hardcode the raw model name, which
        # may contain underscores (e.g. "linear_svm") that aren't valid in a Kaggle slug.
        "title": slug_from_handle(spec.kernel_handle),
        "code_file": "run.py",

        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": bool(spec.internet),
        "machine_shape": spec.accelerator,
        "dataset_sources": sources,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    write_json(stage / "kernel-metadata.json", metadata)
    print(f"Prepared Kaggle kernel at {stage}")
    return stage


def push_kernel(stage: Path, accelerator: str, timeout: int | None = None) -> None:
    kaggle = require_kaggle()
    cmd = [kaggle, "kernels", "push", "-p", str(stage), "--accelerator", accelerator]
    if timeout is not None:
        cmd += ["--timeout", str(timeout)]
    run_cmd(cmd)


def kernel_output(kernel: str, output_dir: Path, file_pattern: str | None = None) -> None:
    """Downloads a kernel's output files with true chunked streaming.

    The official `kaggle kernels output` call (both the CLI and `KaggleApi.kernels_output`)
    fetches each file with `requests.get(url, stream=True)` but then reads the whole response
    into memory via `.content` before writing it to disk — the `stream=True` doesn't actually
    avoid buffering the full file. For this project's multi-GB transformer checkpoint files
    (last.pt/best.pt can be 1-2GB), that reliably raised MemoryError partway through a batch
    download, silently truncating files (observed: xlmr's last.pt landed as a 0-byte file).
    This re-implements just the download loop with real chunked writes, using the same
    `kaggle` package/credentials the rest of this tool relies on.
    """
    require_kaggle()
    validate_handle(kernel)
    output_dir.mkdir(parents=True, exist_ok=True)
    owner_slug, kernel_slug = kernel.split("/", 1)
    for key, value in get_kaggle_env().items():
        os.environ[key] = value

    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    api = KaggleApi()
    api.authenticate()
    compiled_pattern = re.compile(file_pattern) if file_pattern else None

    downloaded = 0
    token: str | None = None
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner_slug
            request.kernel_slug = kernel_slug
            request.page_size = 20
            if token:
                request.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files or []:
                if compiled_pattern and not compiled_pattern.search(item.file_name):
                    continue
                outfile = output_dir / item.file_name
                outfile.parent.mkdir(parents=True, exist_ok=True)
                with requests.get(item.url, stream=True) as resp:
                    resp.raise_for_status()
                    with open(outfile, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                            if chunk:
                                fh.write(chunk)
                print(f"Output file downloaded to {outfile}")
                downloaded += 1
            token = response.next_page_token
            if not token:
                break
    if not downloaded:
        raise KaggleToolError(f"No output files found for kernel {kernel}")


def cmd_doctor(_args: argparse.Namespace) -> None:
    kaggle = require_kaggle()
    version = run_cmd([kaggle, "--version"], capture=True)
    print(version.stdout.strip())
    # A tiny authenticated call catches missing/invalid credentials without mutating anything.
    proc = run_cmd([kaggle, "datasets", "list", "-p", "1"], check=False, capture=True)

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise KaggleToolError("Kaggle CLI is installed but authentication/API check failed:\n" + detail)
    print("Kaggle authentication/API check: OK")


def cmd_prepare_data(args: argparse.Namespace) -> None:
    stage_data_dataset(args.dataset, title=args.title)


def cmd_sync_data(args: argparse.Namespace) -> None:
    stage = stage_data_dataset(args.dataset, title=args.title)
    sync_dataset(stage, args.dataset, message=args.message, public=args.public)


def make_spec(args: argparse.Namespace, *, resume_handle: str | None = None) -> KernelSpec:
    # Kaggle kernel slugs only allow letters/digits/hyphens, so a model name with an
    # underscore (e.g. "linear_svm") must be hyphenated or Kaggle silently mismatches
    # the title-derived slug against the id we asked for.
    kernel = args.kernel or f"{args.owner}/sentenai-{args.model.replace('_', '-')}"
    return KernelSpec(
        model=args.model,
        kernel_handle=kernel,
        data_handle=args.dataset,
        accelerator=args.accelerator,
        resume_handle=resume_handle if resume_handle is not None else getattr(args, "resume_dataset", None),
        use_tuned=args.use_tuned,
        run_test=args.run_test,
        internet=not args.no_internet,
    )


def cmd_prepare_kernel(args: argparse.Namespace) -> None:
    prepare_kernel(make_spec(args))


def cmd_run(args: argparse.Namespace) -> None:
    spec = make_spec(args)
    if args.sync_data:
        stage = stage_data_dataset(spec.data_handle)
        sync_dataset(stage, spec.data_handle, message="SentenAI training data/split sync", public=False)
    kernel_stage = prepare_kernel(spec)
    push_kernel(kernel_stage, spec.accelerator, args.timeout)
    print(f"Kernel submitted: {spec.kernel_handle}")


def cmd_run_all(args: argparse.Namespace) -> None:
    """Submit every requested model's training kernel with a single push attempt each (no
    retry loop) — if the account's concurrent batch-GPU-session cap is full, that push is
    reported and skipped rather than waited on."""
    kaggle = require_kaggle()
    models = args.models or list(MODELS)

    if args.sync_data:
        stage = stage_data_dataset(args.dataset)
        sync_dataset(stage, args.dataset, message="SentenAI training data/split sync", public=False)

    if args.combine:
        specs = []
        for name in models:
            model_args = argparse.Namespace(**{**vars(args), "model": name, "kernel": None, "resume_dataset": None})
            specs.append(make_spec(model_args))
        kernel_handle = args.kernel or f"{args.owner}/sentenai-combined"
        kernel_stage = prepare_combined_kernel(specs, kernel_handle, args.accelerator, internet=not args.no_internet)
        cmd = [kaggle, "kernels", "push", "-p", str(kernel_stage), "--accelerator", args.accelerator]
        proc = run_cmd(cmd, check=False, capture=True)
        output = (proc.stdout or "") + (proc.stderr or "")
        print(output.strip())
        if "Maximum batch GPU session" in output:
            print(f"[combined] GPU session cap full — not submitted ({', '.join(models)}).")
            return
        if proc.returncode != 0:
            raise KaggleToolError(f"kernels push failed for combined kernel:\n{output}")
        print(f"Submitted combined kernel ({', '.join(models)}): {kernel_handle}")
        return

    submitted = []
    for name in models:
        model_args = argparse.Namespace(**{**vars(args), "model": name, "kernel": None, "resume_dataset": None})
        spec = make_spec(model_args)
        kernel_stage = prepare_kernel(spec)
        cmd = [kaggle, "kernels", "push", "-p", str(kernel_stage), "--accelerator", spec.accelerator]
        proc = run_cmd(cmd, check=False, capture=True)
        output = (proc.stdout or "") + (proc.stderr or "")
        print(output.strip())
        if "Maximum batch GPU session" in output:
            print(f"[{name}] GPU session cap full — not submitted.\n")
            continue
        if proc.returncode != 0:
            raise KaggleToolError(f"kernels push failed for {name}:\n{output}")
        print(f"Submitted {name}: {spec.kernel_handle}\n")
        submitted.append((name, spec.kernel_handle))

    print("Submitted:" if submitted else "Nothing submitted (GPU session cap full for all).")
    for name, handle in submitted:
        print(f"  {name}: {handle}")


def cmd_status(args: argparse.Namespace) -> None:
    kaggle = require_kaggle()
    run_cmd([kaggle, "kernels", "status", validate_handle(args.kernel)])


def cmd_logs(args: argparse.Namespace) -> None:
    kaggle = require_kaggle()
    cmd = [kaggle, "kernels", "logs", validate_handle(args.kernel)]
    if args.follow:
        cmd.append("--follow")
    run_cmd(cmd)


def cmd_output(args: argparse.Namespace) -> None:
    kernel_output(args.kernel, Path(args.output_dir), args.file_pattern)


def cmd_resume(args: argparse.Namespace) -> None:
    # Kaggle kernel slugs only allow letters/digits/hyphens, so a model name with an
    # underscore (e.g. "linear_svm") must be hyphenated or Kaggle silently mismatches
    # the title-derived slug against the id we asked for.
    kernel = args.kernel or f"{args.owner}/sentenai-{args.model.replace('_', '-')}"
    out = WORK_ROOT / "outputs" / args.model
    if out.exists() and args.clean_output:
        shutil.rmtree(out)
    kernel_output(kernel, out, r".*(last|best)\.pt$")

    resume_handle = args.resume_dataset or f"{args.owner}/sentenai-{args.model.replace('_', '-')}-resume"
    resume_stage = stage_resume_dataset(args.model, resume_handle, out)
    sync_dataset(resume_stage, resume_handle, message=f"Resume checkpoint for {args.model}", public=False)

    spec = make_spec(args, resume_handle=resume_handle)
    kernel_stage = prepare_kernel(spec)
    push_kernel(kernel_stage, spec.accelerator, args.timeout)
    print(f"Resume kernel submitted: {spec.kernel_handle} using {resume_handle}")



def cmd_collect(args: argparse.Namespace) -> None:
    """Download a completed Kaggle run and ingest its compact output into local experiments/."""
    # Kaggle kernel slugs only allow letters/digits/hyphens, so a model name with an
    # underscore (e.g. "linear_svm") must be hyphenated or Kaggle silently mismatches
    # the title-derived slug against the id we asked for.
    kernel = args.kernel or f"{args.owner}/sentenai-{args.model.replace('_', '-')}"
    out = WORK_ROOT / "outputs" / f"collect-{args.model}"
    if out.exists() and args.clean_output:
        shutil.rmtree(out)
    kernel_output(kernel, out)
    candidates = [p for p in out.rglob(args.model) if p.is_dir() and (p / "metrics.json").exists()]
    if not candidates:
        candidates = [p.parent for p in out.rglob("metrics.json") if p.parent.name == args.model]
    if not candidates:
        raise KaggleToolError(f"No SentenAI output for model={args.model} found under {out}")
    src = candidates[0]
    dst = ROOT / "experiments" / args.model
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"Ingested Kaggle output: {src} -> {dst}")
    if args.register:
        manifest = {}
        mf = dst / "run_manifest.json"
        if mf.exists():
            manifest = json.loads(mf.read_text(encoding="utf-8"))
        cmd = [sys.executable, "-m", "mlops", "register", "--model", args.model, "--run-dir", str(dst)]
        if manifest.get("tracking_run_id"):
            cmd += ["--tracking-run-id", str(manifest["tracking_run_id"])]
        run_cmd(cmd, cwd=ROOT)

def add_training_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--owner", required=True, help="Kaggle username/organization")
    p.add_argument("--dataset", required=True, help="Main dataset handle owner/slug")
    p.add_argument("--kernel", help="Kernel handle owner/slug; default: OWNER/sentenai-MODEL")
    p.add_argument("--accelerator", default=DEFAULT_ACCELERATOR, help="Kaggle accelerator id")
    p.add_argument("--resume-dataset", help="Optional checkpoint dataset owner/slug")
    p.add_argument("--use-tuned", action="store_true")
    p.add_argument("--run-test", action="store_true", help="Unseal test split. Default remote training is validation-only.")
    p.add_argument("--no-internet", action="store_true", help="Disable Kaggle internet (requires model weights to be mounted separately).")
    p.add_argument("--timeout", type=int, help="Optional Kaggle CLI push timeout in seconds")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tools.kaggle_cli",
        description="SentenAI Kaggle orchestration. This tool shells out to the official `kaggle` CLI.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Check Kaggle CLI installation and authentication")
    doctor.set_defaults(func=cmd_doctor)

    pd = sub.add_parser("prepare-data", help="Stage the immutable training dataset for Kaggle")
    pd.add_argument("--dataset", required=True, help="owner/slug")
    pd.add_argument("--title")
    pd.set_defaults(func=cmd_prepare_data)

    sd = sub.add_parser("sync-data", help="Create/version the private Kaggle training dataset")
    sd.add_argument("--dataset", required=True, help="owner/slug")
    sd.add_argument("--title")
    sd.add_argument("--message", default="SentenAI data/split update")
    sd.add_argument("--public", action="store_true", help="Create public dataset (private is default)")
    sd.set_defaults(func=cmd_sync_data)

    pk = sub.add_parser("prepare-kernel", help="Build a Kaggle script bundle without submitting it")
    add_training_args(pk)
    pk.set_defaults(func=cmd_prepare_kernel)

    runp = sub.add_parser("run", help="Build and submit one training kernel")
    add_training_args(runp)
    runp.add_argument("--sync-data", action="store_true", help="Create/version main Kaggle dataset before submit")
    runp.set_defaults(func=cmd_run)

    run_all = sub.add_parser("run-all", help="Submit training kernels for every model (retries while the GPU session cap is full)")
    run_all.add_argument("--models", nargs="*", choices=MODELS, help="Subset to run; default is all 8")
    run_all.add_argument("--owner", required=True, help="Kaggle username/organization")
    run_all.add_argument("--dataset", required=True, help="Main dataset handle owner/slug")
    run_all.add_argument("--kernel", help="Kernel handle for --combine; default: OWNER/sentenai-combined")
    run_all.add_argument("--combine", action="store_true", help="Train every requested model sequentially inside ONE kernel (one GPU session slot total) instead of one kernel per model")
    run_all.add_argument("--accelerator", default=DEFAULT_ACCELERATOR, help="Kaggle accelerator id")
    run_all.add_argument("--use-tuned", action="store_true")
    run_all.add_argument("--run-test", action="store_true", help="Unseal test split. Default remote training is validation-only.")
    run_all.add_argument("--no-internet", action="store_true")
    run_all.add_argument("--timeout", type=int, help="Optional Kaggle CLI push timeout in seconds")
    run_all.add_argument("--sync-data", action="store_true", help="Create/version main Kaggle dataset before submitting any kernel")
    run_all.set_defaults(func=cmd_run_all)

    st = sub.add_parser("status", help="Show latest Kaggle kernel status")
    st.add_argument("--kernel", required=True, help="owner/kernel-slug")
    st.set_defaults(func=cmd_status)

    logs = sub.add_parser("logs", help="Show/stream Kaggle kernel logs")
    logs.add_argument("--kernel", required=True)
    logs.add_argument("--follow", action="store_true")
    logs.set_defaults(func=cmd_logs)

    out = sub.add_parser("output", help="Download latest Kaggle kernel output")
    out.add_argument("--kernel", required=True)
    out.add_argument("--output-dir", default=str(WORK_ROOT / "outputs"))
    out.add_argument("--file-pattern")
    out.set_defaults(func=cmd_output)

    resume = sub.add_parser("resume", help="Download last checkpoint, version a private resume dataset, and resubmit")
    add_training_args(resume)
    resume.add_argument("--clean-output", action="store_true")
    resume.set_defaults(func=cmd_resume)

    collect = sub.add_parser("collect", help="Download completed output into experiments/ and optionally register it")
    collect.add_argument("--model", required=True, choices=MODELS)
    collect.add_argument("--owner", required=True)
    collect.add_argument("--kernel")
    collect.add_argument("--clean-output", action="store_true")
    collect.add_argument("--register", action="store_true")
    collect.set_defaults(func=cmd_collect)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (KaggleToolError, subprocess.CalledProcessError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
