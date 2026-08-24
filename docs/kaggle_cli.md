# Kaggle CLI training tool

SentenAI wraps the **official Kaggle CLI** with `python -m tools.kaggle_cli`. The wrapper does not call Kaggle's Python SDK directly; it shells out to the installed `kaggle` executable, so local credentials and Kaggle CLI behavior stay standard.

## Why this exists

The fair benchmark is designed to run locally or on Kaggle without changing the ML protocol. The Kaggle wrapper keeps the same frozen split, model config, validation metric and checkpoint semantics while adding remote orchestration:

1. stage/version the private training dataset;
2. package the current source tree into a Kaggle script bundle;
3. submit a kernel with a selected accelerator (T4 by default);
4. inspect status or stream logs;
5. download outputs;
6. publish `last.pt` as a small private resume dataset and resubmit with `--resume`.

Remote model training is **validation-only by default**. This preserves the sealed test split. Add `--run-test` only for the final model/evaluation run.

## Prerequisite

The current official Kaggle CLI requires Python 3.11+.

```bash
pip install -U kaggle
kaggle auth login
python -m tools.kaggle_cli doctor
```

Kaggle also supports its normal token/config authentication methods; the wrapper does not store credentials.

## 1. Upload/version the training data

Use a private Kaggle Dataset for the raw v3 JSON, frozen split files, and normalization maps:

```bash
python -m tools.kaggle_cli sync-data \
  --dataset YOUR_USERNAME/sentenai-absa-data
```

If the dataset exists, the wrapper creates a new version; otherwise it creates it as **private**. `--public` must be explicitly requested.

## 2. Submit one model

```bash
python -m tools.kaggle_cli run \
  --owner YOUR_USERNAME \
  --dataset YOUR_USERNAME/sentenai-absa-data \
  --model phobert
```

Default accelerator: `NvidiaTeslaT4`.

Use another currently available Kaggle accelerator if desired:

```bash
python -m tools.kaggle_cli run \
  --owner YOUR_USERNAME \
  --dataset YOUR_USERNAME/sentenai-absa-data \
  --model mdeberta \
  --accelerator NvidiaL4
```

To sync the dataset immediately before submitting:

```bash
python -m tools.kaggle_cli run \
  --owner YOUR_USERNAME \
  --dataset YOUR_USERNAME/sentenai-absa-data \
  --model phobert \
  --sync-data
```

### Submit every model in one command

`run-all` submits every model's kernel back to back, one push attempt each. If the
account's concurrent batch-GPU-session cap is full, that push is reported and skipped
(not retried) — resubmit it yourself later with `run` once a slot frees up:

```bash
python -m tools.kaggle_cli run-all \
  --owner YOUR_USERNAME \
  --dataset YOUR_USERNAME/sentenai-absa-data \
  --sync-data \
  --run-test
```

Add `--models phobert xlmr mdeberta` to run a subset instead of all 8. Accepts the same
`--use-tuned` / `--run-test` / `--accelerator` flags as `run`.

Add `--combine` to train every requested model **sequentially inside one kernel**, so the
whole batch occupies a single GPU-session slot instead of one slot per model:

```bash
python -m tools.kaggle_cli run-all \
  --owner YOUR_USERNAME \
  --dataset YOUR_USERNAME/sentenai-absa-data \
  --models mdeberta vit5 \
  --combine \
  --run-test
```

vit5 always runs last in a combined batch (its dependency pin — `transformers<5`, no
`torchao` — isn't reverted afterward). Output for each model still lands in its own
`sentenai-output/<model>/` folder, same as separate kernels.

## 3. Status and live logs

```bash
python -m tools.kaggle_cli status \
  --kernel YOUR_USERNAME/sentenai-phobert

python -m tools.kaggle_cli logs \
  --kernel YOUR_USERNAME/sentenai-phobert \
  --follow
```

## 4. Download output

```bash
python -m tools.kaggle_cli output \
  --kernel YOUR_USERNAME/sentenai-phobert \
  --output-dir .kaggle_work/outputs/phobert
```

The Kaggle script exports compact training artifacts under `sentenai-output/<model>/`, including `last.pt`, `best.pt`, metrics, metadata, and saved model/tokenizer files when produced by that model family.

## 5. Resume after a Kaggle session

```bash
python -m tools.kaggle_cli resume \
  --owner YOUR_USERNAME \
  --dataset YOUR_USERNAME/sentenai-absa-data \
  --model phobert
```

This command:

1. downloads `last.pt`/`best.pt` from the latest kernel output;
2. creates or versions `YOUR_USERNAME/sentenai-phobert-resume` as a small **private** checkpoint dataset;
3. mounts that checkpoint dataset into the next kernel;
4. copies `last.pt` back to `experiments/phobert/`;
5. launches `python -m ml.train --model phobert --resume --no-test`.

A custom checkpoint dataset can be supplied with `--resume-dataset owner/slug`.

## 6. Tuned config and final test

Use a completed equal-budget tuned config:

```bash
python -m tools.kaggle_cli run \
  --owner YOUR_USERNAME \
  --dataset YOUR_USERNAME/sentenai-absa-data \
  --model phobert \
  --use-tuned
```

Unseal test only when intentionally performing the final evaluation:

```bash
python -m tools.kaggle_cli run \
  --owner YOUR_USERNAME \
  --dataset YOUR_USERNAME/sentenai-absa-data \
  --model phobert \
  --run-test
```

## Commands

```text
python -m tools.kaggle_cli doctor
python -m tools.kaggle_cli prepare-data
python -m tools.kaggle_cli sync-data
python -m tools.kaggle_cli prepare-kernel
python -m tools.kaggle_cli run
python -m tools.kaggle_cli status
python -m tools.kaggle_cli logs
python -m tools.kaggle_cli output
python -m tools.kaggle_cli resume
```

`prepare-kernel` is useful for inspecting `.kaggle_work/kernels/<model>/kernel-metadata.json`, `run.py`, and the source bundle before actually submitting anything.

## Ingest a completed run into MLOps registry

After the kernel completes, download its compact output into the local `experiments/<model>` directory and create a `candidate` registry version:

```bash
python -m tools.kaggle_cli collect \
  --owner YOUR_USERNAME \
  --model phobert \
  --register
```

A normal remote run is validation-only, so this candidate cannot pass the production gate yet. Run the intentionally final kernel with `--run-test`, collect it, then use `python -m mlops gate ... --stage production` / `python -m mlops promote ... --stage production`.

## Troubleshooting

Issues observed in real runs, and how the wrapper (or you) should handle them:

- **`Maximum batch GPU session count of N reached` right after `kernels push`.** Your Kaggle account can only run a limited number of batch GPU kernels concurrently (commonly 1–2). Pushing another one while at that limit can leave the new kernel in a broken, unqueryable state (`status`/`logs` return 404 or "permission denied") instead of cleanly queuing. Don't push kernels back-to-back in a burst — submit one, wait for a free slot (check with `status` on your other running kernels), then submit the next.
- **A kernel shows `CANCEL_ACKNOWLEDGED` with empty logs.** This usually means the kernel was opened in Kaggle's interactive web editor/console while it was still queued or running as a batch job — opening the interactive session cancels the batch one. If you just want to watch progress, use `logs --follow` instead of opening the kernel in the browser.
- **Model names with underscores (e.g. `linear_svm`).** Kaggle kernel slugs only allow letters/digits/hyphens; the wrapper automatically hyphenates the default kernel handle (`sentenai-linear-svm`) so the title always matches the id. If you pass a custom `--kernel`, avoid underscores in it yourself.
- **ViT5 (`--model vit5`) environment quirks.** Kaggle's preinstalled `transformers` has at points shipped an early v5 release whose tokenizer backend can raise `KeyError: 0` for this checkpoint's legacy sentencepiece vocab, and `peft`'s LoRA dispatcher can raise on an incompatible preinstalled `torchao`. The wrapper's kernel bootstrap pins `transformers<5` and removes `torchao` automatically for `vit5` runs only (other models are unaffected). If Kaggle's base image changes again and this resurfaces, check `tools/kaggle_cli/cli.py`'s `kaggle_runner_source()` for the `MODEL == 'vit5'` bootstrap block.
- **Source bundle format.** The packaged source tree is uploaded to the data dataset as `sentenai_src_bundle.dat`, not `*.zip` — Kaggle silently unzips any dataset file that ends in `.zip`/`.tar`, which would otherwise scatter the bundle into hundreds of individual files instead of keeping it as one archive `run.py` can open.
