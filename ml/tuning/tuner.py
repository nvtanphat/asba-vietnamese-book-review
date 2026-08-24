from __future__ import annotations
import copy, json, shutil
from pathlib import Path
import yaml

from ml.train import ROOT, ensure_splits, load_config
from ml.data import TARGET_COLS, load_splits
from ml.models import build_model
from ml.evaluation.calibration import calibrate_absent_thresholds, decode_probabilities
from ml.evaluation.metrics import evaluate_predictions
from ml.tuning.search_space import suggest
from ml.utils.seed import seed_everything
from mlops.lineage import dataset_snapshot
from mlops.tracking import ExperimentTracker


def tune(model_name: str, trials: int = 20):
    try: import optuna
    except ImportError as exc: raise RuntimeError("Install optuna to use tuning: pip install optuna") from exc
    base=load_config(model_name,False);data_root=ROOT/base.get("data_root","data/splits");ensure_splits(data_root);train,val,_=load_splits(data_root);yt=train[TARGET_COLS].to_numpy(int);yv=val[TARGET_COLS].to_numpy(int)
    work=ROOT/f"experiments/tuning/{model_name}";work.mkdir(parents=True,exist_ok=True)
    lineage=dataset_snapshot(ROOT/"data/raw/tiki-book-review_merged_fixed_v3.json",root=ROOT,split_manifest=data_root/"split_manifest.json")
    tracker=ExperimentTracker(experiment="sentenai-absa-tuning",run_name=f"{model_name}-{trials}trials",root=ROOT).start()
    tracker.log_params({"model":model_name,"trial_budget":int(trials),"base_config":base})
    tracker.set_tags({"model":model_name,"data_fingerprint":lineage.get("fingerprint"),"test_policy":"sealed"})
    def objective(trial):
        cfg=copy.deepcopy(base);cfg.update(suggest(trial,model_name));cfg["seed"]=int(base.get("seed",42));seed_everything(cfg["seed"]);trial_dir=work/f"trial_{trial.number:03d}"
        model=build_model(model_name,cfg);model.fit(train["text"].tolist(),yt,val["text"].tolist(),yv,output_dir=trial_dir,resume=False);p=model.predict_proba(val["text"].tolist());th=calibrate_absent_thresholds(p,yv);pred=decode_probabilities(p,th);score=evaluate_predictions(yv,pred)["f1_combined"]
        shutil.rmtree(trial_dir, ignore_errors=True)
        return score
    study=optuna.create_study(direction="maximize",study_name=f"sentenai-{model_name}",storage=f"sqlite:///{work/'study.db'}",load_if_exists=True)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    remaining = max(0, int(trials) - len(complete))
    if remaining:
        study.optimize(objective, n_trials=remaining)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError(f"No successful tuning trial for {model_name}")
    best=dict(study.best_params)
    (work/"best_config.yaml").write_text(yaml.safe_dump(best,sort_keys=False),encoding="utf-8")
    manifest={"model":model_name,"trial_budget":int(trials),"completed_trials":len(complete),"best_value":float(study.best_value),"best_trial":int(study.best_trial.number),"best_params":best}
    (work/"tuning_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    tracker.log_metrics({"best_value":float(study.best_value),"completed_trials":len(complete)})
    tracker.log_params({"best_params":best,"best_trial":int(study.best_trial.number)})
    tracker.log_artifact(work/"tuning_manifest.json");tracker.log_artifact(work/"best_config.yaml");tracker.finish("FINISHED")
    print("best",study.best_value,best);return best
