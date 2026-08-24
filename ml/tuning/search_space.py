from __future__ import annotations


def suggest(trial, model: str) -> dict:
    if model in {"logistic","linear_svm"}:
        return {"C": trial.suggest_float("C", 0.1, 10.0, log=True)}
    if model == "textcnn":
        return {"embedding_dim":trial.suggest_categorical("embedding_dim",[128,200,300]),"channels":trial.suggest_categorical("channels",[96,128,192]),"dropout":trial.suggest_float("dropout",0.2,0.5),"lr":trial.suggest_float("lr",3e-4,2e-3,log=True)}
    if model == "bilstm":
        return {"hidden_dim":trial.suggest_categorical("hidden_dim",[128,192,256]),"dropout":trial.suggest_float("dropout",0.2,0.5),"lr":trial.suggest_float("lr",3e-4,2e-3,log=True)}
    if model in {"phobert","xlmr","mdeberta"}:
        return {"lr":trial.suggest_float("lr",1e-5,5e-5,log=True),"dropout":trial.suggest_float("dropout",0.1,0.3),"weight_decay":trial.suggest_float("weight_decay",0.0,0.08)}
    if model == "vit5":
        return {"lr":trial.suggest_float("lr",5e-5,4e-4,log=True),"lora_r":trial.suggest_categorical("lora_r",[4,8,16]),"lora_dropout":trial.suggest_float("lora_dropout",0.0,0.15)}
    return {}
