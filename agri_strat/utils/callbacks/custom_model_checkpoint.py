from typing import Dict, Any

from lightning.pytorch.callbacks import ModelCheckpoint


class CustomModelCheckpoint(ModelCheckpoint):
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        dirpath_from_ckpt = state_dict.get("dirpath", self.dirpath)

        if self.dirpath == dirpath_from_ckpt:
            self.kth_best_model_path = state_dict.get("kth_best_model_path", self.kth_best_model_path)
            self.kth_value = state_dict.get("kth_value", self.kth_value)
            self.best_k_models = state_dict.get("best_k_models", self.best_k_models)
            self.last_model_path = state_dict.get("last_model_path", self.last_model_path)

        self.best_model_score = state_dict["best_model_score"]
        self.best_model_path = state_dict["best_model_path"]
