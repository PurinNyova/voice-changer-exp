import torch
from torch import device

from voice_changer.RVC.embedder.Embedder import Embedder


class LightHubert(Embedder):
    def loadModel(self, file: str, dev: device, isHalf: bool = True) -> Embedder:
        try:
            from lighthubert import LightHuBERT, LightHuBERTConfig
        except ImportError as exc:
            raise RuntimeError(
                "[Voice Changer][LightHuBERT] lighthubert is not installed. Install it before using the light_hubert embedder."
            ) from exc

        super().setProps("light_hubert", file, dev, isHalf)

        checkpoint = torch.load(file, map_location="cpu")
        if "cfg" not in checkpoint or "model" not in checkpoint:
            raise RuntimeError("[Voice Changer][LightHuBERT] invalid checkpoint format")

        cfg = LightHuBERTConfig(checkpoint["cfg"]["model"])
        cfg.supernet_type = "base"
        model = LightHuBERT(cfg)
        model.load_state_dict(checkpoint["model"], strict=False)
        model.set_sample_config(model.supernet.max_subnet)
        model.remove_pretraining_modules()
        model = model.eval().to(dev)
        if isHalf:
            model = model.half()

        self.model = model
        return self

    def extractFeatures(
        self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True
    ) -> torch.Tensor:
        del useFinalProj

        padding_mask = torch.zeros(feats.shape, device=self.dev, dtype=torch.bool)
        source = feats.to(self.dev)

        with torch.no_grad():
            hidden_states, _ = self.model.extract_features(
                source,
                padding_mask=padding_mask,
                mask=False,
                ret_hs=True,
                output_layer=embOutputLayer,
            )

            if embOutputLayer < 0 or embOutputLayer >= len(hidden_states):
                raise RuntimeError(
                    f"[Voice Changer][LightHuBERT] invalid output layer {embOutputLayer}. Available hidden states: {len(hidden_states)}"
                )

            features = hidden_states[embOutputLayer]

        return features.half() if self.isHalf else features.float()