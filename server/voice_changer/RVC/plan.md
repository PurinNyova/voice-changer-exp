# Plan: Replace HuBERT With LightHuBERT for RVCv2

## Goal

Replace the current HuBERT-based feature extractor used by RVCv2 inference with LightHuBERT while preserving the existing RVC pipeline contract:

- input audio is resampled to 16 kHz
- the embedder returns frame features through `Embedder.extractFeatures(...)`
- RVCv2 continues to use 768-channel features, `embOutputLayer = 12`, and `useFinalProj = False`

## Current RVCv2 Path

The current inference path is:

1. `RVCr2.inference()` in `server/voice_changer/RVC/RVCr2.py`
2. `createPipeline()` in `server/voice_changer/RVC/pipeline/PipelineGenerator.py`
3. `Pipeline.exec()` in `server/voice_changer/RVC/pipeline/Pipeline.py`
4. `Pipeline.extractFeatures()` in `server/voice_changer/RVC/pipeline/Pipeline.py`
5. `EmbedderManager.getEmbedder()` in `server/voice_changer/RVC/embedder/EmbedderManager.py`
6. `FairseqHubert.extractFeatures()` in `server/voice_changer/RVC/embedder/FairseqHubert.py`

For official RVCv2 models, `RVCModelSlotGenerator` sets:

- `embedder = "hubert_base"`
- `embOutputLayer = 12`
- `useFinalProj = False`
- `embChannels = 768`

## Constraints

LightHuBERT cannot be treated as a drop-in replacement until these are verified:

- the checkpoint format can be loaded by the current fairseq-based code, or a new loader is required
- the extracted hidden state shape matches current RVCv2 expectations
- frame stride is compatible with the downstream interpolation logic in `Pipeline.exec()`
- the chosen output layer maps cleanly to the current `embOutputLayer` field
- half precision behavior remains stable on CUDA and does not regress CPU fallback

## LightHuBERT
The LightHuBERT tutorial support features as follows.

Load different checkpoints to the LightHuBERT architecture, such as base supernet, small supernet, and stage 1.
Sample a subnet and set the subnet.
Conduct inference of the subnet with a random input.

```python
import sys
sys.path.append("../")

import torch
from lighthubert import LightHuBERT, LightHuBERTConfig

device="cuda:2"
wav_input_16khz = torch.randn(1,10000).to(device)
```

LightHuBERT Base Supernet

```python
# checkpoint = torch.load('/path/to/lighthubert_base.pt')
checkpoint = torch.load('/workspace/projects/lighthubert/checkpoints/lighthubert_base.pt')
cfg = LightHuBERTConfig(checkpoint['cfg']['model'])
cfg.supernet_type = 'base'
model = LightHuBERT(cfg)
model = model.to(device)
model = model.eval()
print(model.load_state_dict(checkpoint['model'], strict=False))

# (optional) set a subnet
subnet = model.supernet.sample_subnet()
model.set_sample_config(subnet)
params = model.calc_sampled_param_num()
print(f"subnet (Params {params / 1e6:.0f}M) | {subnet}")

# extract the the representation of last layer
rep = model.extract_features(wav_input_16khz)[0]

# extract the the representation of each layer
hs = model.extract_features(wav_input_16khz, ret_hs=True)[0]

print(f"Representation at bottom hidden states: {torch.allclose(rep, hs[-1])}")
```

## Implementation Strategy

### Phase 1: Compatibility spike

Objective: determine whether LightHuBERT can fit behind the existing `Embedder` interface with minimal pipeline changes.

Tasks:

1. Obtain the target LightHuBERT checkpoint and identify its runtime dependency.
2. Confirm whether it is:
   - fairseq-compatible
   - Hugging Face-compatible
   - custom code requiring its own wrapper
3. Run a small probe on a 16 kHz waveform and record:
   - output tensor shape
   - hidden size
   - frame rate / sequence length
   - selectable layer indices
4. Decide whether RVCv2 can reuse `embOutputLayer = 12` and `useFinalProj = False`, or needs a translation layer.

Exit criteria:

- one documented compatibility decision: `drop-in`, `adapter-needed`, or `not-compatible`

### Phase 2: Add a dedicated LightHuBERT embedder

Objective: isolate the new encoder behind the existing embedder abstraction instead of modifying pipeline logic first.

Files to add or change:

- add `server/voice_changer/RVC/embedder/LightHubert.py`
- update `server/voice_changer/RVC/embedder/EmbedderManager.py`
- update `server/const.py`
- update `server/voice_changer/utils/VoiceChangerParams.py`

Tasks:

1. Create `LightHubert` implementing the existing `EmbedderProtocol`:
   - `loadModel(file, dev, isHalf=True)`
   - `extractFeatures(feats, embOutputLayer=9, useFinalProj=True)`
2. Keep the public behavior aligned with `FairseqHubert` so `Pipeline.extractFeatures()` does not need to know which encoder is active.
3. Extend `EmbedderType` with a new value such as `"light_hubert"`.
4. Update `EmbedderManager.loadEmbedder()` to instantiate `LightHubert` for that type.
5. Add a new path in `VoiceChangerParams` for the LightHuBERT checkpoint.

Notes:

- Do not replace `FairseqHubert` in place initially.
- Keep the old HuBERT path available behind a feature flag until parity is verified.

### Phase 3: Wire model and runtime configuration

Objective: make LightHuBERT selectable without breaking existing models or boot flow.

Files to change:

- `server/MMVCServerSIO.py`
- `server/downloader/WeightDownloader.py`
- optionally `server/data/ModelSlot.py`
- optionally `server/voice_changer/RVC/RVCModelSlotGenerator.py`

Tasks:

1. Add a CLI/runtime parameter for the LightHuBERT checkpoint path.
2. Decide whether LightHuBERT is:
   - opt-in only for selected models, or
   - the new default for all RVCv2 models
3. If model metadata should drive selection, allow `RVCModelSlot.embedder` to carry `"light_hubert"`.
4. If boot-time auto-download is desired, add a downloader entry for the LightHuBERT weight.
5. Preserve backward compatibility so existing `hubert_base` model slots continue to load.

Recommended first rollout:

- keep model-slot default as `hubert_base`
- enable LightHuBERT only when explicitly configured
- switch defaults only after measured parity is acceptable

### Phase 4: Validate RVCv2 feature compatibility

Objective: prove that LightHuBERT features are acceptable for the downstream generator.

Validation checks:

1. Unit-level embedder validation
   - model loads on CPU and CUDA
   - `extractFeatures()` returns a tensor on the expected device
   - output dtype is stable for full and half precision
2. Pipeline-level validation
   - `Pipeline.exec()` completes without shape errors
   - interpolation and optional index search still work
   - f0 and no-f0 RVCv2 paths both run
3. Regression validation against current HuBERT
   - compare feature tensor shape for identical audio
   - compare sequence length before and after `F.interpolate(...)`
   - compare inference latency and peak memory
4. Runtime validation
   - load at least one official RVCv2 model
   - verify live conversion does not produce silence, NaNs, or severe artifacts

Acceptance target:

- no pipeline shape regressions
- no new half-precision failures
- acceptable quality and latency relative to current HuBERT baseline

### Phase 5: Replace the old default

Objective: move from optional support to actual replacement.

Tasks:

1. Change the default embedder routing in `EmbedderManager` only after Phase 4 passes.
2. Update `RVCModelSlotGenerator` defaults for official RVCv2 models if LightHuBERT becomes the new standard.
3. Keep a rollback switch so the old HuBERT path can be re-enabled quickly.
4. Update any user-facing docs or startup examples that mention `hubert_base.pt`.

## Recommended Order of Work

1. Compatibility spike for one LightHuBERT checkpoint.
2. Add a new `LightHubert` wrapper behind the existing embedder interface.
3. Add runtime parameter plumbing.
4. Validate end-to-end on one RVCv2 model.
5. Benchmark quality, latency, and memory.
6. Only then change defaults.

## Risks

- hidden size or layer semantics may not match RVCv2 training expectations
- frame timing may differ and break downstream interpolation assumptions
- a new dependency stack may be required if the checkpoint is not fairseq-native
- half precision may fail even if the current HuBERT path is stable
- index search quality may degrade if feature distribution shifts significantly

## Decision Points

These should be answered before replacing the default path:

1. Is the target LightHuBERT checkpoint fairseq-native or not?
2. Does it produce 768-dim features at a usable layer for RVCv2?
3. Should replacement mean:
   - full swap of `hubert_base`, or
   - introducing `light_hubert` as a separate embedder option?
4. Do existing trained RVCv2 models remain acceptable with LightHuBERT features, or is retraining required for quality parity?

## Recommended Outcome

Implement LightHuBERT as a new embedder option first, validate it end-to-end in the existing RVCv2 pipeline, and only replace the old HuBERT default after measured compatibility is confirmed.