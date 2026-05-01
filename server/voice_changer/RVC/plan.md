# Plan: Replace The Aging HiFi-GAN-Style RVC Vocoder With Vocos

## Objective

Replace the current embedded NSF-HiFi-GAN-style decoder used by the RVC inference path with a Vocos-based vocoder stack that is easier to maintain, produces competitive quality, and can support a cleaner future architecture.

This is a migration plan, not a direct one-step swap. The current RVC decoder is tightly coupled to the model checkpoint and to F0 injection, while Vocos expects acoustic features such as mel-spectrograms or EnCodec-style features.

## Current State

The active RVC runtime path is:

1. `RVCr2.initialize()` builds the pipeline.
2. `PipelineGenerator.createPipeline()` loads inferencer, embedder, pitch extractor, and optional FAISS index.
3. `RVCInferencer` or `RVCInferencerv2` loads `SynthesizerTrnMs256NSFsid` or `SynthesizerTrnMs768NSFsid`.
4. `SynthesizerTrn*.infer()` calls `self.dec.infer_realtime(...)`.
5. `self.dec` is `GeneratorNSF`, an embedded NSF-HiFi-GAN-style decoder.

Important consequence:

- RVC does not call the standalone `pretrain/nsf_hifigan/model` path used by Diffusion-SVC and DDSP-SVC.
- The vocoder is embedded in the RVC model definition and checkpoint format.
- F0 is injected directly into the decoder through the NSF source path.

## Vocos Facts That Matter

From `Vocos readme.md`:

- Vocos is an inference-time neural vocoder that decodes acoustic features in a single forward pass.
- The documented pretrained mel model is `charactr/vocos-mel-24khz`.
- Its example mel input shape is `B, C, T`, with `256` mel channels.
- The documented pretrained models are 24 kHz, not 32 kHz, 40 kHz, or 48 kHz.
- Vocos is designed as a vocoder stage, not as a full replacement for the upstream RVC acoustic model.

## Core Architectural Gap

This is the main blocker and should drive the plan.

The current RVC model outputs waveform through `GeneratorNSF` using:

- latent features from the text encoder and flow
- speaker conditioning
- direct F0 conditioning inside the decoder

Vocos does not consume that interface. Vocos expects acoustic features, typically mel-spectrograms, and the README examples use a 24 kHz mel model with 256 mel bins.

That means replacing the current decoder requires one of these architecture changes:

1. Add an acoustic head that predicts Vocos-compatible mel features, then decode those features with Vocos.
2. Train a new RVC decoder stack where the current NSF waveform decoder is replaced by a mel predictor plus Vocos.
3. Keep the existing RVC model frozen and build a bridge module that converts its internal latent representation into Vocos mel space.

Option 1 is the most maintainable target. Option 3 is the fastest experiment path.

## Recommended Strategy

Use a phased migration with a compatibility bridge first, then move to a native architecture.

### Phase 1: Feasibility Prototype

Goal:

- Prove whether Vocos can produce acceptable output quality and latency in the RVC serving model.

Tasks:

1. Introduce a new vocoder abstraction layer under `server/voice_changer/RVC` so the pipeline no longer assumes `GeneratorNSF` is the only waveform decoder.
2. Add a `VocosVocoder` wrapper that can:
	- lazily load `Vocos.from_pretrained(...)`
	- run on the selected device
	- accept mel input in `B, C, T`
	- return waveform tensors compatible with the existing pipeline output handling
3. Use the documented pretrained model `charactr/vocos-mel-24khz` for the first experiment.
4. Build an offline test harness first, outside the realtime path.
5. Compare:
	- decode latency
	- output length alignment
	- artifact rate
	- speaker similarity

Deliverable:

- A standalone script or isolated prototype path that can decode mel features with Vocos and write audio for comparison.

### Phase 2: Bridge Model Experiment

Goal:

- Determine the cheapest path to produce Vocos-compatible features from the existing RVC model.

Tasks:

1. Inspect the tensor right before `GeneratorNSF.infer_realtime()`.
2. Decide whether the bridge input should be:
	- `z * x_mask`
	- speaker-conditioned hidden states
	- another intermediate representation earlier in `SynthesizerTrn*.infer()`
3. Add a small acoustic projection head that predicts mel features expected by Vocos.
4. Train or fine-tune that projection head against target mel-spectrograms generated from paired training audio.
5. Keep the original NSF path available as a baseline and fallback.

Deliverable:

- A branch where the RVC acoustic model can produce mel features and Vocos can decode them offline.

### Phase 3: Sample Rate Strategy

Goal:

- Resolve the mismatch between existing RVC model sample rates and the available Vocos pretrained model.

Tasks:

1. Treat 24 kHz as the initial target because that is what the README documents.
2. For 32 kHz, 40 kHz, and 48 kHz models, choose one of these paths:
	- temporary resample-down to 24 kHz before Vocos and resample-up after decode
	- train or fine-tune Vocos for the repository's actual target sample rates
	- maintain separate vocoder backends by sample rate
3. Measure quality loss from resampling before committing to a training effort.

Recommendation:

- Start with resampling only for prototype validation.
- Do not lock the product architecture to 24 kHz resampling if the quality regression is noticeable.

Deliverable:

- A short comparison table covering quality, latency, and implementation complexity for each sample-rate option.

### Phase 4: Realtime Integration

Goal:

- Integrate Vocos into the low-latency inference path without regressing streaming behavior.

Tasks:

1. Add a runtime-selectable vocoder backend in the RVC pipeline settings.
2. Preserve the current buffering, padding, and truncation logic in `Pipeline.exec()` while testing Vocos decode chunking behavior.
3. Verify whether Vocos can operate chunk-by-chunk without audible seams.
4. If chunk seams appear, add overlap-add or context padding around mel windows.
5. Validate GPU memory usage and half-precision behavior separately from the current NSF path.

Deliverable:

- A feature-flagged Vocos backend selectable per model or by server setting.

### Phase 5: Model Format And Deployment

Goal:

- Make Vocos a supported and maintainable backend in this repository.

Tasks:

1. Extend model metadata so RVC slots can declare vocoder type.
2. Decide how Vocos weights are stored:
	- Hugging Face download at runtime
	- vendored local checkpoint
	- explicit model asset in `model_dir`
3. Add dependency management for `vocos`.
4. Document fallback behavior when Vocos is unavailable.
5. Keep ONNX export scope explicit. Do not assume the existing RVC ONNX exporter can export a Vocos-backed path.

Deliverable:

- A documented packaging and loading strategy.

## Concrete Code Areas To Touch

Expected implementation surface:

- `server/voice_changer/RVC/pipeline/Pipeline.py`
- `server/voice_changer/RVC/pipeline/PipelineGenerator.py`
- `server/voice_changer/RVC/inferencer/InferencerManager.py`
- `server/voice_changer/RVC/inferencer/RVCInferencer.py`
- `server/voice_changer/RVC/inferencer/RVCInferencerv2.py`
- `server/voice_changer/RVC/inferencer/rvc_models/infer_pack/models.py`
- `server/voice_changer/RVC/RVCSettings.py`
- model slot metadata and loader paths under the server model-management layer

Likely new files:

- `server/voice_changer/RVC/vocoder/VocosVocoder.py`
- `server/voice_changer/RVC/vocoder/VocoderProtocol.py`
- a prototype or evaluation script for mel decode experiments

## Risks

1. Vocos is not a drop-in waveform decoder for the current latent interface.
2. The pretrained README path is 24 kHz only, while this repo serves multiple higher sample rates.
3. The current decoder receives explicit F0 conditioning. Vocos itself does not. F0 influence must be preserved upstream in the acoustic feature prediction path.
4. Realtime chunking behavior may be worse than the current specialized `infer_realtime()` path.
5. Existing ONNX export support likely will not carry over without separate work.

## Decision Gates

Do not proceed to full replacement unless all of these are true:

1. Offline Vocos decode quality is competitive with the current NSF decoder.
2. The bridge or acoustic-head approach preserves speaker identity and pitch stability.
3. Realtime latency remains within acceptable limits for the existing server UX.
4. The sample-rate story is acceptable without excessive resampling artifacts.
5. Packaging and runtime loading are simple enough to support in production.

## Acceptance Criteria

The migration can be considered successful when:

1. An RVC model can run with either the legacy NSF decoder or the new Vocos backend.
2. The Vocos path is selectable without changing unrelated RVC pipeline behavior.
3. Audio quality is at least neutral or better on representative models.
4. End-to-end latency remains acceptable for realtime conversion.
5. Documentation clearly explains supported sample rates, dependencies, and fallback behavior.

## Recommended First Implementation Order

1. Add a standalone Vocos inference wrapper and offline decode test.
2. Build a bridge experiment that predicts Vocos mel features from existing RVC internal states.
3. Measure 24 kHz prototype quality and latency.
4. Decide whether to invest in resampling, fine-tuning, or training a native Vocos-compatible acoustic head.
5. Only then wire a feature-flagged Vocos backend into the realtime pipeline.
