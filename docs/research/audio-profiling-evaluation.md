# Audio Profiling — Fingerprinting and Voice Model Evaluation

**Purpose:** Identify the best audio fingerprinting tool for speech-dominant recordings, the optimal Whisper variant for Tier 2 hardware, and the best speaker embedding model per tier for a media identification pipeline.

**Research date:** 2026-04-07

---

## Audio Fingerprinting on Speech

### Evaluation Criteria
1. Match accuracy on speech-dominant audio with background noise
2. Robustness to re-encoding, bitrate reduction, and partial clips
3. CPU-viability (Tier 1 requirement)
4. False positive rate on similar-sounding but distinct recordings

### Technical Background: Why Chromaprint Is Unsuitable for Speech

Chromaprint is built on **chroma features** — 12-bin pitch-class representations that map spectral energy to musical semitones. This design captures harmonic and melodic patterns characteristic of music, making it highly effective for song identification across transpositions and covers.

Speech audio does not concentrate energy in stable harmonic pitch-class patterns. Instead, speech is characterized by:
- Rapidly varying fundamental frequency (F0) with no fixed pitch-class structure
- Formant transitions specific to phoneme sequences and vocal tracts
- Prosodic patterns (stress, intonation) that carry identity and content
- Aperiodic components (fricatives, stops) with no chroma representation

Chromaprint's own documentation explicitly targets "full audio file identification, duplicate audio file detection and long audio stream monitoring" for **music**. Applying chroma-based fingerprinting to speech produces features that are either absent (for unvoiced consonants) or dominated by noise rather than speaker-identity cues.

**Sources:**
- [How Chromaprint works (Lukáš Lalinský)](https://oxygene.sk/2011/01/how-does-chromaprint-work/)
- [Chroma feature Wikipedia](https://en.wikipedia.org/wiki/Chroma_feature)
- [Chromaprint AcoustID documentation](https://acoustid.org/chromaprint)

### Candidate Tool Analysis

**Chromaprint**
- Approach: Chroma feature extraction + compact fingerprint (12-bin pitch classes)
- Speech match accuracy: Poor — chroma features fail to capture speech identity cues; unvoiced phonemes produce no usable fingerprint
- Noise robustness: N/A for speech (wrong feature domain)
- Re-encoding robustness: Good for music, irrelevant for speech
- CPU viable: Yes
- Verdict: **Rejected** for speech-dominant recordings

**audfprint** (Dan Ellis / Columbia)
- Approach: Shazam-style spectral peak-pair hashing on spectrogram local maxima
- Speech match accuracy: Moderate — spectral peaks are domain-agnostic and capture energy concentrations regardless of musical vs. speech content; performs better than chroma for speech
- Noise robustness: Degrades at high distortion levels; BAF benchmark evaluation shows "poor performance at high distortion levels" in broadcast monitoring conditions; robust at low-to-moderate noise
- Re-encoding robustness: Moderate — peak locations shift under heavy bitrate reduction (below ~64kbps); good at 128kbps+
- Partial clip matching: Supported — uses time-offset voting; handles partial queries
- CPU viable: Yes — compact index (19MB per 2,000 references / 74 hours of audio), fast lookup
- Verdict: **Recommended** — strongest of the three for speech

**dejavu** (Will Drevo)
- Approach: Shazam-style spectral peak hashing (similar algorithm to audfprint, Python-native)
- Speech match accuracy: Similar to audfprint — spectral peaks work for speech but no speech-specific evaluation found
- Noise robustness: Similar limitations — degrades at high noise; no published speech-specific benchmark
- Re-encoding robustness: Similar to audfprint
- CPU viable: Yes
- Active maintenance: Lower than audfprint — less active GitHub, fewer recent commits
- Verdict: **Viable fallback** if audfprint integration is problematic; functionally equivalent but less documented for production use

### Evaluation Results

| Tool | Speech match accuracy | Noise robustness | Re-encoding robustness | Partial clip support | CPU viable | Maintenance |
|------|----------------------|------------------|----------------------|---------------------|------------|-------------|
| Chromaprint | Poor — wrong feature domain | N/A | Good (music only) | Yes | Yes | Active |
| audfprint | Moderate — spectral peaks work for speech | Degrades above moderate noise | Moderate (≥128kbps) | Yes | Yes | Active |
| dejavu | Moderate — similar to audfprint | Similar to audfprint | Moderate | Yes | Yes | Lower |

### Recommendation

**Winner: audfprint**

audfprint's Shazam-style spectral peak hashing is domain-agnostic — it fingerprints energy concentrations in the spectrogram regardless of whether the content is music or speech. Chromaprint is disqualified on technical grounds (chroma features are meaningless for speech). Dejavu is functionally equivalent to audfprint but less actively maintained.

**Important caveat:** All three are classical fingerprinting algorithms. At high distortion or significant re-encoding, all will degrade. For a pipeline where recordings may be heavily re-encoded, compressed, or partially overlapping, consider supplementing audfprint exact-match results with the speaker embedding similarity pipeline (Section 6.5) for near-duplicate detection.

**Sources:**
- [audfprint README — spectral peak approach](https://github.com/dpwe/audfprint/blob/master/README-dpwe-audfprint.txt)
- [BAF audio fingerprinting dataset evaluation (ISMIR 2022)](https://archives.ismir.net/ismir2022/paper/000109.pdf)
- [Comparative analysis of audio fingerprinting algorithms (IJCSET)](https://www.ijcset.com/docs/IJCSET17-08-05-021.pdf)

---

## Whisper Tier 2 Viability

### Evaluation Criteria
1. VRAM usage (must fit within 16GB alongside other resident models)
2. Word Error Rate (WER) on noisy/accented speech
3. Inference throughput (real-time factor on 16GB GPU)

### Model Size Reference

| Variant | Parameters | VRAM (PyTorch) | Relative speed vs. large |
|---------|-----------|----------------|--------------------------|
| small | 244M | ~2 GB | ~4× |
| medium | 769M | ~5 GB | ~2× |
| large-v3 | 1,550M | ~10 GB | 1× baseline |
| turbo (large-v3-turbo) | 809M | ~6 GB | ~8× |

### WER Benchmarks

| Variant | LibriSpeech test-clean WER | LibriSpeech test-other WER | Mixed benchmark avg WER | Notes |
|---------|---------------------------|---------------------------|------------------------|-------|
| small | ~3.4% | ~8.7% | — | Significantly degrades on noisy/accented speech |
| medium | ~3.0% | ~7.5% | — | Strong accuracy; medium.en slightly better for English-only |
| large-v3 | ~2.7% | ~6.7% | ~7.4% | Best WER; multilingual |
| turbo | ~2.9% | ~6.9% | Near large-v3 | Distilled from large-v3; near-identical accuracy at 8× speed |

### Tier 2 VRAM Budget (RTX 4060 Ti 16 GB)

Resident models at Tier 2 (approximate):
- MegaLoc VPR: ~5 GB
- ECAPA-TDNN speaker embeddings: ~0.5 GB
- DINOv2 (file dedup, if resident): ~5 GB
- **Whisper allocation: ~4–6 GB available depending on pipeline configuration**

The **turbo** variant (~6GB) and **medium** (~5GB) both fit. large-v3 (~10GB) would conflict with other resident models.

RTX 4060 Ti 16GB is specifically documented as "sweet spot" for Whisper throughput — faster than RTX 4070 Super 12GB for this task despite lower VRAM capacity.

### Recommendation

**Tier 2 variant: whisper-large-v3-turbo (turbo)**

Turbo offers near-large-v3 WER at 8× the speed and ~6GB VRAM — the best accuracy-per-VRAM tradeoff available. If turbo weights are unavailable in a specific deployment environment, **medium** is the fallback (5GB, ~2× large speed, 7.5% WER on test-other).

**Tier 1 (CPU):** small — 2GB footprint, ~4× relative speed; sufficient for transcription leads but higher WER on noisy recordings. Consider Faster-Whisper (CTranslate2) implementation for CPU deployments — 12.5× faster than OpenAI's reference implementation.

**Tier 3 variant: large-v3 confirmed** — 10GB VRAM is negligible at 96GB capacity; best available WER.

**Sources:**
- [Whisper model sizes — OpenWhispr](https://openwhispr.com/blog/whisper-model-sizes-explained)
- [Whisper VRAM discussion — HuggingFace](https://huggingface.co/openai/whisper-large-v3/discussions/83)
- [Whisper GPU benchmark — Tom's Hardware](https://www.tomshardware.com/news/whisper-audio-transcription-gpus-benchmarked)
- [faster-whisper turbo benchmark](https://github.com/SYSTRAN/faster-whisper/issues/1030)
- [OpenAI Whisper GitHub](https://github.com/openai/whisper)

---

## Speaker Embedding Model Evaluation

### Evaluation Criteria
1. Equal Error Rate (EER) on VoxCeleb benchmark
2. VRAM requirements
3. CPU-viability for Tier 1 deployment
4. Operational readiness (maintained codebase, available weights)

### Benchmark Results (VoxCeleb1-O-clean, supervised)

| Model | EER (VoxCeleb1-O-clean) | minDCF | VRAM | CPU viable (Tier 1) | Weights |
|-------|------------------------|--------|------|---------------------|---------|
| ECAPA-TDNN C=1024 (SpeechBrain) | 0.728% | 0.099 | ~0.5 GB | Yes — ~70–300ms/utterance | HuggingFace: speechbrain/spkrec-ecapa-voxceleb |
| WeSpeaker ResNet34 | 0.723% | 0.069 | ~1–2 GB | Possible but slower | GitHub: wenet-e2e/wespeaker |
| TitaNet Large (NVIDIA NeMo) | ~1.71–1.91%* | — | <4 GB (batch 4–16) | Possible | HuggingFace: nvidia/speakerverification_en_titanet_large |

*TitaNet EER measured on a curated comparative dataset, not the standard VoxCeleb1-O-clean protocol — direct comparison with the 0.72% figures is not valid.

### Notes per Model

**ECAPA-TDNN (SpeechBrain):**
- Validated specifically for forensic automatic speaker recognition (ScienceDirect 2024 study)
- TDNN architecture is much lighter than ResNet — CPU inference at 70–300ms per utterance enables Tier 1 deployment
- 4×–8× model size reduction via weight quantization with minor EER degradation (<0.16% increase)
- Available via SpeechBrain with minimal dependencies; HuggingFace weights at `speechbrain/spkrec-ecapa-voxceleb`
- EER on self-supervised (no labels): 2.627% — relevant if custom domain fine-tuning is needed without annotated data

**WeSpeaker ResNet34:**
- Marginal EER advantage over ECAPA (0.723% vs 0.728%) — within noise
- ResNet34 is computationally heavier than TDNN; CPU inference is slower and less practical at Tier 1
- Production-oriented toolkit (WeSpeaker) with score calibration and AS-Norm support
- Strong choice for Tier 2/3 where GPU is available

**TitaNet Large (NVIDIA NeMo):**
- Requires full NeMo framework — heavyweight dependency
- Higher EER on equivalent tasks; not competitive with ECAPA or ResNet34 on VoxCeleb1-O-clean
- Better suited for integration within NVIDIA NeMo diarization pipeline; standalone deployment is cumbersome

### Recommendation

**Winner: ECAPA-TDNN (SpeechBrain)**

ECAPA-TDNN offers the best combination of accuracy (EER ≈ 0.728%, statistically equivalent to ResNet34), CPU viability (70–300ms/utterance on CPU), forensic validation, and minimal deployment complexity. The SpeechBrain implementation requires no proprietary framework and provides pre-trained VoxCeleb weights directly from HuggingFace.

**Per-tier recommendation:**

| Tier | Hardware | Recommended model | Configuration |
|------|----------|-------------------|---------------|
| Tier 1 | CPU | ECAPA-TDNN (SpeechBrain) | C=512 variant for lower CPU footprint; quantized weights optional |
| Tier 2 | RTX 4060 Ti 16 GB | ECAPA-TDNN (SpeechBrain) | C=1024 full model; GPU inference <10ms/utterance |
| Tier 3 | RTX 6000 Ada 96 GB | WeSpeaker ResNet34 or ECAPA-TDNN C=1024 | Either; WeSpeaker adds AS-Norm score calibration for higher-confidence decisions |

**Sources:**
- [SpeechBrain ECAPA-TDNN HuggingFace](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- [WeSpeaker baselines for VoxSRC2023](https://arxiv.org/pdf/2306.15161)
- [Comparison of modern DL speaker verification models (MDPI 2024)](https://www.mdpi.com/2076-3417/14/4/1329)
- [ECAPA-TDNN forensic validation (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S0167639324000177)
- [TitaNet Large — HuggingFace](https://huggingface.co/nvidia/speakerverification_en_titanet_large)
- [WeSpeaker toolkit paper](https://www.fit.vut.cz/research/group/speech/public/publi/2024/wang_speech%20communication_2024.pdf)
