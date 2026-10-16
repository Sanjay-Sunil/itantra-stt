# VAD gain control (measured 2026-10-16)

Test: English clip, Hindi clip, then an English clip ~60x quieter (RMS 0.0014 vs 0.08), 1 s gaps (raw FLEURS levels).
Noise: 20 s white hiss (σ 0.003); 20 s low rumble (smoothed noise). Silero VAD, default segmenter settings.

| Release (per 32 ms) | Quiet clip (20.8–34.7 s) detected | False segments on hiss / rumble |
|---|---|---|
| no gain control | not detected | 0 / 0 |
| 0.995 (~6 s) | 27.6–33.3 s (partial) | 0 / 0 |
| 0.99 | 24.3–33.4 s | 0 / 0 |
| **0.97 (~1 s), shipped** (30x cap, see follow-up) | **22.9–33.3 s** | **0 / 0** |

Gain applies to the VAD input only; ASR and LID get the raw audio. Ad-hoc script in session; small synthetic test.

## Follow-up: gain cap and boost-only

With gain allowed below 1 (attenuation) or capped at 30x, a Tamil segment on the level-matched demo started
0.3 s late and lost its first word ("நீங்கள்"). Fix: boost only (gain ≥ 1) and cap at 10x.

| max gain | quiet clip | hiss | demo segment at ~37 s starts (no AGC: 37.5 s) |
|---|---|---|---|
| 5 | 23.1–33.3 (split) | none | 37.2 s |
| **10 (shipped)** | **22.9–33.3** | **none** | **37.2 s** |
| 15 | 22.9–33.3 | none | 37.2 s |
| 20 | 22.9–33.3 | none | 37.7 s (first word lost) |
