---
title: iTantra STT
emoji: 🗣️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Offline speech-to-text for 10 Indian languages, mixed freely
---

# iTantra STT

Speak Hindi, Gujarati, Marathi, Kannada, Malayalam, Tamil, Telugu, Odia, Bengali or English, switching
freely, and read everything in one language. Every model is INT8 and runs on CPU; on a phone or laptop the same
engine works fully offline. This Space runs it on Hugging Face's free CPU tier (2 vCPU), so it is slower than on
an edge device.

Pipeline: Silero VAD → AmberNet language ID ‖ IndicConformer encoder → RNN-T decoder (Whisper for English) →
IndicTrans2 translation → reorder buffer.
