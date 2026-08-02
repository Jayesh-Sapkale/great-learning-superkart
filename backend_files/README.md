---
title: SuperKart Backend
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

Flask REST API serving the SuperKart sales-prediction model.
- `POST /v1/predict` — single (online) prediction
- `POST /v1/predictbatch` — batch prediction (CSV upload)
