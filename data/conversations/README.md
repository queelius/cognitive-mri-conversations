# Conversation Corpus

This directory is a placeholder for the raw conversation data used in both papers.

## Status: Sanitization in Progress

The corpus consists of 1,908 ChatGPT conversations spanning December 2022 to April 2025. These conversations contain personal information that must be removed before public release.

The sanitized corpus will be included in a future version of this compendium.

## What Will Be Released

- Conversation metadata (timestamps, titles, model used)
- Sanitized message text (personal identifiers replaced with placeholders)
- Pre-computed embeddings for each conversation

## Reproducing Published Results Without Raw Data

All published figures and tables can be reproduced from the derived data already included in this compendium:

- **Journal paper (PLOS Complex Systems):** All figures use data in `../temporal/`
- **Conference paper (Complex Networks 2025):** Parameter validation uses data in `../ablation/`

The derived data captures the complete statistical summaries; the raw conversations are only needed to regenerate embeddings from scratch or to perform new analyses not covered in the papers.
