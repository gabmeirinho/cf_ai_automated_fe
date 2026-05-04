# automated-fe — AI-assisted dataset preparation and automated feature engineering

`automated-fe` is a Cloudflare Workers AI application for exploring how AI can help with the early stages of a machine learning workflow: understanding a CSV dataset, selecting the target and usable feature columns, suggesting safe preprocessing steps, and preparing the data for modeling.

The current version focuses on **CSV profiling, AI-assisted review, deterministic train/test splitting, preprocessing, and export**. The long-term objective is to extend this foundation into a controlled **automated feature engineering system** that can generate new predictive features from existing columns.

---

## Why this project?

Feature engineering is often one of the most manual and repetitive parts of building a machine learning model.

A data scientist usually needs to:

- inspect the dataset,
- identify the prediction target,
- decide which columns are usable,
- clean inconsistent values,
- handle missing data,
- encode categorical variables,
- avoid target leakage,
- and test many possible feature transformations.

This project explores how Cloudflare Workers AI can support that process by combining:

- deterministic local profiling,
- safe preprocessing rules,
- AI-assisted dataset review,
- human-in-the-loop decisions,
- and a future automated feature generation layer.

The goal is **not** to let an LLM freely mutate the dataset. Instead, the project uses AI to guide a controlled, explainable, and auditable feature preparation workflow.

---

## Current workflow

```mermaid
flowchart LR
  A[Upload CSV] --> B[Validate file]
  B --> C[Create train/test split]
  C --> D[Profile training data]
  D --> E[AI-assisted review]
  E --> F[Apply preprocessing]
  F --> G[Export train/test CSVs]
```