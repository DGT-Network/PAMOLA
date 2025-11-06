# PAMOLA.CORE

<p align="center">
  <img src="https://raw.githubusercontent.com/realm-inveo/pamola-core/main/docs/images/pamola_core_logo.png" alt="PAMOLA.CORE Logo" width="300"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/pamola-core/"><img alt="PyPI" src="https://img.shields.io/pypi/v/pamola-core"></a>
  <a href="https://github.com/realm-inveo/pamola-core/actions"><img alt="Build Status" src="https://github.com/realm-inveo/pamola-core/workflows/CI/badge.svg"></a>
  <a href="https://github.com/realm-inveo/pamola-core/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/realm-inveo/pamola-core"></a>
  <a href="https://pamola-core.readthedocs.io/"><img alt="Documentation" src="https://readthedocs.org/projects/pamola-core/badge/?version=latest"></a>
  <a href="https://codecov.io/gh/realm-inveo/pamola-core"><img alt="Coverage" src="https://codecov.io/gh/realm-inveo/pamola-core/branch/main/graph/badge.svg"></a>
  <a href="https://pepy.tech/project/pamola-core"><img alt="Downloads" src="https://pepy.tech/badge/pamola-core"></a>
</p>

## About the Project

**PAMOLA.CORE** is an open-source Python library for handling confidential data, developed by **Realm Inveo Inc.** and **DGT Network Inc.**. This library is part of the PAMOLA ecosystem—a professional platform for privacy-preserving data management.

**PAMOLA.CORE** provides a set of tools and processors for **data anonymization, synthetic data generation, and privacy analysis** that can be used independently of the commercial PAMOLA platform.

### PAMOLA vs. PAMOLA.CORE

- **PAMOLA** – a proprietary platform for privacy-preserving project management with a full set of features ([Product Page](https://realdata.io/pages/pamola.html)).
- **PAMOLA.CORE** – an open-source library with essential data processors and algorithms available for all.

## Features

PAMOLA.CORE includes various algorithms and techniques for:

- **Data Anonymization:**
  - Generalization (for numerical, categorical, and temporal data)
  - Suppression (data removal)
  - Masking
  - Noise addition
  - Fake data generation

- **Privacy Models:**
  - k-Anonymity
  - l-Diversity
  - t-Closeness

- **Attack Simulations:**
  - Probabilistic attacks (Fellegi-Sunter method)
  - Vector-cluster attacks (CVPL)

- **Synthetic Data Generation:**
  - PATE-GAN
  - Differential copulas
  - Other synthetic data generators

- **Cryptography:**
  - Homomorphic encryption
  - Private Set Intersection

- **Privacy and Quality Metrics**
- test

## Installation

```bash
pip install pamola-core

