# Portfolio Code Repository

## Overview
This repository contains the source code used in my portfolio projects.  
For detailed explanations, please visit:<br><br>
🔗 https://giyonglee.com


## Projects

### Uplift Modeling
X-Learner-based Uplift Modeling for Advertising Incremental Effect Estimation

#### Project info
- Directory: `/uplift`  
- Portfolio: [광고 증분 효과 추정을 위한 X-learner 기반의 업리프트 모델 구현 및 실험](https://giyonglee.com/posts/5)
- Dataset: [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)

#### Workflow
1. `data.py` – data preprocessing
2. `train.py` – XGBoost model training
3. `evaluate.py` – model evaluation
4. `uplift.py` – X-Learner uplift estimation
5. `report.py` – uplift performance evaluation

### Multi-Touch Attribution
Comparison of Attention-based GRU and Transformer Models for Multi-Touch Attribution

#### Project info
- Directory: `/attribution`  
- Portfolio: [Attention 메커니즘을 활용한 GRU 및 Transformer 기반 멀티 터치 어트리뷰션 모델 비교 연구](https://giyonglee.com/posts/4) 
- Dataset: [Criteo Attribution Modeling for Bidding Dataset](https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/)  
  
#### Workflow
1. `data.py` – data preprocessing  
2. `optimize.py` – hyperparameter optimization (GRU / Transformer)  
3. `train.py` – final model training with optimized parameters  
4. `attribution.py` – conversion prediction  
5. `evaluate.py` – prediction performance evaluation  
6. `report.py` – attention-based attribution analysis  


## Experiments

### Tracking validation
- Simulation scripts for validating GA4/GTM event collection
- Checks page views, scroll, click, engagement time, traffic parameter
- Directory: `/traffic_simulation`
- workflow:
  1. `utils.py` - common utilities for request handling and event configuration
  2. `steps.py` - defines simulated user interaction steps
  3. `simulate.py` - executes traffic simulation and triggers events

### A/B Test Statistical Analysis
- Exploratory comparison of statistical inference approaches for A/B testing.
- Comparison of Neyman–Pearson and Bayesian sequential approaches for A/B test result interpretation.
- Neyman–Pearson 
  - Directory: `/ab_neyman` 
  - Dataset: simulated dataset (`dataset/fake_ab_test_2000.csv`)
- Bayesian sequential 
  - Directory: `/ab_beysian`
  - Dataset: [ASOS Digital Experiments Dataset](https://osf.io/64jsb/overview)

