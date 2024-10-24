# Federated Unlearning 
This repo contains the implementation of the work described in [Federated Unlearning: How to Efficiently Erase a Client in FL?](https://arxiv.org/pdf/2207.05521.pdf) on the free-spoken-digit-dataset

## Table of Contents

- [Abstract](#abstract)
- [Introduction](#introduction)
- [Methods](#methods)
  - [Federated Unlearning Setup](#federated-unlearning-setup)
  - [Projected Gradient Descent (PGD)](#Projected-Gradient-Descent-(PGD))
- [Results](#results)
  
## Abstract
This project implements federated unlearning for sound data, adapting the methodology outlined in Federated Unlearning: How to Efficiently Erase a Client in FL? by Halimi et al. The focus is on evaluating the capability of unlearning the contribution of a client within federated learning environments. Unlike image data used by Halimi et al, this application is based on sound data that presents different challenges, which are addressed through specific preprocessing and analysis techniques. The results demonstrate the effectiveness of the procedure in removing a client’s contribution from the global model while minimizing both computational and communication overhead.

## Introduction
With the rise of privacy regulations such as GDPR, ensuring that machine learning systems can "forget" specific user data has become a key concern, especially in decentralized environments like Federated Learning (FL). In FL, multiple clients collaborate to train a global model without sharing raw data, making traditional unlearning methods impractical. This project investigates federated unlearning in sound data classification tasks, applying the approach to remove a client's data contribution from a global model while maintaining overall model performance and minimizing retraining costs. This introduces new challenges due to the temporal and spectral properties of sound data, which require specialized handling.

## Methods
### Federated Unlearning Setup
A federated learning model is trained using federated averaging, with one or more clients contributing data. When a client requests to be "forgotten," the project applies unlearning by reversing their contribution to the model. A backdoor is introduced to verify the effectiveness of the unlearning process.

### Projected Gradient Descent (PGD)
This unlearning process involves maximizing the client’s empirical loss within a constrained distance from a reference model. PGD projects the model back into a feasible region, ensuring that the client’s data is removed without arbitrary model divergence.

## Results
The results compare models trained from scratch to those where federated unlearning has been applied. The unlearning procedure shows that the model can recover performance similar to that of a retrained model, without the need for costly retraining. Performance is maintained across nominal test datasets while the influence of compromised or backdoor data is efficiently removed. This confirms the feasibility of federated unlearning in sound data classification tasks.
