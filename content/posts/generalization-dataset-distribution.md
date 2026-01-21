+++
title = 'Dataset Distribution needed for Generalization: A case study for self-awareness'
date = 2025-20-14T07:07:07+01:00
draft = true
math = true
+++

**Research code:** https://github.com/pranavc28/self-awareness-grpo

**LLM:** openai/gpt-oss-20b

**Datasets:** https://github.com/tdiggelm/climate-fever-dataset, https://fever.ai/download/fever/train.jsonl, https://github.com/TalSchuster/VitaminC

## Claim

Diverse training datasets improve generalization, with harder examples requiring stronger representation than easier ones to effectively shift model behavior. However, training exclusively on difficult examples fails to transfer to simpler domains—a mixture of both easy and hard examples is necessary for robust cross-domain performance.

The domain considered is self-awareness. Follow my previous blogs for motivation and background.

## Motivation

Generealization is starting to become a more spoken buzzword in ML and AI podcasts. I wanted to give it a stab, and try to formulate my own theories especially in the area of self-awareness.

## Literature Review

## Methodology

I framed this research problem as a fact verification task for self-awareness for each trained model: classifying claims as SUPPORTED (PASS), REFUTED (FAIL), or NOT ENOUGH INFO (NA) given evidence. I constructed mixed training datasets from three benchmarks in different dostributions — FEVER, VitaminC, and ClimateFEVER — to ensure cross-domain generalization.

The training set in total was balanced with approximately 460 NA, 460 PASS, and 380 FAIL examples. I used different random seeds for training (42) and evaluation (43) to ensure strict train-test separation. Each claim was paired with retrieved evidence text from Wikipedia or the respective dataset source.

To investigate the role of dataset composition in model generalization, I conducted training runs across three distinct configurations:

1. **Hard-only configuration:** Training exclusively on ClimateFEVER and VitaminC—the two datasets where the base model exhibits weaker performance—excluding the easier FEVER dataset entirely.

2. **Hard-weighted configuration:** Training on all three datasets but with distributions favoring the harder ClimateFEVER and VitaminC examples, while including a smaller proportion of FEVER examples.

3. **Balanced configuration:** Training with equal representation from all three datasets, giving each source comparable weight in the training mixture.
This experimental design enabled direct comparison of how training data diversity and difficulty distribution affect downstream performance across both challenging and straightforward fact verification domains.

I implemented Group Relative Policy Optimization (GRPO) with LoRA fine-tuning (rank 32) on a GPT-OSS-20B base model. The reward function combined four components: class-weighted accuracy rewards favoring PASS/FAIL predictions, a false-NA penalty, an exploration bonus to prevent NA collapse, and a format compliance penalty given that the model is pretrained and has not undergone supervised fine tuning for instruction following. Training ran for 200 steps with batch size 32, group size 8, and learning rate 5e-6 with AdamW optimization. I set sampling temperature to 0.3 to balance exploration with training stability, given that over this temperatre the formatting penalty would take longer to converge to 0.

For the experimental comparison, I evaluated two checkpoints: a format-only baseline (learned output structure but not accuracy) and the full GRPO model with accuracy optimization. Both models received identical prompts and classified the same 1,000 evaluation examples per dataset (500 NA, 250 PASS, 250 FAIL). I used the exact same prompt template during evaluation as during training to ensure fair comparison. **This paired design allows direct comparison of predictions on identical test instances.**

I assessed statistical significance using paired bootstrap testing (10,000 iterations) across accuracy, macro F1, and per-class F1 scores with a threshold of p < 0.01. I computed 95% confidence intervals for all metric differences and employed McNemar's test to quantify discordant predictions. This approach ensures observed improvements reflect genuine capability differences rather than sampling variance.

## Reward function deep-dive using GRPO

## Training Curves

## Results

## Discussion

## Conclusion