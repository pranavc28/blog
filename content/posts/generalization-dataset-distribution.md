+++
title = 'Dataset Distribution needed for Generalization: A case study in self-awareness'
date = 2026-01-20T07:07:07+01:00
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

I situate this work within the study of self-awareness in LLMs—the capacity to recognize knowledge boundaries and calibrate outputs accordingly. Prior research shows LLMs are frequently overconfident rather than acknowledging uncertainty [1], while the VitaminC benchmark demonstrated that contrastive evidence training improves calibration by forcing attention to subtle factual distinctions [2]. I extend this by framing self-awareness as trainable: through GRPO, I incentivize correct "Not Enough Info" predictions when evidence is genuinely insufficient, directly addressing the "NA collapse" phenomenon where models exploit the neutral class to avoid penalty.

My claim that diverse training distributions are essential for self-aware generalization builds on hard example mining literature while identifying a critical limitation: training exclusively on difficult datasets fails to transfer to simpler domains. Research confirms hard examples shift model behavior toward nuanced reasoning [3], yet over-specialization degrades foundational performance [4]. I employ GRPO—computing advantages relative to sampled outputs without a separate critic [5]—to reward calibrated predictions across the full difficulty spectrum, ensuring the model develops genuine self-awareness rather than domain-specific shortcuts.

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

## Reward Function Deep Dive in GRPO

The total reward for each sampled completion is:

\[
R = R_{\text{correct}} + R_{\text{false\_na}} + R_{\text{explore}} + R_{\text{fmt}}
\]

**Component Definitions:**

\[
R_{\text{correct}} = \begin{cases} +4.0 \times w_{\hat{y}} & \text{if } \hat{y} = y \\ -0.5 \times w_{\hat{y}} & \text{if } \hat{y} \neq y \end{cases}
\]

\[
R_{\text{false\_na}} = \begin{cases} -0.5 \times w_{\text{NA}} & \text{if } \hat{y} = \text{NA} \land y \neq \text{NA} \\ 0 & \text{otherwise} \end{cases}
\]

\[
R_{\text{explore}} = \begin{cases} +8.0 & \text{if } \hat{y} \in \{\text{PASS}, \text{FAIL}\} \\ 0 & \text{if } \hat{y} = \text{NA} \end{cases}
\]

\[
R_{\text{fmt}} = \begin{cases} -0.25 & \text{if output format invalid} \\ 0 & \text{otherwise} \end{cases}
\]

---

**Where:**
- \(\hat{y}\) is the predicted label (PASS, FAIL, or NA)
- \(y\) is the ground truth label
- \(w_{\hat{y}}\) is the class weight for the predicted label

---

**Class Weights:**

| Class | Weight | Rationale |
|-------|--------|-----------|
| \(w_{\text{NA}}\) | 0.15 | Low weight discourages defaulting to "safe" NA |
| \(w_{\text{PASS}}\) | 2.5 | Higher weight—PASS requires confident assertion |
| \(w_{\text{FAIL}}\) | 3.5 | Highest weight—FAIL is hardest to predict correctly |

---

**Component Explanations:**

| Component | Purpose |
|-----------|---------|
| \(R_{\text{correct}}\) | Accuracy signal—rewards correct predictions, penalizes incorrect ones |
| \(R_{\text{false\_na}}\) | Penalizes NA predictions when evidence clearly exists |
| \(R_{\text{explore}}\) | Rewards PASS/FAIL attempts to encourage exploration |
| \(R_{\text{fmt}}\) | Penalizes invalid output format |

---

**Design Rationale.** The asymmetric class weights (NA=0.15, PASS=2.5, FAIL=3.5) combat NA collapse—the model's tendency to default to the "safe" NA prediction. The large exploration bonus (+8.0) makes attempting PASS/FAIL predictions immediately rewarding, while the gentle incorrect penalty (0.5) reduces risk aversion during exploration.

## Training Curves

## Results

## Discussion

## Conclusion

## Reference

[1] Kadavath et al. "Language Models (Mostly) Know What They Know." 2022. arXiv:2207.05221

[2] Xiong et al. "Can LLMs Express Their Uncertainty?" 2023. arXiv:2305.18153

[3] Schuster et al. "Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence." NAACL 2021. ACL Anthology

[4] VitaminC GitHub. github.com/TalSchuster/VitaminC

[5] Aly et al. "FEVEROUS: Fact Extraction and VERification Over Unstructured and Structured Information." 2021. arXiv:2106.05707

[6] DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." 2025. arXiv:2501.12948