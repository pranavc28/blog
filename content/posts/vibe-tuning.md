+++
title = 'Vibe Tuning: From a Dataset to a Better Model'
date = 2026-10-18T07:07:07+01:00
draft = false
math = true
+++

*"For the things we have to learn before we can do them, we learn by doing them." - **Aristotle***

**Research code:** [github.com/pranavc28/vibe-tuning](https://github.com/pranavc28/vibe-tuning)  
**LLM:** Qwen/Qwen2.5-7B-Instruct  
**Dataset:** GSM8K  
**Training:** GRPO with LoRA, running on Modal

## Purpose

In my [generative UI experiment](https://pranavc28.github.io/blog/posts/generative-ui-tinker/), I fine-tuned Qwen to produce React components using rewards for completeness, formatting, and interactivity. In my [self-awareness experiments](https://pranavc28.github.io/blog/posts/generalization-dataset-distribution/), I explored how rewards and dataset composition affect a model's willingness to say it does not have enough information.

Both involved decisions that come up again in almost every fine-tuning project. What does the data actually contain? Which training method fits it? What should the model be rewarded for? How do I know whether the resulting model is better?

I wanted to bring these decisions into a reusable workflow. I built **Vibe-Tuner**, a Python pipeline that analyzes a dataset, recommends a training strategy, runs fine-tuning, and provides guidance on where to run it. I also built a Modal workflow for training and comparing the resulting model with its base model.

On the saved GSM8K run, accuracy improved from **81.8% to 89.2% across 500 test questions**, after training on 1,000 examples for 250 GRPO steps. This is an initial result for one task. The broader question I want to explore is how much of the work around that experiment can become accessible to someone who understands their problem but has never written a training loop.

## What is vibe tuning?

With vibe coding, a user describes software they want and iterates on the generated implementation. I think a similar interface could work for adapting models: describe the behavior you want, provide examples, inspect the results, and refine the training process through feedback. Sort of by **maintaining an building your own feedback loop** ;)

For example, a support team might ask:

> Here are examples of how we resolve customer requests. Train a model to follow these policies, and ask for clarification when the information is incomplete.

That request leaves several things unresolved. Are the examples demonstrations of good responses, comparisons between better and worse responses, or conversations with measurable outcomes? Does “follow these policies” mean selecting the right action, using the right tone, or both?

A useful platform would help turn those questions into a training objective and an evaluation set. The user should be able to inspect the choices and correct them.

The current repository implements an early version of this workflow through a Python API and CLI. Strategy selection is rule-based, using properties of the dataset.

## Building the pipeline

I separated the implementation into modules for data analysis, curation, strategy recommendation, training, and platform advice. Each stage returns a structured result, so I can inspect it independently or replace it without rewriting the rest of the pipeline.

![Vibe-Tuner pipeline](/blog/images/vibe-tuning-pipeline.png)

The augmentation branch requires both a dataset flagged for augmentation and an available API key. The core orchestrator runs these stages. The Modal workflow adds dataset loading and evaluation around it; platform advice is a recommendation, not a deployment step.

### Dataset

The analyzer detects chat, instruction, and preference-pair formats. It estimates sequence lengths, removes exact duplicates, and checks whether the dataset meets the pipeline's sample-count and quality heuristics.

These checks are deliberately limited. A well-formatted answer can still be wrong, and a dataset with enough rows can still omit the situations a model needs to handle. The analyzer catches basic problems before training; it does not establish that the dataset is suitable for production.

I also implemented an optional synthetic-data curator. It generates examples from seed data, then makes a second pass to review and refine them. Format checks, duplicate filtering, diversity scoring, and an estimated spending limit help constrain the process. Note, the component for synthetic data creation needs further integration work across training methods, the GSM8K result below does not demonstrate a benefit from synthetic augmentation.

### Match the method to the feedback

The strategy recommender uses the kind of feedback available in the dataset to suggest a method:

| Available feedback | Training approach |
| --- | --- |
| Answers that can be checked automatically | GRPO with a correctness reward |
| A preferred response and a rejected response | Direct Preference Optimization, or DPO |
| Individual responses labeled good or bad | Kahneman–Tversky Optimization, or KTO |
| Demonstrations of the desired response | Supervised fine-tuning, or SFT |

The implementation returns candidate strategies, explanations, and suggested base models. These are starting points. It does not yet run competing methods and determine which one performs best on a held-out evaluation.

That is a distinction I care about. A rule can suggest GRPO for a math dataset because the final answer is verifiable. Whether GRPO is the best use of the training budget still requires an experiment.

### Run training and keep the evidence

The training engine wraps Hugging Face TRL's GRPO, DPO, KTO, and SFT trainers. It handles model loading, dataset preparation, LoRA adapters, reward functions, and saving the trained model and logs. It also supports 4-bit model loading to reduce memory requirements.

For the GSM8K workflow, I used an H100 on Modal for training and split evaluation across four A10G workers per model. The repository includes a separate evaluation script so I can rerun evaluation without repeating training.

This was useful in the saved run: the initial training output records an incomplete evaluation because some shards were missing. The subsequent standalone evaluation artifacts contain the completed 500-question comparison. A finished training job and a finished experiment are separate states, and the platform needs to make that visible.

The platform advisor also provides recommendations based on budget, local GPU availability, and training method. It currently returns guidance and setup instructions. It does not automatically provision every recommended provider or deploy an inference endpoint.

## The first experiment: math reasoning

I used GSM8K because its final numerical answers provide a relatively direct correctness signal. That makes it a useful first test for the full training and evaluation workflow.

The saved run used Qwen2.5-7B-Instruct, the first 1,000 examples from the training split, and 250 GRPO steps. The configuration sampled eight completions per prompt and combined TRL's accuracy reward with its think-format reward. Training took approximately 2.05 hours on an H100.

GRPO samples several responses to the same prompt and compares their rewards within that group. Responses with higher relative rewards receive an update that makes them more likely. Here, the accuracy reward encourages correct answers, while the formatting reward encourages the expected reasoning structure.

![GRPO training loop](/blog/images/vibe-tuning-grpo-loop.png)

Formatting is easier to measure than reasoning quality. A response can follow the requested structure and still reach the wrong answer, so final-answer accuracy remains the evaluation metric for this experiment.

I evaluated the base model and fine-tuned model on the same first 500 examples from GSM8K's test split. The evaluation code uses greedy decoding, a maximum of 2,048 generated tokens, and the same answer-extraction logic for both models.

| Model | Correct answers | Accuracy |
| --- | --- | --- |
| Qwen2.5-7B-Instruct | 409 / 500 | 81.8% |
| Qwen2.5-7B-Instruct after GRPO | 446 / 500 | 89.2% |
| Change | +37 | +7.4 percentage points |

The [saved per-question results](https://github.com/pranavc28/vibe-tuning/blob/main/run-results/eval_results.json) show more than the aggregate improvement. The fine-tuned model corrected 62 questions that the base model missed, but got 25 questions wrong that the base model had answered correctly.

I find that breakdown more useful than accuracy alone. It gives me a set of regressions to inspect and asks a practical question: which behaviors did training improve, and which did it damage?

The evaluation took about 24 minutes for the base model and 52 minutes for the fine-tuned model, measured by the slowest shard in each evaluation. These are run-level timings, with possible execution differences, rather than a controlled serving-latency benchmark. They are still a reason to measure output length and latency alongside accuracy in the next experiment.

## Companies working on related problems

Several companies already make model customization easier. Their products address different parts of the workflow I am exploring.

| Company / product | Relevant work |
| --- | --- |
| [Thinking Machines Lab / Tinker](https://tinker-docs.thinkingmachines.ai/) | A training API that lets developers control their data and training loop while remote infrastructure handles the GPU operations. I used Tinker in my earlier generative UI project. |
| [River AI](https://docs.river.ai/) | A training and serving API covering SFT, reinforcement learning, and distillation. Developers define the learning task in Python while River runs training and inference. |
| [OpenPipe / ART](https://art.openpipe.ai/getting-started/about/) | An open-source framework for training multi-turn agents with reinforcement learning, including GRPO. |
| [Fireworks AI](https://fireworks.ai/reinforcement-fine-tuning) | Managed reinforcement fine-tuning using task-specific reward signals. |
| [Together AI](https://www.together.ai/blog/introducing-fine-tuning-platform) | Managed fine-tuning workflows, including supervised training and DPO, with a web interface as well as programmatic access. |

### River AI and the learning loop

[River AI](https://docs.river.ai/) is a particularly relevant comparison because its API connects sampling, training updates, and serving. Its [RL quickstart](https://docs.river.ai/guides/rl-sync/) also uses GSM8K: the developer supplies questions and a reward function, and the framework generates responses, computes advantages, and updates the model. The example samples eight answers per question and uses group-relative advantages, so the workflow is close to the experiment in this post, although the model and optimization configuration differ.

River also documents [tool and environment integration](https://docs.river.ai/guides/rl-tools/), where a rollout can include model decisions followed by tool results. That extends the training problem beyond scoring a single answer to learning from a sequence of actions.

Vibe-Tuner currently uses TRL and Modal; River is a comparison, not an integration in this repository. My work focuses on connecting dataset inspection and strategy selection to an executable experiment. River supplies managed training and inference machinery. Both still depend on a useful learning signal and an evaluation that measures the behavior the user cares about.
