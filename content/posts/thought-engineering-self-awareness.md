+++
title = 'Are LLMs "self-aware" in their thought process?'
date = 2025-10-19T07:07:07+01:00
draft = true
+++

"And this is wisdom and temperance and self-knowledge — for a man to know what he knows, and what he does not know." - Plato, Charmides

"To say you know when you know, and to say you do not when you do not — that is knowledge." - Confucius

Research code: https://github.com/pranavc28/thought-engineering-and-self-awareness

## Motivation

With the rise of thinking and reasoning models, there does not seem to be much research on how confident an LLM is in its thought - or how one can test that. This blog aims to propose a novel framework for these models regarding self-awareness: automated confidence refin post hoc thinking. Also, we will have a deep dive on what it means for an LLM to be self-aware, and why this matters in the context of what I propose as confidence maximization.

## Background

Most researchers and engineers typically prompt LLMs to reason/think. They may also use several muti-agent reasoning architectures, such as Chain of Thought (CoT), or multi shot LLM calls picking the most frequent answer for a multi classification problem.

However, as humans, we do not simply involve thinking as a multi-step process, breaking down each reason to a new step. We also spend time analyzing how confident we are in our thoughts. For example, a junior software engineer is expected to know when to ask for more context when solving a project, or look for exisitng solutions when solving their projects in situations that they are not as confident.

As we add more context to our thoughts, for example new websites to scrape solutions from or from talking to other people, our thought process becomes more succinct, and we gain more confidence in our solutions. Thus, one of the best ways to build trust and gage the effectiveness of an individual is in their ability to articulate their condience in their proposition towards a thought.

## Past research and inspiration

In the past, we can see that several researchers at Anthropic, IBM research AI, and FAIR @ Meta have proposed self-evaluation, metacoginition (Overthinking and underthinking), and self-reflection frameworks to better understand how LLMs think.

(Add context from previous papers here: https://arxiv.org/pdf/2508.13141v1, https://arxiv.org/pdf/2207.05221, file:///Users/pranavchaudhary/Desktop/Canada%20B1:B2%202026/2310.11511v1.pdf)

Talk about lilian wang's blog as well

## Problem overview and generalization

Before we dive into the experiment, and how it was conducted, one may argue about the impact of measuring confidence in thought. Why does this matter?

With the rise of thinking models, such as gpt-5, o3, gemini 2.5, claude etc.. we have access to which tokens the LLM focuses on, and how it breaks the problem down into several steps. For example, if one were to prompt ChatGPT: Which continent is the US in? One could expect the following thought chain:

Thought 1 - What are the total number of continents in the world?
Thought 2 - What are the countries per continent?
Thought 3 - Is the US part of any of those continents?

Let's say I also add confidence scores as a natural language question in part of the prompts: "How confident are you in your answer for this thought?" This would mean, I could expect the following responses:

Thought 1 - What are the total number of continents in the world? How confident am I in my answer for this thought?
Thought 1 - 7 continents in the world. 90% confident. 
Thought 2 - What are the countries per continent? How confident am I in my answer for this thought?
Thought 2 - [*List all the countries per continent]. 40% confident.
Thought 3 - Is the US part of any of those continents? How confident am I in my answer for this thought?
Answer 3 - Yes, it is part of South America. 90% confident.

The answer could be: The US is part of south-america. However, we know that to be incorrect. As a developer, I would not be sure where the LLM went wrong, and how to improve the prompt/add more context per context engineering [1].

This is where the confidence scores **could** be highly impactful. If the LLM was truly self-aware, and knew where it lacked context or data to come to a conclusion, which in our case is thought 2 or "What are the countries per continent?" I could provide it with a tool [2] such as a wesbite or an API call to another service that lists all the countries per continent.

By being more accurate in its confidence scores, developers will without doubt trust that LLM more. In fact, we see this in modern economies and marketplaces already. A senior engineer will trust a junior engineer more if they are very vocal about what they know/don't know. The same will apply to workforces of agents being maintained by developers/other roles. The company that can accurately relay confidence in thought for its LLM will gain the trust of their users faster than those that cannot build this ability. In some sense, this blog post is more around the metaphysical or psychological nature of how humans gain trust in other people - and how this should be applied to LLMs.

This experiment evaluates how self-awareness in confidence has changed with new models developed by OpenAI. I will also propose calling idea of handling and measuring confidence self-awareness in an LLM as thought engineering. I will also highlight a method that I developed called automated confidence refinement to determine at which confidence thereshold, for a reason, an additional tool call could be made to add more context for the prompt.

## Experiment

### Dataset

We evaluate our framework on the SciFact dataset, a scientific claim verification benchmark. The dataset contains scientific claims paired with research paper abstracts, requiring models to classify relationships as:
- **SUPPORT**: Evidence confirms the claim
- **CONTRADICT**: Evidence refutes the claim  
- **NOINFO**: No relevant information found

We sampled 200 claims from cross-validation folds to test across multiple models. Each claim requires both retrieving relevant papers from a corpus of 5,000+ scientific abstracts and classifying the claim-evidence relationship. This two-stage task (retrieval + classification) mirrors real-world scenarios where LLMs must first identify relevant context before reasoning about it.

### Evaluation Methodology

#### Retrieval and Search Process

Unlike traditional classification tasks where context is provided, our experiment requires the LLM to **actively query** the corpus to find relevant papers. This is crucial because it tests whether the model can:
1. Formulate effective search queries from a claim
2. Assess if retrieved evidence is sufficient
3. Recognize when additional retrieval is needed

We implement a simple term-overlap retrieval system (matching 4+ character words between queries and documents, with 2x weight for title matches). While basic, this ensures all strategies operate on the same retrieval backend, isolating the effect of confidence-aware reasoning rather than retrieval quality.

#### Three Retrieval Strategies

We compare three retrieval strategies to test how confidence-aware reasoning affects performance:

**1. Naive: Direct Query Generation**

The model generates search queries without confidence assessment:

```
Generate 2-3 scientific search queries to find research papers that could verify or refute this claim.

Claim: \{claim\}

Focus on:
- Key entities (genes, proteins, drugs, diseases)
- Core mechanisms or relationships mentioned
- Specific phenomena or effects

Return JSON: \{"search_queries": ["query1", "query2", "query3"]\}
```

After retrieving papers, classification proceeds:

```
Verify this scientific claim using the provided evidence.

Claim: \{claim\}

Evidence from papers:
\{retrieved_papers\}

Classify as:
- SUPPORT: The papers provide evidence that confirms the claim
- CONTRADICT: The papers provide evidence that refutes the claim
- NOINFO: The papers don't contain relevant information about this specific claim

Return JSON: \{"label": "SUPPORT"|"CONTRADICT"|"NOINFO", "confidence": 0.0-1.0, "justification": "explanation"\}
```

**2. Overthinking: Pre-Reasoning Before Retrieval**

The model analyzes the claim before searching, adapting strategy based on initial confidence:

```
You're a scientific claim verifier. Before searching, analyze this claim.

Claim: \{claim\}

Think step-by-step:
1. What is this claim asserting? What are the key concepts?
2. What evidence would SUPPORT this? What would CONTRADICT it?
3. Initial confidence in finding relevant evidence (0.0-1.0)
4. Search strategy:
   - If confidence < 0.5: Generate 3-4 broad exploratory queries
   - If confidence 0.5-0.8: Generate 2-3 targeted queries
   - If confidence > 0.8: Generate 1-2 precise queries

Return JSON: \{"reasoning": "your analysis", "initial_confidence": 0.0-1.0, "search_queries": [...]\}
```

This tests whether pre-reasoning about confidence improves retrieval quality.

**3. Post-hoc: Automated Confidence Refinement (Our Framework)**

Our proposed framework retrieves papers first, then assesses if evidence is sufficient:

*Step 1: Initial Retrieval* (same as NAIVE)

*Step 2: Confidence Assessment*

```
You retrieved these papers for the claim. Assess if they provide sufficient evidence.

Claim: \{claim\}

Retrieved Papers:
\{paper_previews\}

Evaluate:
1. Confidence these papers can verify/refute the claim (0.0 = need more, 1.0 = sufficient)
2. What critical information is MISSING to properly assess the claim?
3. Should additional papers be retrieved? (yes only if confidence < 0.7)

Return JSON: \{"confidence": 0.0-1.0, "need_refinement": "yes"|"no", "gaps": "what's missing"\}
```

*Step 3: Conditional Refinement*

If confidence is below the model-specific threshold AND refinement is needed:

```
Initial retrieval was insufficient. Missing information: \{identified_gaps\}

Claim: \{claim\}

Generate 1-2 NEW refined search queries to fill these specific gaps.
Focus on what's missing, not what you already retrieved.

Return JSON: \{"refined_queries": ["query1", "query2"]\}
```

**Key insight**: This framework directly tests whether LLMs can accurately self-assess their confidence and know when to seek additional context—analogous to a junior engineer knowing when to ask for help.

#### Models and Optimization

We test five models across all strategies: **o3**, **o4-mini**, **gpt-4o**, **gpt-5-mini**, and **gpt-5**.

Each prediction includes a confidence score, enabling two threshold-based optimizations:

1. **Classification Threshold**: Converts low-confidence SUPPORT/CONTRADICT predictions to NOINFO (range: 0.50-0.75)
2. **Post-hoc Refinement Threshold**: Triggers additional retrieval when confidence is insufficient (range: 0.70-0.95)

**Grid Search Optimization**: We perform exhaustive grid search over all threshold combinations to find optimal values per model. For each combination, we:
- Apply thresholds to raw model outputs
- Compute macro F1 score
- Select the configuration maximizing F1

This is impactful because:
- **Removes manual tuning**: Automatically discovers optimal operating points per model
- **Accounts for model differences**: Different models have different confidence calibration (e.g., o3 may be overconfident, gpt-5 well-calibrated)
- **Maximizes real-world utility**: Finds the sweet spot between conservative (too much NOINFO) and aggressive (wrong classifications)

The grid search tests 6 classification thresholds × 6 post-hoc thresholds = 36 configurations per model per strategy, totaling 540 evaluations across the experiment.

#### Evaluation Metrics

We use **macro F1 score** as the primary metric rather than accuracy because:

1. **Class imbalance handling**: The dataset has unequal distribution of SUPPORT/CONTRADICT/NOINFO. Accuracy would be biased toward the majority class, while macro F1 treats all classes equally.

2. **Precision-recall balance**: F1 rewards models that balance both precision (avoiding false positives) and recall (catching true positives). A model that naively predicts NOINFO for everything might have high accuracy but terrible F1.

3. **Clinical relevance**: In scientific claim verification, false positives (claiming evidence when none exists) are as harmful as false negatives (missing valid evidence). F1 captures both error types through its harmonic mean of precision and recall.

4. **Per-class interpretability**: Macro F1 (averaging F1 across classes) reveals which classes the model struggles with, essential for diagnosing confidence calibration issues.

The experiment uses 200 parallel workers for efficient processing across the 1,000 total predictions (200 claims × 5 models).

### Results

#### Overall Performance (Macro F1)

![Retrieval Strategy Comparison - Bar Chart](/blog/images/thought_vs_naive_bar.png)
![Retrieval Strategy Comparison - Line Plot](/blog/images/thought_vs_naive_line.png)

The macro F1 trends reveal distinct patterns across model generations:

- **Older models (o3, o4-mini)**: NAIVE performs best (o3: 0.801, o4-mini: 0.758). Overthinking shows comparable performance (o3: 0.767, o4-mini: 0.786), while post-hoc lags behind (o3: 0.736, o4-mini: 0.783).

- **Mid-tier models (gpt-4o, gpt-5-mini)**: All strategies converge to similar performance (~0.73-0.75), suggesting these models struggle with confidence calibration regardless of strategy.

- **Advanced model (gpt-5)**: POST-HOC dramatically outperforms (0.799 F1), with NAIVE (0.779) and OVERTHINKING (0.752) trailing. This represents a **2.1 point improvement** over naive retrieval and demonstrates effective self-awareness in confidence assessment.

#### Per-Class Analysis

**NOINFO Classification:**
![NOINFO F1 Comparison](/blog/images/noinfo_f1_comparison.png)

NOINFO is the most challenging class, requiring models to recognize when evidence is insufficient:

- **o3**: NAIVE excels (0.786 F1), while OVERTHINKING (0.763) and POST-HOC (0.733) show degradation
- **gpt-4o & gpt-5-mini**: All strategies struggle (0.68-0.75 F1), with minimal differentiation
- **gpt-5**: POST-HOC achieves highest NOINFO F1 (0.793), significantly outperforming NAIVE (0.774) and OVERTHINKING (0.747)

**SUPPORT Classification:**
![SUPPORT F1 Comparison](/blog/images/support_f1_comparison.png)

Support classification shows the clearest evidence of improved confidence awareness:

- **o3 & o4-mini**: NAIVE performs well, OVERTHINKING peaks early (o4-mini: 0.800 F1)
- **gpt-4o & gpt-5-mini**: Performance plateaus around 0.75-0.78 F1 across all strategies
- **gpt-5**: POST-HOC achieves exceptional performance (0.817 F1), with NAIVE (0.797) and OVERTHINKING (0.745) trailing

**CONTRADICT Classification:**
![CONTRADICT F1 Comparison](/blog/images/contradict_f1_comparison.png)

Contradict shows the most volatile patterns:

- **o3**: NAIVE dominates (0.841 F1), with OVERTHINKING (0.782) and POST-HOC (0.706) underperforming
- **o4-mini**: All strategies converge (~0.79 F1)
- **gpt-4o**: Significant drop for all approaches (0.68-0.73 F1)
- **gpt-5**: POST-HOC rebounds strongly (0.787 F1), matching NAIVE (0.766) and OVERTHINKING (0.764)

The per-class analysis reveals that **gpt-5 with POST-HOC** achieves the most balanced performance across all three classes, suggesting better confidence calibration enables knowing when additional context is needed versus when initial evidence suffices.

## Assumptions

The developer has access to an additional API/tool that can be used to add more context where the LLM is not confident about the reason.

## References

[1] - Context engineering

[2] - Tools in LLMs



