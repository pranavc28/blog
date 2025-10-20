+++
title = 'From Thinking to Knowing: Measuring Natural Language Confidence in LLM Thought Processes'
date = 2025-10-19T07:07:07+01:00
draft = true
+++

"And this is wisdom and temperance and self-knowledge — for a man to know what he knows, and what he does not know." - Plato, Charmides

"To say you know when you know, and to say you do not when you do not — that is knowledge." - Confucius

Research code: https://github.com/pranavc28/thought-engineering-and-self-awareness

## Motivation

With the rise of thinking and reasoning models, there appears to be limited research on how confident Large Language Models (LLMs) are in their reasoning processes and how such confidence can be systematically evaluated. This paper proposes a novel framework for enhancing model self-awareness through automated confidence refinement in post-hoc thinking. We provide a comprehensive analysis of what constitutes self-awareness in LLMs and demonstrate why this capability is critical in the context of confidence maximization and trustworthy AI systems.

## Background

Contemporary research and engineering practices typically employ prompting strategies to elicit reasoning from LLMs. Common approaches include multi-agent reasoning architectures such as Chain of Thought (CoT) and ensemble methods that aggregate predictions across multiple model invocations to determine consensus for classification tasks.

However, human cognition extends beyond sequential reasoning; it encompasses metacognitive processes that evaluate confidence in our conclusions. For instance, a junior software engineer demonstrates professional maturity by recognizing knowledge gaps and seeking additional context or consulting existing solutions when faced with uncertainty.

As humans acquire additional context—through research, consultation with peers, or examination of reference materials—their reasoning becomes more refined and their confidence in conclusions increases. Consequently, one of the most reliable indicators of competence and trustworthiness is an individual's capacity to accurately articulate their confidence level in their assertions and identify areas of uncertainty.

## Related Work

Recent work from research groups at Anthropic, IBM Research AI, and Meta FAIR has explored self-evaluation, metacognition (including phenomena of overthinking and underthinking), and self-reflection frameworks to better understand LLM reasoning processes. These foundational studies have established the importance of model introspection and confidence calibration in building reliable AI systems.

Prior research has demonstrated that LLMs can benefit from explicit self-evaluation mechanisms, though systematic frameworks for confidence-aware reasoning remain underexplored. Our work builds upon these foundations by proposing a structured approach to automated confidence refinement that enables models to recognize when additional context is necessary for reliable predictions.

## Problem Formulation

Before examining our experimental methodology, we must first establish the significance of measuring confidence in reasoning processes and its implications for practical AI systems.

The emergence of reasoning-capable models (e.g., GPT-4o, GPT-5, O3, Gemini 2.5, Claude) provides unprecedented visibility into token-level attention patterns and step-by-step problem decomposition. Consider a simple geographical query: "Which continent is the US in?" A reasoning model might generate the following thought chain:

- **Thought 1**: What are the total number of continents in the world?
- **Thought 2**: What are the countries per continent?
- **Thought 3**: Is the US part of any of those continents?

If we augment the prompt to elicit confidence scores for each reasoning step ("How confident are you in your answer for this thought?"), we might observe the following:

- **Thought 1**: What are the total number of continents in the world? → 7 continents. **Confidence: 90%**
- **Thought 2**: What are the countries per continent? → [Lists countries per continent]. **Confidence: 40%**
- **Thought 3**: Is the US part of any of those continents? → Yes, it is part of South America. **Confidence: 90%**

This reasoning chain produces an incorrect conclusion: "The US is part of South America." Without confidence scores, practitioners face significant challenges in diagnosing where the reasoning failed and how to improve the prompt or provide additional context through context engineering [1].

Confidence scores become critical in this scenario. If the model possessed accurate self-awareness and recognized its knowledge gap at Thought 2 ("What are the countries per continent?"), a system could automatically provide additional context through tool calls [2]—such as querying a geographical database or API—to resolve the uncertainty before proceeding to subsequent reasoning steps.

Accurate confidence calibration directly impacts trust in AI systems. Just as senior engineers place greater trust in junior colleagues who clearly articulate their knowledge boundaries, users will trust LLMs that reliably signal uncertainty. Organizations that develop LLMs with accurate confidence assessment capabilities will establish user trust more rapidly than those without this feature. This phenomenon reflects fundamental principles of human trust formation and their applicability to human-AI interaction.

This study evaluates how confidence self-awareness has evolved across successive generations of OpenAI models. We introduce the concept of **thought engineering**—the systematic approach to managing and measuring confidence self-awareness in LLMs. Furthermore, we present **automated confidence refinement**, a novel method that determines optimal confidence thresholds for triggering additional context retrieval during the reasoning process.

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



