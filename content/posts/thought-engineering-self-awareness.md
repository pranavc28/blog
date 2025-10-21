+++
title = 'From Thinking to Knowing: Using Natural Language Confidence From LLM Thought Processes'
date = 2025-10-19T07:07:07+01:00
draft = true
+++

"And this is wisdom and temperance and self-knowledge — for a man to know what he knows, and what he does not know." - Plato, Charmides

"To say you know when you know, and to say you do not when you do not — that is knowledge." - Confucius

Research code: https://github.com/pranavc28/thought-engineering-and-self-awareness

# Motivation

With the rise of thinking and reasoning models, there appears to be limited research on how confident Large Language Models (LLMs) are in their reasoning processes and how such confidence can be systematically prompted and evaluated as natural language - similar to humans. This idea of confidence in their reasoning is what we term **self-awareness**. This blog proposes a method for evaluating model self-awareness using natural language confidence score, and implementing a novel frameowrk that we have coined **automated confidence refinement** to improve LLM accuracy in multi-classification problems. For now, **automated confidence refinement** will be the first tool explored under the overarching umbrella of **thought engineering**, as the industry begins to release more powerful thinking models.

We provide a comprehensive discussion of what constitutes self-awareness in LLMs and demonstrate why this capability is critical in providing additional context to an LLM based on its thought. The validity and accuracy of this method is to be discussed in this blog.

# Definition of automated confidence refinement - engineering confidence scores from thought

Let's assume that a thinking LLM is given a prompt, and a query requiring the LLM to return a multi-classification outcome, such as "Maybe", "Yes", or "No". In it's chain of thought and reasoning, it will be  confident in certain steps over others. One can extract this confidence as a numerical score between say 0.0 and 1.0 by adding it as natural language to the prompt (*Discussed further in the [Background](http://localhost:1313/blog/posts/thought-engineering-self-awareness/#background) section*).

For steps where it is less confident based on a certain numerical value, e.g. less than 0.5, we will hardcode an outcome or provide more context to the LLM to help it come to a better solution.

I have proposed a method known as automated confidence refinement, to determine what these thresholds should be per outcome in a multi-classification problem.

# Background

Contemporary research and engineering practices typically employ prompting strategies to elicit reasoning from LLMs. Common approaches include multi-agent reasoning architectures such as Chain of Thought (CoT) and ensemble methods that aggregate predictions across multiple model invocations to determine consensus for classification tasks.

However, human cognition extends beyond sequential reasoning; it encompasses metacognitive processes that evaluate confidence in our conclusions. For instance, a junior employee demonstrates professional maturity by recognizing knowledge gaps and seeking additional context or consulting existing solutions when faced with uncertainty.

As humans acquire additional context—through research, consultation with peers, or examination of reference materials—their reasoning becomes more refined and their confidence in conclusions increases. Consequently, one of the most reliable indicators of competence and trustworthiness is an individual's capacity to accurately articulate their confidence level in their assertions and identify areas of uncertainty. This is something that LLMs have not been researched to be capable of, yet.

# Related Work

Recent work from research groups at Anthropic, IBM Research AI, and Meta FAIR has explored self-evaluation, metacognition (including phenomena of overthinking and underthinking), and self-reflection frameworks to better understand LLM reasoning processes. These foundational studies have established the importance of model introspection and confidence calibration in building reliable AI systems.

OptimalThinkingBench proposes a unified way to measure when LLMs “think” too much on easy questions and too little on hard ones: it pairs OverthinkingBench (1,440 simple queries across 72 domains) with UnderthinkingBench (11 procedurally generated reasoning tasks) and scores models using a thinking-adjusted accuracy curve (AUCOAA) plus standard accuracy, combined into an F1-style metric for overall “optimal thinking.” The headline finding from evaluating 33 models is that no system yet balances accuracy and efficiency: “thinking” models often spill hundreds of Chain of Thought(CoT) tokens without gains on trivial queries. The authors also test mode routing (switching between thinking/non-thinking), and prompts like “don’t overthink” — which help in narrow ways but trade off across the two benches. In short, the industry has started to focus on what it means to "think".  Nonethless, there has been no exploration in this paper around natural language confidence scores from LLMs, and if that were to be valuable in evaluating self-awareness [1].

Second, there have been discussions of frameworks to evaluate whether an LLM thinks correctly, in terms of the tokens used. This blog tries to take the idea of OverThinking a step further by finding a way to optimize "OverThinking" and yield even higher accuracy for LLM predictions.

SELF-RAG trains a single LM to decide when to retrieve, generate, and critique itself via special “reflection tokens” (e.g., retrieve/no-retrieve, relevance, support, utility), making retrieval on-demand and the model’s behavior controllable at inference time. A lightweight critic model inserts these tokens offline into training data; at test time the generator alone uses them to pick evidence, score its own segments, and produce cited, more factual outputs—beating strong RAG and even ChatGPT on several QA, verification, and long-form tasks. Conceptually, this ties “thinking in LLMs” to metacognition and adaptive computation: the model monitors its own reasoning, selectively allocates effort (retrieve vs. proceed), and checks support for claims before moving on [2]. We want to use the same idea around monitoring an LLM's reasoning, and ten providing it with additional context based on how confident it is in its own reasoning.

# Problem Formulation

Before examining our experimental methodology, we must first establish the significance of measuring confidence in reasoning processes and its implications for practical AI systems.

The emergence of reasoning-capable models (e.g., GPT-4o, GPT-5, o3, Gemini 2.5, Claude) provides unprecedented visibility into token-level attention patterns and step-by-step problem decomposition. Consider a simple geographical query: "Which continent is the US in?" A reasoning model might generate the following thought chain:

- **Thought 1**: What are the total number of continents in the world?
- **Thought 2**: What are the countries per continent?
- **Thought 3**: Is the US part of any of those continents?

If we augment the prompt to elicit confidence scores for each reasoning step ("How confident are you in your answer for this thought?"), we might observe the following:

- **Thought 1**: What are the total number of continents in the world? → 7 continents. **Confidence: 90%**
- **Thought 2**: What are the countries per continent? → [Lists countries per continent]. **Confidence: 40%**
- **Thought 3**: Is the US part of any of those continents? → Yes, it is part of South America. **Confidence: 90%**

This reasoning chain produces an incorrect conclusion: "The US is part of South America." Without confidence scores, practitioners face significant challenges in diagnosing where the reasoning failed and how to improve the prompt or provide additional context through context engineering [3].

Confidence scores become critical in this scenario. If the model possessed accurate self-awareness and recognized its knowledge gap at Thought 2 ("What are the countries per continent?"), a system could automatically provide additional context through tool calls — such as querying a geographical database or API — to resolve the uncertainty before proceeding to subsequent reasoning steps [3].

Accurate confidence calibration directly impacts trust in AI systems. Just as senior employees place greater trust in junior colleagues who clearly articulate their knowledge boundaries, users will trust LLMs that reliably signal uncertainty. Organizations that develop LLMs with accurate confidence assessment capabilities will establish user trust more rapidly than those without this feature. This phenomenon reflects fundamental principles of human trust formation and their applicability to human-AI interaction.

This study evaluates how confidence self-awareness has evolved across successive generations of OpenAI models. We introduce the concept of **thought engineering** — the systematic approach to managing and measuring confidence self-awareness in LLMs. Furthermore, we present **automated confidence refinement**, a novel method that determines optimal confidence thresholds for triggering additional context retrieval during the reasoning process.

# Experiment

## Dataset

We evaluate our framework on the SciFact dataset, a scientific claim verification benchmark. The dataset contains scientific claims paired with research paper abstracts, requiring models to classify relationships as:
- **SUPPORT**: Evidence confirms the claim
- **CONTRADICT**: Evidence refutes the claim  
- **NOINFO**: No relevant information found to support/contradict the claim

We sampled 200 claims from cross-validation folds to test across multiple models. Each claim requires both retrieving relevant papers from a corpus of 5,000+ scientific abstracts and classifying the claim-evidence relationship. This two-stage task (retrieval + classification) mirrors real-world scenarios where LLMs must first identify relevant context before reasoning about it.

## Evaluation Methodology

### Retrieval and Search Process

Unlike traditional classification tasks where context is provided, our experiment requires the LLM to **actively query** the corpus to find relevant papers. This is crucial because it tests whether the model can:
1. Formulate effective search queries from a claim
2. Assess if retrieved evidence is sufficient based on how confident it "feels" that it is
3. Recognize when additional retrieval is needed

We implement a simple term-overlap retrieval system (matching 4+ character words between queries and documents, with 2x weight for title matches). While basic, this ensures all strategies operate on the same retrieval backend, isolating the effect of confidence-aware reasoning on query reformulation rather than other more complex keyword based retrieval quality improvements. In other words, the LLM will only reformat the query if and add additional retrieval in the case that it is aware it has low confidence in its reasons.

### Three Retrieval Strategies

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

**3. Automated Confidence Refinement (Our Framework)**

Our proposed framework retrieves papers first, then assesses if evidence is sufficient:

*Step 1: Initial Retrieval* (same as naive)

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

**Key insight**: This framework directly tests whether LLMs can accurately self-assess their confidence and know when to seek additional context—analogous to a junior employee knowing when to ask for help.

### Models and Optimization

We test five models across all strategies: **o3**, **o4-mini**, **gpt-4o**, **gpt-5-mini**, and **gpt-5**. This was inititially over a set of 50 images provided as input. I used these to get the ideal thresholds for confidence per model and scaled up the next set of requests to 200 for cost optimization.

Each prediction includes a confidence score, enabling two threshold-based optimizations:

1. **Classification Threshold**: Converts low-confidence SUPPORT/CONTRADICT predictions to NOINFO (range: 0.50-0.75)
2. **Iterative Refinement Threshold**: Triggers additional retrieval when confidence is insufficient (range: 0.70-0.95)

**Grid Search Optimization**: We perform exhaustive grid search over all threshold combinations to find optimal values per model. For each combination, we:
- Apply thresholds to raw model outputs
- Compute macro F1 score
- Select the configuration maximizing F1

This is impactful because:
- **Removes manual tuning**: Automatically discovers optimal operating points per model
- **Accounts for model differences**: Different models have different confidence calibration (e.g., o3 may be overconfident, gpt-5 well-calibrated)
- **Maximizes real-world utility**: Finds the sweet spot between conservative (too much NOINFO) and aggressive (wrong classifications)

The grid search tests 6 classification thresholds × 6 iterative refinement thresholds = 36 configurations per model per strategy, totaling 540 evaluations across the experiment.

### Evaluation Metrics

We use **macro F1 score** as the primary metric rather than accuracy because:

1. **Class imbalance handling**: The dataset has unequal distribution of SUPPORT/CONTRADICT/NOINFO. Accuracy would be biased toward the majority class, while macro F1 treats all classes equally.

2. **Precision-recall balance**: F1 rewards models that balance both precision (avoiding false positives) and recall (catching true positives). A model that naively predicts NOINFO for everything might have high accuracy but terrible F1.

3. **Clinical relevance**: In scientific claim verification, false positives (claiming evidence when none exists) are as harmful as false negatives (missing valid evidence). F1 captures both error types through its harmonic mean of precision and recall.

4. **Per-class interpretability**: Macro F1 (averaging F1 across classes) reveals which classes the model struggles with, essential for diagnosing confidence calibration issues.

The experiment uses 200 parallel workers for efficient processing across the 1,000 total predictions (200 claims × 5 models).

# Results

## Overall Performance (Macro F1)

![Retrieval Strategy Comparison - Bar Chart](/blog/images/thought_vs_naive_bar.png)
![Retrieval Strategy Comparison - Line Plot](/blog/images/thought_vs_naive_line.png)

The macro F1 trends reveal distinct patterns across model generations:

- **Older models (o3, o4-mini)**: Naive performs best (o3: 0.801, o4-mini: 0.758). Overthinking shows comparable performance (o3: 0.767, o4-mini: 0.786), while iterative refinement lags behind (o3: 0.736, o4-mini: 0.783).

- **Mid-tier models (gpt-4o, gpt-5-mini)**: All strategies converge to similar performance (~0.73-0.75), suggesting these models struggle with confidence calibration regardless of strategy.

- **Advanced model (gpt-5)**: Iterative refinement dramatically outperforms (0.799 F1), with naive (0.779) and overthinking (0.752) trailing. This represents a **2.1 point improvement** over naive retrieval and demonstrates effective self-awareness in confidence assessment.

## Per-Class Analysis

**NOINFO Classification:**
![NOINFO F1 Comparison](/blog/images/noinfo_f1_comparison.png)

NOINFO is the most challenging class, requiring models to recognize when evidence is insufficient:

- **o3**: Naive excels (0.786 F1), while overthinking (0.763) and iterative refinement (0.733) show degradation
- **gpt-4o & gpt-5-mini**: All strategies struggle (0.68-0.75 F1), with minimal differentiation
- **gpt-5**: Iterative refinement achieves highest NOINFO F1 (0.793), significantly outperforming naive (0.774) and overthinking (0.747)

**SUPPORT Classification:**
![SUPPORT F1 Comparison](/blog/images/support_f1_comparison.png)

Support classification shows the clearest evidence of improved confidence awareness:

- **o3 & o4-mini**: Naive performs well, overthinking peaks early (o4-mini: 0.800 F1)
- **gpt-4o & gpt-5-mini**: Performance plateaus around 0.75-0.78 F1 across all strategies
- **gpt-5**: Iterative refinement achieves exceptional performance (0.817 F1), with naive (0.797) and overthinking (0.745) trailing

**CONTRADICT Classification:**
![CONTRADICT F1 Comparison](/blog/images/contradict_f1_comparison.png)

Contradict shows the most volatile patterns:

- **o3**: Naive dominates (0.841 F1), with overthinking (0.782) and iterative refinement (0.706) underperforming
- **o4-mini**: All strategies converge (~0.79 F1)
- **gpt-4o**: Significant drop for all approaches (0.68-0.73 F1)
- **gpt-5**: Iterative refinement rebounds strongly (0.787 F1), matching naive (0.766) and overthinking (0.764)

The per-class analysis reveals that **gpt-5 with iterative refinement** achieves the most balanced performance across all three classes, suggesting better confidence calibration enables knowing when additional context is needed versus when initial evidence suffices.

# Assumptions and Caveats

The developer has access to an additional API/tool that can be used to add more context where the LLM is not confident about the reason.

We used a sample size n of size 200. The primary reason is that it is costly to runa total of 1000 requests over 5 models, and even more so changing the evaluation methodology in the process, which requires making more LLM calls. In total, it can cost $x00 to get results. Readers wanting to experiment this on their own must be aware that researching with multiple models is costly, and should account for that :)

A sample size of 1000+ images would be ideal. Yet, 200 images should not be overlooked as insignifcant or a reason enough to ignore this. I want to emphasize that one must not see a pattern between how each label's F1 score changes per model. For example, it is not important to anlyze how o3 performs better in NAIVE classification over gpt-5. Instead, one must analyze the results where o3 performs very poorly on an overthinkig or thought engineering related methods. Yet, GPT-5 outperforms o3 when it comes to thought engineering such as overthinking and using confidence scores. It is the latter learning whihc is more frequent as can be seen with the newer models.


# Conclusion



# References

[1] - OptimalThinkingBench: Evaluating Over and Underthinking in LLMs; FAIR at Meta, Carnegie Mellon (https://arxiv.org/pdf/2508.13141v1)

[2] - SELF-RAG: LEARNING TO RETRIEVE, GENERATE, AND CRITIQUE THROUGH SELF-REFLECTION; University of Washington, Allen Institute for AI, IBM Research AI (https://arxiv.org/pdf/2310.11511)

[3] - Prompt Engineering; Weng, Lilian, (https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)

[4] - Tool Use; Weng Lilian, (https://lilianweng.github.io/posts/2023-06-23-agent/#component-three-tool-use)



