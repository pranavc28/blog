+++
title = 'From Thinking to Knowing: Using Natural Language Confidence From LLM Thought Processes'
date = 2025-10-19T07:07:07+01:00
draft = false
+++

"And this is wisdom and temperance and self-knowledge — for a man to know what he knows, and what he does not know." - Plato, Charmides

"To say you know when you know, and to say you do not when you do not — that is knowledge." - Confucius

Research code: https://github.com/pranavc28/thought-engineering-and-self-awareness

# Motivation

Despite the rise of reasoning models, limited research exists on how confident LLMs are in their reasoning processes and whether this confidence can be systematically evaluated through natural language—similar to humans. We term this capability **self-awareness**. This blog proposes a method for evaluating model self-awareness using natural language confidence scores and introduces **automated confidence refinement**, a novel framework to improve LLM accuracy in multi-classification problems. This represents the first exploration under the broader umbrella of **thought engineering** as the industry releases more powerful thinking models.

We discuss what constitutes self-awareness in LLMs, demonstrate why this capability is critical for providing context based on an LLM's reasoning, and validate this method's accuracy.

# Definition of Automated Confidence Refinement

Consider a thinking LLM given a multi-classification task (e.g., "Maybe", "Yes", or "No"). During its reasoning chain, the model exhibits varying confidence across different steps. We can extract this confidence as numerical scores (0.0-1.0) by prompting for them in natural language.

For steps with low confidence (e.g., <0.5), we can either hardcode an outcome or provide additional context to improve the model's solution. **Automated confidence refinement** systematically determines these optimal confidence thresholds per outcome in multi-classification problems. For confidence between 0.5 - 0.75 for example, we can prompt the LLM that it needs to increase the context added, for the purpose of this blog, on its own.

# Background

Current LLM reasoning relies on prompting strategies like Chain of Thought (CoT) and ensemble methods that aggregate predictions to determine consensus in classification tasks.

However, human cognition extends beyond sequential reasoning—it includes metacognitive processes that evaluate confidence. A junior employee demonstrates maturity by recognizing knowledge gaps and seeking help when uncertain. As humans acquire context through research or consultation, their reasoning refines and confidence increases. The capacity to accurately articulate confidence levels is a key indicator of competence and trustworthiness—a capability largely unexplored in LLMs.

# Related Work

Recent work from Anthropic, IBM Research AI, and Meta FAIR has explored LLM self-evaluation, metacognition (overthinking and underthinking), and self-reflection frameworks, establishing the importance of model introspection and confidence calibration.

**OptimalThinkingBench** [1] measures when LLMs "think" too much on easy questions and too little on hard ones, pairing OverthinkingBench (1,440 simple queries) with UnderthinkingBench (11 reasoning tasks). Evaluating 33 models revealed that no system balances accuracy and efficiency—"thinking" models often waste hundreds of CoT tokens without gains on trivial queries. While the industry has begun focusing on what it means to "think," this work doesn't explore natural language confidence scores or their value in evaluating self-awareness. Our blog extends this by optimizing "overthinking" to yield higher prediction accuracy.

**SELF-RAG** [2] trains models to monitor their reasoning via special "reflection tokens" (retrieve/no-retrieve, relevance, support), making retrieval on-demand and behavior controllable at inference. The model selectively allocates effort and checks claim support before proceeding, tying thinking to metacognition. We apply this concept by monitoring LLM reasoning through natural language confidence scores and providing additional context when confidence is low.

While ample research exists on thinking in LLMs, natural language confidence scores remain underexplored. As conversational interfaces and agent-to-agent communication grow, this information becomes crucial for building trust and enabling agents to exchange context like humans do.

# Problem Formulation

Reasoning-capable models (GPT-4o, GPT-5, o3, Gemini 2.5, Claude) provide visibility into step-by-step problem decomposition. Consider the query "Which continent is the US in?" with confidence-augmented reasoning:

- **Thought 1**: What are the total number of continents? → 7 continents. **Confidence: 90%**
- **Thought 2**: What are the countries per continent? → [Lists countries]. **Confidence: 40%**
- **Thought 3**: Is the US part of any continent? → Yes, South America. **Confidence: 90%**

This produces an incorrect conclusion. Without confidence scores, diagnosing where reasoning failed is difficult. However, if the model recognized its knowledge gap at Thought 2 (40% confidence), a system could automatically provide context via tool calls—such as querying a geographical database—before proceeding [3].

Accurate confidence calibration directly impacts trust. Just as senior employees trust junior colleagues who articulate knowledge boundaries, users trust LLMs that signal uncertainty reliably. Organizations developing LLMs with accurate confidence assessment will establish trust faster.

This study evaluates how confidence self-awareness has evolved across OpenAI model generations. We introduce **thought engineering**—the systematic approach to managing and measuring confidence self-awareness in LLMs—and present **automated confidence refinement**, determining optimal confidence thresholds for triggering additional context retrieval, or resorting to a default classification outcome when the natural language confidence score is too low.

# Experiment

## Dataset

Hugging face link: https://huggingface.co/datasets/allenai/scifact

We evaluate our framework on SciFact, a scientific claim verification benchmark. The dataset pairs scientific claims with research paper abstracts, requiring classification as:
- **SUPPORT**: Evidence confirms the claim
- **CONTRADICT**: Evidence refutes the claim  
- **NOINFO**: No relevant information found

We sampled 200 claims to test across multiple models. Each requires retrieving relevant papers from 5,000+ scientific abstracts and classifying the claim-evidence relationship. This two-stage task (retrieval + classification) mirrors real-world scenarios where LLMs must identify relevant context before reasoning.

## Evaluation Methodology

### Retrieval and Search Process

Unlike traditional classification where context is provided, our experiment requires LLMs to **actively query** the corpus. This tests whether models can:
1. Formulate effective search queries from claims
2. Assess if retrieved evidence is sufficient based on confidence
3. Recognize when additional retrieval is needed

We implement a simple term-overlap retrieval system (matching 4+ character words, with 2x weight for title matches). While basic, this ensures all strategies use the same retrieval backend, isolating confidence-aware reasoning's effect on query reformulation. The LLM only reformats queries and adds retrieval when aware of low confidence.

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

**Key insight**: This framework tests whether LLMs can accurately self-assess confidence and know when to seek additional context—analogous to knowing when to ask for help.

### Automated Confidence Refinement

We test five models: **o3**, **o4-mini**, **gpt-4o**, **gpt-5-mini**, and **gpt-5**. Initially, 50 claims determined ideal confidence thresholds per model. We then scaled to 200 claims for final testing.

Each prediction includes a confidence score, enabling two threshold-based optimizations:

1. **Classification Threshold**: Converts low-confidence SUPPORT/CONTRADICT predictions to NOINFO (range: 0.50-0.75)
2. **Iterative Refinement Threshold**: Triggers additional retrieval when confidence is insufficient (range: 0.70-0.95)

**Grid Search Optimization**: We perform exhaustive grid search over threshold combinations to find optimal values per model. For each combination, we apply thresholds to raw outputs, compute macro F1 score, and select the configuration maximizing F1.

This is impactful because it:
- **Removes manual tuning**: Automatically discovers optimal operating points per model
- **Accounts for model differences**: Models have different confidence calibration (o3 may be overconfident, gpt-5 well-calibrated)
- **Maximizes real-world utility**: Finds the balance between conservative (excessive NOINFO) and aggressive (wrong classifications)

Grid search tests 6 classification × 6 iterative refinement thresholds = 36 configurations per model per strategy (540 total evaluations).


#### **Automated Threshold Optimization via Grid Search**

We developed `optimize_thresholds.py` to discover optimal confidence thresholds per model via **exhaustive grid search** across classification thresholds (0.50, 0.55, 0.60, 0.65, 0.70, 0.75) and refinement thresholds (0.70, 0.75, 0.80, 0.85, 0.90, 0.95). For each of the 36 combinations, the algorithm:

1. **Applies classification threshold**: Converts predictions where `confidence < threshold` and `label ≠ NOINFO` to NOINFO
2. **Simulates iterative refinement**: Uses refined result if `initial_confidence < refinement_threshold`, otherwise falls back to naive
3. **Computes macro F1**: Calculates F1 per class (SUPPORT, CONTRADICT, NOINFO) and averages them
4. **Selects best configuration**: Identifies the threshold pair maximizing macro F1

![Grid Search Visualization](/blog/images/grid_search_diagram.png)
*Figure: Grid search for gpt-5 model exploring all threshold combinations. Each cell represents one configuration with its F1 score. The red star marks the optimal threshold pair. Darker green indicates better performance.*

This automated approach is critical because **models have fundamentally different confidence calibration**. Our optimizer discovered o3 performs best with a classification threshold of 0.60, while gpt-4o requires 0.50—suggesting o3 is overconfident while gpt-4o is better calibrated. Iterative refinement thresholds vary from 0.70 (o4-mini, o3) to 0.95 (gpt-4o), revealing some models need aggressive refinement triggers while others benefit from conservative ones. **Without optimization, manually tuning 5 models × 3 strategies × 2 thresholds = 30 configurations would be impractical** [5]. Grid search identifies each model's "sweet spot" between conservative (excessive NOINFO) and aggressive (incorrect SUPPORT/CONTRADICT), improving F1 scores by 0.01-0.03 points.

Notably, GPT-5's flatness indicates a mature, well-calibrated model requiring minimal tuning—perhaps OpenAI implements similar evaluations internally.

### Evaluation Metrics

We use **macro F1 score** as the primary metric because:

1. **Class imbalance handling**: The dataset has unequal class distribution. Accuracy biases toward the majority class, while macro F1 treats all classes equally.

2. **Precision-recall balance**: F1 rewards balancing precision (avoiding false positives) and recall (catching true positives). A model predicting only NOINFO might have high accuracy but terrible F1.

3. **Clinical relevance**: In scientific claim verification, false positives (claiming evidence when none exists) are as harmful as false negatives (missing valid evidence). F1 captures both through its harmonic mean.

4. **Per-class interpretability**: Macro F1 reveals which classes struggle, essential for diagnosing confidence calibration issues.

The experiment uses 200 parallel workers for efficient processing across 1,000 total predictions (200 claims × 5 models).

# Results

## Overall Performance (Macro F1)

![Retrieval Strategy Comparison - Bar Chart](/blog/images/thought_vs_naive_line.png)

Macro F1 trends reveal distinct patterns across model generations:

- **Older models (o3, o4-mini)**: Naive performs best (o3: 0.801, o4-mini: 0.758). Overthinking shows comparable performance (o3: 0.767, o4-mini: 0.786), while iterative refinement lags (o3: 0.736, o4-mini: 0.783).

- **Mid-tier models (gpt-4o, gpt-5-mini)**: All strategies converge (~0.73-0.75), suggesting these models struggle with confidence calibration regardless of strategy.

- **Advanced model (gpt-5)**: Iterative refinement dramatically outperforms (0.799 F1) vs. naive (0.779) and overthinking (0.752). This **2.1 point improvement** demonstrates effective self-awareness in confidence assessment.

## Per-Class Analysis

**NOINFO Classification:**
![NOINFO F1 Comparison](/blog/images/noinfo_f1_comparison.png)

NOINFO is the most challenging class, requiring models to recognize insufficient evidence:

- **o3**: Naive excels (0.786 F1), while overthinking (0.763) and iterative refinement (0.733) degrade
- **gpt-4o & gpt-5-mini**: All strategies struggle (0.68-0.75 F1) with minimal differentiation
- **gpt-5**: Iterative refinement achieves highest F1 (0.793), outperforming naive (0.774) and overthinking (0.747)

**SUPPORT Classification:**
![SUPPORT F1 Comparison](/blog/images/support_f1_comparison.png)

Support classification shows clearest evidence of improved confidence awareness:

- **o3 & o4-mini**: Naive performs well, overthinking peaks early (o4-mini: 0.800 F1)
- **gpt-4o & gpt-5-mini**: Performance plateaus around 0.75-0.78 F1 across all strategies
- **gpt-5**: Iterative refinement achieves exceptional performance (0.817 F1) vs. naive (0.797) and overthinking (0.745)

**CONTRADICT Classification:**
![CONTRADICT F1 Comparison](/blog/images/contradict_f1_comparison.png)

Contradict shows the most volatile patterns:

- **o3**: Naive dominates (0.841 F1), with overthinking (0.782) and iterative refinement (0.706) underperforming
- **o4-mini**: All strategies converge (~0.79 F1)
- **gpt-4o**: Significant drop for all approaches (0.68-0.73 F1)
- **gpt-5**: Iterative refinement rebounds strongly (0.787 F1), matching naive (0.766) and overthinking (0.764)

Per-class analysis reveals **gpt-5 with iterative refinement** achieves the most balanced performance across all classes, suggesting better confidence calibration enables knowing when additional context is needed.

# Assumptions and Caveats

This framework assumes developers have access to additional APIs/tools for adding context when the LLM has low confidence.

We used a sample size of 200 claims. Running 1,000 total requests (5 models × 200 claims) plus grid search optimization is costly—hundreds of dollars. While 1,000+ claims would be ideal, 200 provides meaningful insights. 

The key finding is not absolute performance differences between models (e.g., o3 vs. gpt-5 on naive), but rather **within-model trends**: older models (o3) perform poorly on thought engineering methods (overthinking, confidence-based refinement), while newer models (gpt-5) excel at them. This pattern strengthens with model advancement.


# **Conclusion**

Our results demonstrate a clear evolutionary trend: **as models advance, they develop more accurate self-awareness in confidence assessments**. We should not compare how each model performs against each other given that the data set size is not 1,000+ claims, instead look at the trends of naive vs overthinking vs automated confidence refinement. While older models (o3, o4-mini) performed best with naive retrieval (F1: 0.801, 0.758), gpt-5 performed best with automated confidence refinement achieved best results(F1: 0.799), outperforming naive (0.779) and overthinking (0.752).

This validates our hypothesis: **modern LLMs can be engineered to recognize insufficient reasoning and proactively seek additional context**. The automated confidence refinement framework provides a systematic mechanism for thought engineering, enabling models to operate like professionals who know when to request more information. As models improve, this metacognitive ability becomes more reliable, transforming confidence scores from noisy estimates into actionable signals for optimizing multi-classification through threshold tuning and conditional retrieval.

Notably, **overthinking exhibited high variability across models**, mirroring human cognitive bias. While overthinking improved o4-mini (+0.028 F1), it degraded o3 (-0.034 F1) and gpt-5 (-0.027 F1). This reflects how pre-reasoning introduces noise—models may overanalyze simple claims or generate overly complex search strategies missing relevant evidence [1]. Automated confidence refinement showed more stable behavior: it either matched baseline or, for gpt-5, significantly exceeded it.

This stability advantage is critical for production systems where predictable behavior matters more than occasional high performance. Effective thought engineering must account for overthinking's variability. Our framework offers a principled alternative: let models reason naturally, then assess confidence post-prediction to determine if additional context is needed. As LLMs advance, this pattern should strengthen—future models will exhibit better confidence calibration, making automated refinement a reliable strategy for high-stakes reasoning tasks requiring accuracy and self-awareness.

## **The Critical Importance of NOINFO Classification**

**NOINFO F1 score emerges as the most critical metric for evaluating confidence-aware reasoning**. It directly measures whether models **can recognize insufficient context and are aware of the situation**. Our threshold optimization framework converts low-confidence SUPPORT/CONTRADICT predictions to NOINFO, meaning NOINFO F1 captures both: (1) the model's ability to recognize genuinely insufficient evidence, and (2) confidence-based filtering preventing overconfident misclassifications.

The performance gap is striking: gpt-5 with automated confidence refinement achieves 0.793 NOINFO F1, outperforming naive (0.774) and overthinking (0.747). This 4.6-point improvement demonstrates genuine metacognitive ability—assessing evidence quality post-retrieval and determining sufficiency. In contrast, o3's naive approach dominates NOINFO (0.786), while automated refinement underperforms (0.733), suggesting earlier architectures lack reliable confidence calibration. The trend is clear: **as models advance, their NOINFO performance under automated confidence refinement improves relative to baseline strategies**, validating that self-awareness is becoming measurable, tunable, and deployable for real-world applications where admitting uncertainty is as valuable as providing answers.

Currently, ~5% improvement is modest. As time progresses, we expect this to increase. Research labs like OpenAI, DeepMind, and Anthropic should focus on what we term self-awareness.

# References

[1] - OptimalThinkingBench: Evaluating Over and Underthinking in LLMs; FAIR at Meta, Carnegie Mellon (https://arxiv.org/pdf/2508.13141v1)

[2] - SELF-RAG: LEARNING TO RETRIEVE, GENERATE, AND CRITIQUE THROUGH SELF-REFLECTION; University of Washington, Allen Institute for AI, IBM Research AI (https://arxiv.org/pdf/2310.11511)

[3] - Prompt Engineering; Weng, Lilian, (https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)

[4] - Tool Use; Weng Lilian, (https://lilianweng.github.io/posts/2023-06-23-agent/#component-three-tool-use)

[5] Grid Search, Random Search, Genetic Algorithm: A Big Comparison for NAS; Department of Computer Engineering Ternopil National Economic University, (https://arxiv.org/pdf/1912.06059)
