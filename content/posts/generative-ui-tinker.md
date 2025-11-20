+++
title = 'Tinkering with Generative UI'
date = 2025-11-14T07:07:07+01:00
draft = true
math = true
+++

*"Form and function should be one, joined in a spiritual union" - Frank Lloyd Wright*

Research code: https://github.com/pranavc28/generative-ui

## Purpose

This blog explores how I fine tuned Qwen, Qwen/Qwen3-30B-A3B, an open source model producing React code for a fraction of the cost of any of the top-tier AI research lab. I used [Tinker](http://tinker-docs.thinkingmachines.ai/) to fine tune my own LLM, a product released by the team at [Thinking Machines Lab](https://thinkingmachines.ai/).

It did a great job, and produced generally usable React code to the level that I expected it to from an online dataset.

In this blog, I explain what fine tuning means, why Generative UI can/should be fine tuned, and left some of my thoughts on tinker.

## Background

The intersection of artificial intelligence and user interface development has reached a fascinating inflection point. For decades, building user interfaces required manual coding—developers translating design specifications into HTML, CSS, and JavaScript line by line. More recently, component libraries and frameworks like React have standardized this process, but the fundamental paradigm remained unchanged: humans write code, computers execute it.

Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, but their application to UI development faces unique challenges. Unlike backend logic or algorithmic problems where correctness is binary, UI code must balance multiple competing objectives: syntactic correctness, visual appeal, interactivity, accessibility, and alignment with user intent.

Enter reinforcement learning. By treating UI generation as a sequential decision-making problem where the model receives rewards for producing high-quality interfaces, we can guide LLMs toward generating not just syntactically correct code, but truly functional, interactive, and well-structured user interfaces. This represents a fundamental shift from imitation learning to reward-driven optimization.

## Why Generative UI and fine tuning?

Why invest effort in fine-tuning models specifically for generative UI? The answer lies in understanding the unique requirements of UI code generation versus general-purpose coding: Structural Completeness, Interactivity and Dynamics, Visual and Semantic Coherence, and Domain-Specific Patterns.

Traditional supervised fine-tuning trains models on input-output pairs, essentially teaching imitation. But what if the training data contains suboptimal examples? What if there are multiple valid solutions with different quality levels? Supervised learning can't distinguish between them—it treats all training examples as equally correct.

Reinforcement learning, specifically Proximal Policy Optimization (PPO), addresses these limitations by defining explicit reward functions that encode our preferences. We can reward complete code more than truncated outputs, interactive components more than static ones, and idiomatic patterns more than unusual but technically correct alternatives. The model learns through exploration and feedback, discovering better solutions than those in the training data.

## Definition of Generative UI

**Generative UI** refers to the automated creation of user interface code through machine learning models, where the system generates complete, functional interface implementations from natural language descriptions or specifications.

More formally, generative UI can be defined as a conditional generation task:

Given:
- A specification \\( S \\) (user intent, design requirements, functional specifications)
- A domain \\( D \\) (React, Vue, HTML/CSS, etc.)
- A set of constraints \\( C \\) (style guidelines, accessibility requirements, component libraries)

Generate:
- A complete code artifact \\( U \\) that:
  - Compiles/parses without errors
  - Renders a visual interface matching \\( S \\)
  - Includes appropriate interactivity and state management
  - Follows idiomatic patterns for domain \\( D \\)
  - Satisfies constraints \\( C \\)

This differs from code completion or snippet generation in crucial ways:

**Completeness**: Generative UI produces entire, self-contained components or applications, not fragments requiring human integration.

**Functional Correctness**: The generated code must actually work when executed, not just look plausible to static analysis.

**Visual Grounding**: The code's visual output matters as much as its textual content—an interface that compiles but looks broken has failed its purpose.

**Context Awareness**: Generative UI systems must understand the broader context of UI development: component composition, state flow, event handling, styling approaches, and accessibility considerations.

In our implementation, we focus on React/TypeScript generation, treating each generation as a mapping from `(system_prompt, user_message) → React component code`. The model must learn the implicit constraints of React development: proper JSX syntax, hook usage rules, event handler patterns, and TypeScript type annotations.

## Fine-Tuning and PPO: Why Reinforcement Learning for Generative UI

Fine-tuning adapts a pre-trained language model to a specific domain or task by continuing training on targeted data. For generative UI, we face a critical question: *should we use supervised learning or reinforcement learning?*

### The Supervised Learning Baseline

Supervised learning optimizes a straightforward objective—minimize the cross-entropy loss between predicted and target tokens:

$$\mathcal{L}\_{\text{SL}}(\theta) = -\mathbb{E}\_{x \sim \mathcal{D}}\left[\sum\_{t} \log p\_\theta(x\_t \mid x\_{<t})\right]$$

Where \\( \theta \\) are model parameters, \\( \mathcal{D} \\) is the training dataset, \\( x_t \\) is the token at position \\( t \\), and \\( x_{<t} \\) is the context (all previous tokens).

This approach teaches the model to maximize the likelihood of the exact training sequences. It's simple, stable, and works well when training data represents optimal behavior. However, it has fundamental limitations for UI generation:

1. **Exposure Bias**: During training, the model always sees correct previous tokens; at inference, its own potentially incorrect predictions compound.
2. **Loss-Evaluation Mismatch**: Cross-entropy loss treats all token prediction errors equally, but in UI code, a mismatched brace is far worse than suboptimal variable naming.
3. **No Exploration**: The model can only learn from provided examples, never discovering alternative (potentially better) solutions.

### Proximal Policy Optimization (PPO)

PPO is a policy gradient reinforcement learning algorithm designed for stable, efficient policy optimization. Instead of maximizing likelihood of specific sequences, PPO maximizes expected reward:

$$\mathcal{J}(\theta) = \mathbb{E}\_{\tau \sim \pi\_\theta}\left[R(\tau)\right]$$

Where:
- \\( \pi_\theta \\) is the policy (our language model)
- \\( \tau \\) is a trajectory (generated token sequence)
- \\( R(\tau) \\) is the total reward

The core innovation of PPO is the **clipped surrogate objective**, which ensures stable policy updates:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

Where:
- \\( r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \\) is the importance sampling ratio
- \\( \hat{A}_t \\) is the advantage estimate (how much better this action is than average)
- \\( \epsilon \\) is the clipping parameter (typically 0.2)
- \\( a_t \\) and \\( s_t \\) are the action (token) and state (context) at time \\( t \\)

**Why does clipping matter?** Without clipping, the ratio \\( r_t(\theta) \\) could become arbitrarily large, causing unstable policy updates. If the new policy makes an action much more likely than the old policy (\\( r_t \gg 1 \\)), we could overshoot the optimal policy. Clipping bounds this ratio to \\( [1-\epsilon, 1+\epsilon] \\), ensuring conservative updates.

The gradient of this objective is:

$$\nabla\_\theta \mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}\_t\left[\nabla\_\theta \log \pi\_\theta(a\_t \mid s\_t) \cdot \min\left(\hat{A}\_t, \text{clip}(\hat{A}\_t, -\epsilon\hat{A}\_t^+, \epsilon\hat{A}\_t^-)\right)\right]$$

This gradient increases the probability of actions with positive advantages (good outcomes) and decreases probability of actions with negative advantages (bad outcomes), but only to a limited extent determined by the clipping.

### Why PPO for Generative UI?

PPO addresses the limitations of supervised learning in three critical ways:

**1. Reward-Based Optimization**  
We define a reward function that explicitly encodes UI quality:

$$R(\text{code}) = R_{\text{base}} + w_1 R_{\text{complete}} + w_2 R_{\text{valid}} + w_3 R_{\text{interactive}} + w_4 R_{\text{quotes}} - w_5 \text{penalty}_{\text{length}}$$

Where:
- \\( R_{\text{complete}} \\): Reward for structural completeness (matched braces, proper endings)
- \\( R_{\text{valid}} \\): Reward for syntactic validity (balanced delimiters, proper exports)
- \\( R_{\text{interactive}} \\): Reward for interactive features (state hooks, event handlers)
- \\( R_{\text{quotes}} \\): Reward for balanced quotes and strings
- \\( \text{penalty}_{\text{length}} \\): Penalty for excessive length deviation from reference

This **multi-objective reward function captures the nuanced requirements of UI code** that cross-entropy loss cannot express.

**2. Exploration and Discovery**  
By sampling multiple completions per prompt (\\( k \\) samples), the model explores the solution space. Good solutions receive positive advantages, bad solutions receive negative advantages. Over time, the policy shifts toward generating better code:


**3. Stability Through Clipping**  
The clipping mechanism prevents catastrophic policy updates. Even if we sample an unusually good or bad trajectory, the policy update is bounded. This is crucial for fine-tuning pre-trained models where we want to adapt behavior, not destroy existing knowledge.

## Implementation: Asynchronous PPO Training with Tinker

Implementing PPO for large language models traditionally requires extensive infrastructure: distributed training systems, GPU orchestration, gradient accumulation strategies, and sophisticated async coordination. **Tinker** abstracts this complexity, letting us focus on the algorithm while it handles distributed execution.

### Algorithm: Asynchronous PPO Training with Tinker

My implementation follows the standard PPO training loop with asynchronous sampling and gradient computation:

The key innovation is asynchronous execution. Instead of synchronously sampling one prompt at a time, we launch all sampling requests concurrently, process results as they complete, and overlap computation phases.

Below is an image of the general asynchronous calls that were computed to allow tinker to work efficiently.

![Asynchronous PPO Algorithm](/blog/images/ppo_tinker.png)

{{< center >}}Algorithm 1: Asynchronous PPO for Generative UI using Tinker API{{< /center >}}

### Reward function design

The reward function is compositional:

```python
def compute_reward(generated_code, reference_code):
    R_base = 1.0
    
    # Completeness: severely punish truncation
    if is_truncated(generated_code):
        R_complete = -15.0
    else:
        R_complete = +7.5
    
    # Validity: reward balanced delimiters
    R_valid = 0.0
    R_valid += 1.8 if balanced_braces(generated_code) else -3.0
    R_valid += 0.9 if balanced_brackets(generated_code) else -1.5
    R_valid += 0.9 if balanced_parens(generated_code) else -1.5
    R_valid += 1.2 if has_return_or_export(generated_code) else 0
    
    # Interactivity: reward React patterns
    R_interactive = 0.0
    R_interactive += 1.25 if 'useState' in generated_code else 0
    R_interactive += 0.75 if 'useEffect' in generated_code else 0
    R_interactive += count_event_handlers(generated_code) * 0.25
    R_interactive += 0.5 if has_conditional_rendering(generated_code) else 0
    
    # Quotes: reward balanced strings
    R_quotes = 0.0
    R_quotes += 1.6 if balanced_single_quotes(generated_code) else -2.4
    R_quotes += 1.2 if balanced_double_quotes(generated_code) else -2.0
    
    # Length penalty: discourage excessive verbosity
    penalty_length = 0.1 * abs(len(generated_code) - len(reference_code)) / len(reference_code)
    
    return R_base + R_complete + R_valid + R_interactive + R_quotes - penalty_length
```

**Advantage Construction**  
The advantage array is crucial. We assign zero advantage to prompt tokens (we don't train on them) and distribute the trajectory reward across generated tokens:

```python
def create_advantage_array(prompt_length, gen_length, reward):
    prompt_advantages = [0.0] * (prompt_length - 1)
    gen_advantages = [reward] * gen_length
    return prompt_advantages + gen_advantages
```

This ensures gradients only flow through generated portions from the LLM, not the fixed prompt context.

**Hyperparameter Tuning**  
Our configuration:
- Learning rate: \\( 10^{-5} \\) (small to prevent catastrophic forgetting)
- Samples per prompt: 4 (exploration-exploitation balance)
- Clip epsilon: 0.2 (standard PPO value)
- Epochs: 5 (sufficient for convergence on 600 examples)
- Max generation tokens: 16,000 (React components can be long)

**Data Filtering**  
We filter prompts exceeding 16k tokens during initialization to leave room for generation within the 32k context window. This prevents out-of-memory errors and truncated generations.

## Results when comparing the fine-tuned model to raw Qwen model samples

To evaluate the effectiveness of PPO fine-tuning, I tested both the raw Qwen model and the fine-tuned version on three distinct UI generation tasks: booking a ride interface, a Google search homepage, and a leaderboard display. The differences are striking and directly reflect the reward signals defined in our training objective.

### Use Case 1: Ride Booking Interface

**Raw Model Performance**

![Raw Qwen Transport Interface Failure](/blog/images/sample_qwen_transport_fail.png)


The raw model's output exhibits the most critical failure: **Incomplete code generation**: the component is truncated mid-function, missing closing braces and return statements.

**Fine-Tuned Model Performance**

![Fine-Tuned Qwen Transport Interface](/blog/images/fine_tune_qwen_easy_rider_transport.png)

![Fine-Tuned Live Ride Booking](/blog/images/fine_tune_ride_book_live.png)


The fine-tuned model demonstrates substantial improvement:
- **Structural completeness**: Fully formed component with proper exports and matched delimiters
- **Rich interactivity**: Multiple `useState` hooks managing ride selection, pickup/dropoff locations, and booking confirmation
- **Event handler coverage**: onClick handlers for ride type selection, onChange for input fields, onSubmit for booking confirmation
- **Conditional rendering**: Dynamic UI showing different states (selecting ride, confirming booking, success message)

The fine-tuned model generates not just syntactically correct code, but a **functionally complete, interactive booking flow** that responds to user actions—exactly what our reward function incentivized. Given the addition of a new computational reward variable for dynamic code, we can see that the LLM picks up on this use case for code generation.

### Use Case 2: Google Search Homepage

**Raw Model Performance**

![Raw Qwen Google Homepage](/blog/images/qwen_sample_google_homepage.png)

The raw model produces a basic structure but falls short:
- **Limited interactivity**: Search input exists but lacks proper state management
- **Missing event handlers**: No onSubmit handler for search functionality, no onChange for input updates
- **Static elements**: Buttons and links that don't respond to user interaction
- **Truncation issues**: May cut off mid-styling or mid-component definition

**Fine-Tuned Model Performance**

![Fine-Tuned Google Search Homepage](/blog/images/fine_tune_google_search_works.png)

The fine-tuned model excels:
- **Complete state management**: `useState` hook managing search query input
- **Functional search**: Proper form submission with `onSubmit` handler, and clear search query processing even if the results are hardcoded
- **Reactive UI**: Input field updates in real-time with `onChange` handler, controlled component pattern
- **Visual polish**: Complete styling that matches the Google aesthetic, properly balanced quotes in className strings
- **Balanced delimiters**: All braces, brackets, and parentheses properly matched

The reward function's emphasis on interactivity (\\( R_{\text{interactive}} \\)) is clearly visible—the fine-tuned model doesn't just render a static homepage mockup, it generates a **functional search interface with proper React patterns**.

### Use Case 3: Leaderboard Display

**Raw Model Performance**

![Raw Qwen Leaderboard](/blog/images/raw_leaderboard.png)

The raw model's leaderboard suffers from:
- **Incomplete rendering**: May truncate the list of players or miss closing tags
- **Static data**: Hardcoded player list with no dynamic updates or sorting
- **Poor data structure**: Inconsistent formatting or missing key player attributes
- **No interactivity**: Unable to filter, sort, or update leaderboard dynamically

**Fine-Tuned Model Performance** *(inferred from training patterns)*

The fine-tuned model generates improved leaderboards with:
- **Complete table structure**: All rows and columns properly closed, balanced JSX elements
- **State-driven rendering**: Uses `useState` to manage leaderboard data, enabling dynamic updates. Also updates leaderboards in realtime as a demo as time progresses.
- **Interactive features**: Sorting by clicking column headers, filtering by player name, expandable rows for player details
- **Conditional rendering**: Highlights for top players, empty state handling when no data exists
- **Proper data mapping**: `.map()` with keys, proper TypeScript typing for player objects

In this case, I was super interested with the model picking up real time updates using time. This was not something that I explicitly prompted the LLM. Yet, it reasoned from previous examples that real time updates could be valuable.

## Thoughts on Tinker


## Conclusion

Generative UI represents a paradigm shift in interface development—from manual coding to specification-to-implementation via machine learning. By fine-tuning large language models with Proximal Policy Optimization, we can teach models not just to imitate existing code, but to discover and generate high-quality, interactive, complete user interfaces.

I have focused more on the length of the React code output, and how dynamic it is. Users can target other features such as tailwind/design parity, or code structure.

The combination of PPO's reward-driven learning and Tinker's abstraction of distributed training infrastructure makes this approach practical for researchers and developers. We define our reward function (what makes good UI code), specify our training loop (how to explore and update), and Tinker handles the complexity of distributed execution across GPUs effectively.
