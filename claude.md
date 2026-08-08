# Project Collaboration Guide

This project is an educational ML project for a possible internship evaluation. The goal is not only to finish the implementation, but also for the user to understand the system well enough to explain and defend design choices.

## Collaboration Style

- Do not work on autopilot.
- Keep the user involved in important decisions: project scope, dataset choice, model choice, evaluation metrics, architecture, dependencies, and deployment strategy.
- Explain unfamiliar ML, engineering, and tooling concepts concisely before or while using them.
- Prefer short, clear explanations over long theory dumps.
- Answer exactly what the user asks first. Do not add extra detail unless requested.
- Distinguish strict necessity from recommendation. If the user asks whether something is required, answer required/not required, not what is cleaner or preferred.
- Avoid guessing. If something is an inference, label it as an inference.
- Do not flatter or overstate certainty. Give honest, calculated answers.
- Use concise direct answers by default; expand only when the user asks for explanation or detail.
- Be concrete and confident. Do not default to safe hedging. When the user asks for a decision, give a clear yes/no or chosen direction.
- Do not prescribe exact implementation steps unless the user asks for an opinion, recommendation, or plan. Act as a teammate: discuss tradeoffs and options first, then let the user decide.
- Be decisive by default. Avoid both-sides answers unless the tradeoff is genuinely necessary to answer the question.
- When several options exist, present the tradeoffs and recommend one.
- Ask for confirmation before making decisions that strongly affect project direction.
- Make reasonable small implementation choices independently when they are easy to change.

## User Context

- The user has limited ML experience and wants to learn through the project.
- Avoid assuming prior knowledge of ML vocabulary.
- When introducing a concept, explain:
  - what it means,
  - why it matters here,
  - how it affects the project.
- Use concrete examples from the current code or dataset whenever possible.

## ML Project Principles

- Start by defining the task clearly: input, output, target user, and success criteria.
- Establish a simple baseline before building a more complex model.
- Keep data handling explicit and reproducible.
- Track train/validation/test splits carefully to avoid leakage.
- Choose evaluation metrics that match the real project goal.
- Prefer understandable models and workflows unless complexity clearly improves results.
- Document assumptions, limitations, and next steps as the project evolves.



## Current Communication Rules

- The user is in charge of project decisions.
- The assistant acts as an assistant, not a project leader.
- Answer the user questions concisely.
- Avoid double-edged or hedged answers unless the user explicitly asks for tradeoffs.
- Give confident direct answers when enough information is available.
- If the answer is not known, say clearly: "I do not know."
- Do not pretend certainty.
- Do not prescribe steps unless the user asks for a plan, recommendation, or implementation.

## Current Project Objectives

The project goal is to build a measurable policy-improvement system for a simplified Factorio green-science design task. The assignment is intentionally vague, so defining the environment, reward, policy, and evaluation is part of the work.

### Main Claim To Demonstrate

Show that an iterative reward-guided policy-improvement loop can make a cheap baseline language-model policy better at producing or editing green-science factory floorplans.

In experiment terms:

```text
mean_reward(policy_final) > mean_reward(policy_0)
```

A stronger version, matching the task prompt, is:

```text
mean_reward(policy_final) is much larger than the typical reward gain from one individual improvement iteration
```

### Objective 1: Minimal Policy Improvement Demo ✓ COMPLETE

Implemented in `task1_gridworld/`. Notebook at `notebooks/task1_policy_iteration.ipynb`.

What was built:
- `environment.py` — finite gridworld MDP with traps (flat -2 cost), walls, optional slip transitions. Exact P[s,a,s'] and R[s,a] tensors.
- `policy_iteration.py` — closed-form linear-solve policy evaluation (derived from Bellman p.74), greedy improvement Eq 4.9 with 1e-14 tie tolerance, policy iteration p.80 with Ex 4.4 fix.
- `random_envs.py`, `viz.py` — random env generator and matplotlib/animation visualization.
- `tests.py` — 16 tests; headline: Eq 4.8 V_new ≥ V_old verified across 200 random envs (100 deterministic, 100 slip=0.1), sizes 5x5 to 20x20.
- Notebook: Bellman derivation, 4 showcase animations (snake dumb policy → optimal in 2 steps, 10 traps → 0), convergence plot.

Design decisions documented in plan.md and notebooks.

### Setup status (complete as of 2026-08-07)

- Task 1 complete and committed.
- Directory structure created: `mini_factorio/`, `harness/`, `training/`, `translator/`.
- factoriolab data downloaded: `mini_factorio/data/factoriolab_data.json` (Factorio 2.0.77, MIT).
- FLE confirmed: `pip install factorio-learning-environment`, Docker Desktop installed (free), connects via RCON. No Factorio license needed.
- Baseline model: `Qwen2.5-Coder-1.5B-Instruct` (HuggingFace).
- All deps in `pyproject.toml` including `[llm]` extra (transformers, trl, peft, bitsandbytes, datasets).
- **Next session starts Task 2 implementation immediately.**

### Objective 2: Factorio Green Science Environment

Create a measurable simplified Factorio-like environment focused on green science output.

Expected result:

- Define the floorplan representation, likely grid or JSON.
- Define fixed constraints such as map size, resource sources, build budget, machine costs, allowed machines, and allowed edits.
- Simplification decision: deleting a machine removes it permanently and gives no refund or inventory item.
- Define green science production rules.
- Reward decision: maximize green science output first; use lower resource cost only as a tie-breaker.
- Build a simulator/scorer that can evaluate a floorplan deterministically.
- Identify a cheap, fast, easy-to-serve language model as the baseline policy.
- Show the baseline is not already optimal by finding layouts that score better than baseline outputs.

### Objective 3: Iterated LLM Policy Improvement

Use DeepSeekMath-inspired reward-guided iteration to improve the policy over multiple rounds.

Expected result:

- Policy 0 proposes layout edits.
- The harness applies and validates edits.
- The simulator scores resulting layouts.
- High-scoring attempts are used to improve the next policy.
- Repeat for several iterations.
- Define policy_final as the best or final policy produced by this loop.
- Compare policy_0, intermediate policies, and policy_final using repeated trials and mean reward.

### What The Harness Means

The harness is the code that connects the model to the experiment:

```text
layout -> prompt/model -> proposed edit -> validation -> simulation -> reward -> logged result -> next iteration
```

The model alone is not enough. The harness makes model outputs testable and measurable.

### Important Design Choices For The User

Pause and discuss before deciding:

- Whether the Factorio environment is real Factorio, simplified Mini-Factorio, or a hybrid.
- The map size and starting layouts.
- The resource/build budget and machine costs.
- The allowed edit actions.
- The baseline language model.
- Current likely direction: train or fine-tune a model and use RL-style reward-guided improvement, because this seems closest to the DeepSeekMath expectation. Prompt/search-only methods may be fallback or baseline methods, not the preferred main interpretation.
- The exact reward formula.
- What evidence is enough for the final submission: code, plots, logs, examples, report, demo, or presentation.

### Likely Final Deliverables

- Working code repository.
- Simulator and reward function.
- Baseline policy harness.
- Iterative improvement loop.
- Experiment results comparing baseline and final policy.
- Plots or tables of reward over iterations.
- Example floorplans before and after improvement.
- Short writeup explaining the RL theory, Factorio simplification, DeepSeekMath inspiration, design choices, limitations, and results.

## Sutton And Barto Sections 4.2 And 4.3 Summary

### Section 4.2: Policy Improvement

The reason to compute a policy value function is to find a better policy.

Key idea:

```text
v_pi(s) = expected future reward from state s if we follow policy pi
q_pi(s, a) = expected future reward if we take action a once in state s, then follow pi afterward
```

To decide whether changing the policy at a state is good, compare:

```text
q_pi(s, new_action) vs v_pi(s)
```

If the new action has higher or equal expected value than the old policy choice, then switching to that action is safe. The policy improvement theorem says that if a new policy pi_prime chooses actions satisfying:

```text
q_pi(s, pi_prime(s)) >= v_pi(s) for every state s
```

then pi_prime is at least as good as pi:

```text
v_pi_prime(s) >= v_pi(s) for every state s
```

In plain English: if the new policy makes choices that are no worse than the old policy choices, measured by expected future reward, then the whole policy is no worse. If at least one important choice is strictly better, the new policy is better.

A common way to improve a policy is to make it greedy with respect to the old policy value function:

```text
pi_prime(s) = action with highest q_pi(s, action)
```

This means: at each state, choose the action that looks best after one step of lookahead plus the old policy value estimate.

### Section 4.3: Policy Iteration

Policy iteration repeats two steps:

```text
1. Policy evaluation: estimate v_pi for the current policy.
2. Policy improvement: create a better policy by choosing better actions according to the value estimates.
```

The loop is:

```text
policy_0 -> evaluate -> improve -> policy_1 -> evaluate -> improve -> policy_2 -> ... -> optimal or stable policy
```

For finite Markov decision processes, this process must eventually converge because there are only finitely many possible policies.

Important nuance: policy improvement gives a strictly better policy unless the current policy is already optimal. Sometimes one improvement step can already reach the optimal policy, as in the simple gridworld example from the book.

### Connection To This Project

For our Mini-Factorio project:

```text
state = current factory floorplan
action = proposed floorplan edit
reward = green science output or improvement
policy = model or strategy that proposes edits
value = expected future reward from a layout
q-value = expected future reward from applying a specific edit to a layout
```

A practical version of policy improvement is:

```text
1. Let policy_0 propose edits.
2. Score edits with the simulator.
3. Prefer edits with higher reward or expected future reward.
4. Use those results to define policy_1.
5. Repeat to get policy_final.
```

We may not compute exact textbook v-values for every possible floorplan because the layout space is too large. Instead, we can approximate improvement using sampled edits, simulator rewards, best-of-N selection, prompts, rankers, or fine-tuning.

## DeepSeekMath GRPO Theory To Potentially Use

DeepSeekMath introduces Group Relative Policy Optimization (GRPO), a PPO-like reinforcement learning method for improving an LLM policy.

Key idea:

```text
For the same prompt, sample a group of outputs.
Score every output.
Use the group average score as the baseline.
Update the policy toward above-average outputs and away from below-average outputs.
```

Why GRPO matters:

- PPO usually trains a separate value model/critic.
- That value model can be expensive because it may be comparable in size to the policy model.
- In LLM tasks, rewards often arrive only at the end of the generated output, making token-level value estimation awkward.
- GRPO avoids a separate value model by comparing outputs within the same sampled group.

DeepSeekMath notation:

```text
q = prompt/question
o_i = sampled output from old policy
r_i = reward for output o_i
baseline = mean reward of outputs for the same q
advantage = relative score of o_i compared with group baseline
```

Project mapping:

```text
q = current factory layout / task prompt
o_i = candidate floorplan edit
r_i = simulator reward after applying edit
baseline = average reward of candidate edits for that layout
advantage = edit reward - group average reward
policy update = make above-average edits more likely, below-average edits less likely
```

DeepSeekMath iterative GRPO loop:

```text
1. Start with initial policy model.
2. For each iteration, set current policy as reference.
3. Sample a batch of prompts.
4. For each prompt, sample multiple outputs from the old policy.
5. Compute rewards for those outputs.
6. Compute group-relative advantages.
7. Update the policy using the GRPO objective.
8. Optionally update the reward model using replay/history.
9. Repeat.
```

Possible project adaptation:

```text
1. Start with cheap baseline LLM policy.
2. For each layout, generate multiple candidate edits.
3. Score each edit with the Mini-Factorio simulator.
4. Compare edits within the same layout group.
5. Improve the policy using the high-scoring edits.
6. Repeat for multiple iterations.
```

Important distinction:

- Full GRPO means actually updating model weights with the GRPO objective.
- GRPO-inspired means using the same group-relative sampling and scoring idea without necessarily implementing full RL weight updates.
- Current likely direction: prefer actual model training/fine-tuning with RL-style improvement if feasible, because it more directly matches DeepSeekMath.

### Outcome Supervision vs Process Supervision

Outcome supervision:

```text
Score only the final output.
```

DeepSeekMath version:

```text
question -> full answer -> final reward
```

Project mapping:

```text
layout -> full edit or final edited layout -> final green-science reward
```

GRPO normalizes each output reward inside the group:

```text
normalized_reward_i = (reward_i - mean(group_rewards)) / std(group_rewards)
```

Then every token/action in that output receives the same normalized advantage.

Process supervision:

```text
Score intermediate steps, not only the final output.
```

DeepSeekMath version:

```text
score each reasoning step in a math solution
```

Project mapping:

```text
score each edit step in a multi-edit factory plan
```

Example:

```text
step 1: add gear assembler -> useful
step 2: connect copper -> useful
step 3: delete science assembler -> bad
```

Process supervision can give more detailed feedback, but requires a reward signal for each step.

### Iterative GRPO

DeepSeekMath also describes iterative GRPO:

```text
As the policy improves, the old reward model may become insufficient.
Generate new samples from the current policy.
Use those samples to update the reward model.
Continue training the policy with the updated reward model.
Keep some historical data through replay, e.g. 10% old data.
```

Project mapping:

```text
As the factory-edit policy changes, collect new layout/edit examples.
Use new examples plus some old examples to update the scorer/ranker if we train one.
Then continue improving the policy.
```

For our project, the simulator can serve as a reliable reward function, so updating a learned reward model may be unnecessary unless we decide to train a separate ranker/reward model.

This section records theory that may be used; the implementation choice remains a user decision.

## Engineering Principles

- Read the existing codebase before editing.
- Follow the existing style and structure once the project has one.
- Keep changes focused and easy to review.
- Add tests or validation checks for important behavior.
- Do not overwrite user changes.
- Avoid unnecessary frameworks or abstractions.
- Prefer reproducible commands and scripts over one-off manual steps.

## Decision Checkpoints

Pause and involve the user before:

- selecting or changing the project objective,
- choosing the dataset,
- choosing the ML model family,
- changing the evaluation metric,
- adding major dependencies,
- restructuring the repository,
- deleting or replacing existing work,
- preparing the final demo, report, or presentation.

## Explanation Style

Use this pattern for new ML concepts:

```text
Concept: short definition.
Why it matters: connection to our project.
Practical effect: what we will do differently because of it.
```

Example:

```text
Concept: A validation set is data used to compare models during development.
Why it matters: It helps estimate performance on examples the model has not trained on.
Practical effect: We should not tune decisions using the final test set.
```

## Working Agreement

The assistant should act as a senior technical collaborator and teacher. The user makes major project decisions. The assistant should propose, explain, implement, verify, and keep the project moving without hiding important reasoning.
