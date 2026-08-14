# Factorio Green-Science Policy Improvement

## Task 1: Simple Pretext Task

For Task 1, I chose a very simple pretext task: optimal grid walking with traps. The agent starts on an `n x m` grid and needs to reach a portal in the minimum number of steps. Some cells are traps. If the agent steps on a trap, it loses one step. The agent can move up, down, left, or right.

I chose this because it lets me use the Sutton/Barto policy-improvement theorem exactly. Since the task is small and finite, I can calculate the value of every state, improve the policy greedily, and iterate.

This directly demonstrates the requested idea: improvement of any policy under any circumstance, unless the policy is already optimal.

In real LLM/factory problems, Sutton/Barto is more of a theoretical layer. We usually cannot calculate every state value exactly. DeepSeekMath is more powerful for the LLM part, but it does not give this exact guarantee. So for Task 1, I use Sutton/Barto for the guarantee.

## Task 2: Factorio Green Science Environment

For Task 2, I first tried to stay very close to the real game and ask the model to build the full layout from the beginning. It became clear that this is too complex task even for frontier models, let alone small open-source model.

Therefore I simplified the task as much as possible while keeping the real objective: maximize green-science production.

The final setup has three fixed chests:

- one input chest for transport belts
- one input chest for inserters
- one output chest for green science

The model sees the current layout and rules, then outputs JSON edits. The main edits are:

- place assembler
- place conveyor line

Then my simulator applies the edits and calculates green science delivered to the output chest.

The interpretation is: suppose in Factorio you already have some belt production and inserter production, and now you want to dedicate those streams to green science. The model tries to build the best local structure for that.

Because the green-science recipe uses `1 transport belt + 1 inserter`, it usually does not make sense to have very different input rates. The smaller input becomes the bottleneck. So for simplicity I use equal input rates.

I use JSON instead of a raw number matrix because it is easier for the model to understand.

FLE (1000+ stars on github) validation is the check against the reference game environment. I built a translator/driver that exports our layouts into FLE, adds the needed power setup, sets recipes, runs the game, and measures real production. On a hand-built validation layout, my simulator reported `0.3125` items/sec and FLE measured `0.3117` items/sec, only about `0.3%` error.

## Task 3: Improving the LLM Policy

For the baseline policy I use `Qwen2.5-Coder-1.5B-Instruct`.

I chose it because it is cheap, fast, and easy to run on Colab (saves a lot of time). It is also weak enough on this task that improvement is measurable.

The base model is not good at this task. It can produce valid-looking JSON, but it usually does not produce green science. So before GRPO, I run SFT (from deepseekmath paper).

## SFT Step

SFT is important because GRPO needs the model to produce at least somewhat useful attempts. If every sampled layout gets zero reward, the RL signal is very weak.

The SFT dataset has `446` verified training examples:

- half are empty chest-only maps where the model builds everything
- half are partial maps where some assemblers or conveyor lines were deleted and the model repairs them
- examples contain `1` to `9` assemblers depending on the template
- every target example is checked by the simulator and produces green science

After SFT, results on validation:

| Policy | Mean reward | Green science/sec | Parse OK | Valid output |
|---|---:|---:|---:|---:|
| Base policy | -0.752 | 0.0000 | 100% | 100% |
| SFT policy | 495.483 | 1.0417 | 100% | 100% |

So SFT moved the model from zero green-science production to `1.0417` science/sec on validation, with `100%` parse and validity rate. Clear and huge improvement. 

## GRPO Step

For GRPO, I use the DeepSeekMath paper idea, which is super smart. 

For the same prompt, the model generates several candidate answers. We score each answer, compare them inside the group, and train the model toward the better ones.

In my run:

1. Give the model one layout prompt.
2. Sample `8` possible JSON outputs.
3. Run each output through the simulator.
4. Calculate reward.
5. Compare rewards inside the group.
6. Update the LoRA adapter toward better outputs.
7. Save intermediate policies at steps `25`, `50`, `75`, and final.

Final deterministic checkpoint evaluation:

| Policy | Mean reward | Green science/sec | Machines | Materials | Parse OK | Valid output |
|---|---:|---:|---:|---:|---:|---:|
| Base policy | -0.752 | 0.0000 | 1.00 | 8.4 | 100% | 100% |
| SFT policy | 495.483 | 1.0417 | 5.00 | 60.0 | 100% | 100% |
| GRPO 25 | 495.483 | 1.0417 | 5.00 | 60.0 | 100% | 100% |
| GRPO 50 | 577.973 | 1.2187 | 5.85 | 66.0 | 100% | 100% |
| GRPO 75 | 558.564 | 1.1771 | 5.65 | 64.5 | 100% | 100% |
| GRPO final | 592.530 | 1.2500 | 6.00 | 67.0 | 100% | 100% |

Final GRPO improved over SFT by `+97.047` reward and `+0.2083` green science/sec, about `20%`.

The sampled-policy diagnostic also improved strongly: sampled mean reward went from `170.676` to `592.404`, and sampled green science went from `0.3151` to `1.25`.

## Reward Function

The reward is not one variable function because pure green-science reward is too sparse. Early bad layouts often produce nothing, and then the model cannot tell “almost useful” from “completely useless.” We also need to consider certain tradeoffs like spending resources to unlock next tier, or to waste too much space (and produce pollution) etc.

But the main reward is still green science delivered to the output chest as it is the main objective.

Reward terms:

- `+500 * green_science_rate`
- `+20` once if any green science is delivered
- `+0.5` for each assembler receiving transport belts
- `+0.5` for each assembler receiving inserters
- `+1` for each assembler producing science
- `+100 * produced_but_not_delivered_rate`

Penalties/costs:

- `-30` if no assembler is placed
- `-10` for each missing required chest
- assembler cost: tier 1 `0.53`, tier 2 `3.22`, tier 3 `8.94`
- conveyor cost: tier 1 `0.03`, tier 2 `0.23`, tier 3 `0.63`
- tier unlock penalties: assembler tier 2 `3.25`, assembler tier 3 `9.0`, conveyor tier 2 `0.45`, conveyor tier 3 `1.25`
- random bonus is disabled

## Notes and Possible Improvements

I run everything on Google Colab.

I use the 1.5B model mainly because it is fast. A 7B Qwen model would be smarter and better at the layout JSON, but it would make training and GRPO slower. Since the 1.5B model already shows clear improvement after SFT and GRPO, I used it.

Possible improvements:

- use Qwen 7B with LoRA
- add more SFT examples
- tune the reward formula more
- run more GRPO steps
- use larger GRPO group size
- validate more top layouts in FLE
