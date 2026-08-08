# Andrii Plan

## Task 1

We take a simple finite task. Current idea: an `n x m` grid.

The agent spawns at some point and needs to reach a portal in the minimum possible number of steps. There are traps on the grid. If the agent steps on a trap, it loses one turn and cannot move for one step. The agent can move in any of four directions: up, down, left, right.

For any policy, we can calculate the values/rewards exactly over this finite MDP. Then we can apply the Sutton/Barto policy improvement theorem: if a changed action has better or equal expected value, the new policy is guaranteed to be at least as good. We can repeat this process and show policy iteration.

## Task 2

We take an open-source model that is not huge and can be locally fine-tuned. Probably some Chinese/open-source code-capable model. This model is our baseline policy.

We use FLE as the real Factorio environment/harness. We represent Factorio floorplans using numbers, with a prompt explaining what each number means. For example, each number can represent a different object/entity type. Restrict layouts only to objects that can help green science production in any way.

We build a program that can translate a real Factorio floorplan into this numeric representation and translate the numeric representation back into a real Factorio-compatible floorplan.

For reward measurement, we use a formula that we can adjust with time. The main positive term is green science production. Other terms are penalties, such as floor area used and resources/materials used. So the reward is basically:

```text
green science production - penalties for inefficient solutions
```

Green science production is the main reward. Everything else is secondary punishment.

## Harness / Representation Idea

We will make a database of available objects/machines in the game: requirements, outputs, inputs, costs, and other useful metadata.

We will provide the model with JSON format describing the map. Every JSON entry describes the id, location, type, and direction of object at that location. Assemblers also include recipe. Connections are represented separately so edits are unambiguous.

Example idea:

```json
{"id": "assembler_1", "type": "assembler", "recipe": "green_science", "x": 4, "y": 2, "direction": "east"}
{"from": "belt_assembler_1", "to": "science_assembler_1"}
```

The prompt format should stay mostly the same. The game rules/object database stay the same, and only the map/layout changes between examples.

To generate random maps, we assign probabilities for which object type is added next. We track currently available open spaces on the map, then place the chosen object into a random open space. This algorithm can be improved later.

## Task 3

We use the GRPO method.

For a given floorplan/task prompt, the model generates multiple candidate action plans or floorplan modifications within constraints. Each candidate is applied to produce a final floorplan. We calculate the reward for each final floorplan.

Then we use GRPO: compare the candidates within the same group, reinforce the higher-reward outputs, and reduce probability of lower-reward outputs. In this way, we train the model to become better at producing floorplan changes that improve reward.

The goal is to show that after several iterations, the trained policy performs better than the baseline policy.


## Validation Note

We need separate training and validation floorplans.

Training floorplans are used for GRPO updates. Validation floorplans are not used for training. After each training iteration, we evaluate the current policy on the same validation floorplans.

If validation reward improves, it shows the model learned a general strategy instead of only memorizing training layouts.

