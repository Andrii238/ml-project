from harness.evaluator import evaluate_policy
from harness.prompt_builder import render_grid
from harness.qwen_policy import QwenPolicy
from training.reward_wrapper import layout_from_prompt, reward_breakdown
from harness.edit_applier import apply_edits
from harness.edit_parser import parse_edits
from training.template_sft_generator import build_template_dataset


def main() -> None:
    rows = build_template_dataset([9001], variants_per_seed=4)
    row = rows[2]

    print("TARGET RATE:", row["sim_gs_rate"])
    print("TARGET ASSEMBLERS:", row["n_assemblers"])
    print("\nTARGET COMPLETION:")
    print(row["completion"][:5000])

    policy = QwenPolicy(adapter_path="./ckpts/sft", load_in_4bit=False)

    def gen(prompts, **_kwargs):
        outs = policy.generate(
            prompts,
            max_new_tokens=4096,
            temperature=0.2,
            batch_size=1,
        )
        print("\nMODEL OUTPUT:")
        print(outs[0][:8000])
        return outs

    summary = evaluate_policy(
        gen,
        [row["prompt"]],
        seeds=[row["seed"]],
        samples_per_prompt=1,
        batch_size=1,
        max_new_tokens=4096,
        temperature=0.2,
    )

    print("\nSUMMARY:")
    print(summary)
    if summary.per_sample:
        sample = summary.per_sample[0]
        print("\nKEY METRICS:")
        print("parse_ok:", sample.parse_ok)
        print("edits_parsed:", sample.edits_parsed)
        print("edits_applied:", sample.edits_applied)
        print("green_science_rate:", sample.green_science_rate)
        print("reward:", sample.reward)

        print("\nREWARD BREAKDOWN:")
        print(reward_breakdown(row["prompt"], sample.completion))

        layout = layout_from_prompt(row["prompt"])
        parsed = parse_edits(sample.completion)
        if layout is not None:
            applied = apply_edits(layout, parsed.edits)
            print("\nAPPLY ERRORS:")
            print(applied.errors)
            print("\nFINAL GRID:")
            print(render_grid(applied.layout))


if __name__ == "__main__":
    main()
