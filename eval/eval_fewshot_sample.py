import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def _parse_script_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_id", type=str, default="openai/gpt-5.5")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="name used in the output filename; defaults to the last path segment of --model_id",
    )
    parser.add_argument("--num_fewshot", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--training_set", type=str, default="dataset/stage3/stage3_train_direct.jsonl")
    parser.add_argument("--test_set", type=str, default="dataset/stage3/stage3_test_direct.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parsed, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return parsed


_script_args = _parse_script_args()

import jsonlines  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common.llm import APIGenerator  # noqa: E402


def sample_few_shot(training_set: list[dict], num_fewshot: int = 2) -> list[dict]:
    if num_fewshot <= 0:
        return []
    demonstrations = []
    sampled_examples = random.sample(training_set, num_fewshot)
    for i, example in enumerate(sampled_examples):
        demonstration = {
            "input": f"Example{i+1}:\nUser: {example['input']}",
            "image": example["image"],
            "output": f"Assistant: {example['output']}",
        }
        demonstrations.append(demonstration)
    return demonstrations


def build_inputs(data: dict, demonstrations: list[dict]) -> list[tuple[str, str]]:
    user_prompt: list[tuple[str, str]] = []
    for demonstration in demonstrations:
        for key, value in demonstration.items():
            if key == "input" or key == "output":
                user_prompt.append(("text", value))
            elif key == "image":
                user_prompt.append(("image", f"dataset/stage3/images/{value}"))
    return user_prompt + [
        ("text", f"User: {data['input']}"),
        ("image", f"dataset/stage3/images/{data['image']}"),
        ("text", "Assistant:"),
    ]


def main():
    args = _script_args
    model_name = args.model_name or args.model_id.split("/")[-1]
    output_file = (
        args.output_file
        or f"eval_data/outputs/{model_name}_{args.num_fewshot}shot_sample.jsonl"
    )

    sys_prompt = (
        "You are a helpful assistant. You will be provided with a fossil picture input and you should depict this picture. "
        "The user will provide you with the pixel information of the image and its actual length and width. "
        "You need to infer the real data and description based on the image and the user's input. "
        "Do not use sub-titles and lists to devide the output; instead, merge the paragraph into a whole. "
        "You do not need a conclusion, and you should try to mimic the given examples. "
        "You need to complete the output of 'Your turn' based on the examples."
    )

    random.seed(args.seed)

    with open(args.training_set, "r") as f:
        training_set = [json.loads(line) for line in f]

    generator = APIGenerator(args.model_id, max_tokens=8196, sys_prompt=sys_prompt, temperature=1.0)

    with open(args.test_set, "r") as f:
        lines = list(f)

    tasks_input: list[tuple[dict, list[tuple[str, str]]]] = []
    for line in lines:
        data = json.loads(line)
        demonstrations = sample_few_shot(training_set, num_fewshot=args.num_fewshot)
        inputs = build_inputs(data, demonstrations)
        tasks_input.append((data, inputs))

    results: list[dict | None] = [None] * len(tasks_input)

    def process_one(idx: int):
        data, inputs = tasks_input[idx]
        try:
            resp = generator.get_one_response(inputs)
        except Exception as e:
            print(f"[warn] idx={idx} image={data['image']} failed: {e}", file=sys.stderr)
            resp = ""
        return idx, {"image": data["image"], "question": data["input"], "response": resp}

    desc = f"{model_name} {args.num_fewshot}shot"
    if args.num_workers > 1:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_one, i) for i in range(len(tasks_input))]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                idx, result = fut.result()
                results[idx] = result
    else:
        for i in tqdm(range(len(tasks_input)), desc=desc):
            idx, result = process_one(i)
            results[idx] = result

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with jsonlines.open(output_file, mode="w") as writer:
        for result in results:
            writer.write(result)
    print(f"[done] wrote {len(results)} entries to {output_file}")


if __name__ == "__main__":
    main()
