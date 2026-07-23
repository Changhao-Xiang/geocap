"""Inference for the Qwen3.5 MoE foscap stage3 checkpoint with tag-ordered decoding.

For each test image, we walk through a fixed ordered list of XML-like tags
(`<size>`, `<shape>`, ...) and let the model generate the content between
`<tag>` and `</tag>` one tag at a time. This mirrors `stage3/internvl/tag_decoding.py`,
adapted to a Qwen3-VL style chat template where the assistant response is grown
by repeatedly re-rendering the prompt with the partial response appended.

Reads jsonl with `image` and `input` fields (e.g. dataset/stage3/stage3_test.jsonl)
and writes jsonl rows of `image`, `question`, `response`.

Supports data-parallel inference: set --num_gpus >1 to shard entries evenly
across GPUs. Each worker writes to <output_file>.rank{r} and rank 0 merges
shards into --output_file at the end.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


ALL_TAGS = [
    "<size>",
    "<shape>",
    "<equator>",
    "<lateral slopes>",
    "<poles>",
    "<axis>",
    "<length>",
    "<width>",
    "<ratio>",
    "<number of volutions>",
    "<coil tightness>",
    "<heights of volutions>",
    "<spirotheca>",
    "<septa>",
    "<proloculus>",
    "<tunnel shape>",
    "<tunnel angle>",
    "<chomata>",
    "<axial filling>",
]


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/qwen35_stage3_only")
    parser.add_argument("--input_file", type=str, default="dataset/stage3/stage3_test.jsonl")
    parser.add_argument("--image_dir", type=str, default="dataset/stage3/images")
    parser.add_argument("--output_file", type=str,
                        default="eval_data/outputs/qwen35_stage3_only.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Per-tag max new tokens (tag mode) or total max new tokens (natural mode).")
    parser.add_argument("--decode_mode", type=str, default="tag", choices=["tag", "natural"],
                        help="`tag`: walk ALL_TAGS in order, generating content between each "
                             "<tag>/</tag>. `natural`: a single free-form generation call.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Keep <think> in the prompt; default disables it (matches qwen3_5_nothink).")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip images already present in the (shard) output file.")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs for data-parallel inference. "
                             "Each worker shards `entries` and binds to one GPU.")
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Number of stochastic passes to run while keeping the model loaded.")
    parser.add_argument("--run_start", type=int, default=1,
                        help="First run number used to format `{run}` in --output_file.")
    return parser.parse_args()


def build_messages(question: str, partial_response: str | None = None) -> list[dict]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    if partial_response is not None:
        messages.append({"role": "assistant", "content": partial_response})
    return messages


def render_prompt(processor, question: str, partial_response: str, enable_thinking: bool) -> str:
    """Render the chat template, leaving the assistant turn open at `partial_response`."""
    # We render with the partial assistant turn included, then strip the trailing <|im_end|>
    # so generation continues that same turn.
    text = processor.apply_chat_template(
        build_messages(question, partial_response),
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    suffix = "<|im_end|>\n"
    if text.endswith(suffix):
        text = text[: -len(suffix)]
    elif text.endswith("<|im_end|>"):
        text = text[: -len("<|im_end|>")]
    return text


def shard_path(output_file: str, rank: int, world_size: int) -> str:
    if world_size == 1:
        return output_file
    return f"{output_file}.rank{rank}"


def run_output_path(output_file: str, run_number: int, num_runs: int) -> str:
    """Resolve a per-run output path without changing single-run behavior."""
    if "{run" in output_file:
        return output_file.format(run=run_number)
    if num_runs == 1:
        return output_file
    path = Path(output_file)
    return str(path.with_name(f"{path.stem}.run{run_number:02d}{path.suffix}"))


@torch.inference_mode()
def run_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    # Pin this process to a single GPU. `device_map="auto"` would spread the
    # model across every visible GPU, which collides with sibling workers, so
    # we set the current device and pass it explicitly to from_pretrained.
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    entries = load_jsonl(args.input_file)
    entries = entries[args.start : args.end]
    # Stride-based sharding keeps load balanced if entries are clustered by
    # difficulty/length in the input file.
    entries = entries[rank::world_size]

    if rank == 0:
        print(f"Loading processor & model from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    # Single-GPU: keep `device_map="auto"` so very large models can still shard
    # across visible cards. Multi-GPU data-parallel: each rank loads the full
    # model onto its own card.
    if world_size == 1:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
    model.eval()

    # Stop as soon as the model emits the start of the next tag. The tokenizer encodes
    # the leading "<" (and common joined variants) as single tokens, so we use those
    # ids as eos. The model is expected to stop right before `</current_tag>`, and the
    # closing tag is then appended manually.
    stop_strings = ["<", " <", "</", " </", ".<", ".</"]
    stop_token_ids: list[int] = []
    for s in stop_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            stop_token_ids.append(ids[0])
    stop_token_ids = sorted(set(stop_token_ids))
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    do_sample = args.temperature > 0
    base_gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        eos_token_id=stop_token_ids,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if do_sample:
        base_gen_kwargs["temperature"] = args.temperature
        base_gen_kwargs["top_p"] = args.top_p

    # Natural mode uses standard EOS stopping and a larger budget covering the
    # full response rather than a per-tag slice.
    natural_gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if tokenizer.eos_token_id is not None:
        natural_gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if do_sample:
        natural_gen_kwargs["temperature"] = args.temperature
        natural_gen_kwargs["top_p"] = args.top_p

    for run_offset in range(args.num_runs):
        run_number = args.run_start + run_offset
        output_file = run_output_path(args.output_file, run_number, args.num_runs)
        out_path = shard_path(output_file, rank, world_size)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        done_images: set[str] = set()
        if args.resume and os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        done_images.add(json.loads(line).get("image", ""))
                    except json.JSONDecodeError:
                        continue
            if rank == 0:
                print(
                    f"[rank {rank}] Run {run_number}: skipping "
                    f"{len(done_images)} already-processed entries"
                )

        mode = "a" if args.resume else "w"
        desc = (
            f"qwen35 {args.decode_mode}-decode run {run_number} "
            f"[rank {rank}/{world_size}]"
        )
        # Only rank 0 shows the bar to avoid clobbering the terminal; others run silently.
        disable_bar = rank != 0
        with open(out_path, mode, encoding="utf-8") as fout:
            for entry in tqdm(entries, desc=desc, disable=disable_bar):
                image_name = entry["image"]
                if image_name in done_images:
                    continue
                question = entry["input"]

                image = Image.open(Path(args.image_dir) / image_name).convert("RGB")

                if args.decode_mode == "tag":
                    response = ""
                    for tag in ALL_TAGS:
                        close_tag = "</" + tag[1:]

                        # Append the opening tag to the partial assistant response, re-render
                        # the chat template with that partial response left open, then let the
                        # model continue.
                        partial = (response + (" " if response else "") + tag).strip()
                        prompt_text = render_prompt(
                            processor, question, partial, args.enable_thinking
                        )

                        inputs = processor(
                            text=[prompt_text],
                            images=[image],
                            return_tensors="pt",
                        ).to(model.device)

                        output_ids = model.generate(**inputs, **base_gen_kwargs)
                        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
                        piece = tokenizer.decode(new_tokens, skip_special_tokens=True)

                        # The model may have emitted the leading '<' of the next tag (that's our
                        # stop token). Cut at the first '<' so we keep only the content.
                        piece = piece.split("<", 1)[0].rstrip()

                        response = f"{partial}{piece}{close_tag}"

                    response = response.strip()
                else:
                    # Natural mode: a single open-ended generation call.
                    prompt_text = processor.apply_chat_template(
                        build_messages(question),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=args.enable_thinking,
                    )
                    inputs = processor(
                        text=[prompt_text],
                        images=[image],
                        return_tensors="pt",
                    ).to(model.device)
                    output_ids = model.generate(**inputs, **natural_gen_kwargs)
                    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
                    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                fout.write(json.dumps(
                    {"image": image_name, "question": question, "response": response},
                    ensure_ascii=False,
                ) + "\n")
                fout.flush()


def merge_shards(output_file: str, world_size: int) -> None:
    """Concatenate per-rank shard files into output_file, then remove shards."""
    if world_size <= 1:
        return
    shard_files = [shard_path(output_file, r, world_size) for r in range(world_size)]
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    n_total = 0
    with open(output_file, "w", encoding="utf-8") as fout:
        for sf in shard_files:
            if not os.path.exists(sf):
                continue
            with open(sf, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line if line.endswith("\n") else line + "\n")
                        n_total += 1
    for sf in shard_files:
        try:
            os.remove(sf)
        except OSError:
            pass
    print(f"Merged {world_size} shards -> {output_file} ({n_total} rows)")


def main() -> None:
    args = parse_args()

    if args.num_runs < 1:
        raise ValueError("--num_runs must be at least 1")

    available = torch.cuda.device_count()
    world_size = max(1, args.num_gpus)
    if world_size > available:
        print(f"Requested num_gpus={world_size} but only {available} visible; "
              f"falling back to {available}.")
        world_size = max(1, available)

    if world_size == 1:
        run_worker(0, 1, args)
    else:
        # spawn binds each child to a single GPU via CUDA_VISIBLE_DEVICES inside
        # run_worker. We must not touch CUDA in the parent before spawning.
        mp.spawn(run_worker, args=(world_size, args), nprocs=world_size, join=True)
        for run_offset in range(args.num_runs):
            run_number = args.run_start + run_offset
            output_file = run_output_path(args.output_file, run_number, args.num_runs)
            merge_shards(output_file, world_size)


if __name__ == "__main__":
    main()
