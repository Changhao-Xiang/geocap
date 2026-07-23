import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from common.args import fossil_eval_args, logger, run_args
from common.llm import generator_mapping, model_path_mapping
from eval.cal_score import statistics
from eval.utils import (
    calculate_axis_shape_rating,
    calculate_chomata_rating,
    calculate_score,
    characteristics,
    extract_range_or_num,
    extract_tunnel_shape,
    find_first_json_block,
    rule_based_eval_features,
)


def record_err(input, output, error, idx, mode):
    import time

    with open(f"{fossil_eval_args.eval_result_dir}/Error_log.log", "a") as log:
        log.write(
            f"\n{time.ctime()} - {error} @ entry[{idx}] with mode {mode}, examine:\nInput:{input}\nOutput:{output}\n"
        )


def load_json_records(path):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def get_image_name(entry):
    return entry.get("img") or entry.get("image") or entry.get("image_path") or ""


def get_question(entry):
    if "question" in entry:
        return entry["question"]
    if "input" in entry:
        return entry["input"]
    for message in entry.get("conversations", []):
        if message.get("from") == "human":
            return message.get("value", "")
    return ""


def get_generated_output(entry):
    return entry.get("output") or entry.get("response") or ""


def get_reference_description(entry):
    if "reference" in entry:
        return entry["reference"]
    if "output" in entry:
        return entry["output"]
    for message in entry.get("conversations", []):
        if message.get("from") == "gpt":
            return message.get("value", "")
    return ""


def build_image_map(entries, image_key_getter=get_image_name):
    image_map = {}
    basename_map = {}
    for entry in entries:
        image_name = image_key_getter(entry)
        if not image_name:
            continue
        image_map[image_name] = entry
        basename_map[Path(image_name).name] = entry
    return image_map, basename_map


def get_entry_by_image(image_name, image_map, basename_map):
    return image_map.get(image_name) or basename_map.get(Path(image_name).name)


def normalize_extracted_item(item, image_path=""):
    normalized_item = {"image_path": image_path or get_image_name(item)}
    for char in characteristics:
        value = item.get(char, "")
        normalized_item[char] = "" if value is None else value
    return normalized_item


def normalize_extracted_batch(extracted_batch):
    return [normalize_extracted_item(item) for item in extracted_batch]


def validate_extracted_json_item(json_item):
    if not isinstance(json_item, dict):
        raise ValueError("Extracted JSON must be an object.")

    missing_fields = [char for char in characteristics if char not in json_item]
    if missing_fields:
        raise ValueError(f"Extracted JSON missing fields: {', '.join(missing_fields)}")


def build_caption_batch(output_entries, reference_entries=None):
    reference_map, reference_basename_map = build_image_map(reference_entries or [])
    caption_batch = []
    for output_entry in output_entries:
        image_name = get_image_name(output_entry)
        reference_entry = get_entry_by_image(image_name, reference_map, reference_basename_map)
        if reference_entries is not None and reference_entry is None:
            logger.warning(f"No reference description found for image {image_name}")
        caption_batch.append(
            {
                "img": image_name,
                "question": get_question(output_entry),
                "output": get_generated_output(output_entry),
                "reference": get_reference_description(reference_entry) if reference_entry else "",
            }
        )
    return caption_batch


def load_caption_batch():
    if fossil_eval_args.eval_output_file:
        output_entries = load_json_records(fossil_eval_args.eval_output_file)
        reference_entries = None
        has_extracted_reference = (
            fossil_eval_args.eval_reference_file
            and os.path.exists(fossil_eval_args.eval_reference_file)
        )
        if fossil_eval_args.eval_reference_desc_file and not has_extracted_reference:
            reference_entries = load_json_records(fossil_eval_args.eval_reference_desc_file)
        caption_batch = build_caption_batch(output_entries, reference_entries)
    else:
        if not fossil_eval_args.eval_origin_file:
            raise ValueError(
                "eval_output_file is required. Pass eval_reference_desc_file for raw references, "
                "or pass an existing extracted JSON with eval_reference_file."
            )
        logger.warning("eval_origin_file is deprecated; use eval_output_file and eval_reference_desc_file instead")
        caption_batch = load_json_records(fossil_eval_args.eval_origin_file)
    return caption_batch[fossil_eval_args.eval_start_pos : fossil_eval_args.eval_end_pos]


def align_extracted_reference(caption_batch, extracted_reference):
    reference_map, reference_basename_map = build_image_map(extracted_reference)
    aligned_reference = []
    missing_images = []
    for entry in caption_batch:
        image_name = get_image_name(entry)
        reference_entry = get_entry_by_image(image_name, reference_map, reference_basename_map)
        if reference_entry is None:
            missing_images.append(image_name)
            continue
        aligned_reference.append(reference_entry)
    if missing_images:
        raise ValueError(
            f"Missing extracted reference info for {len(missing_images)} images, "
            f"first missing image: {missing_images[0]}"
        )
    return aligned_reference


class Evaluater:
    def __init__(self) -> None:
        self.loaded_llm = False

    def load_llm_generator(self):
        """Initialize the LLM generator"""
        assert not self.loaded_llm
        # Initialize llm
        model_name, model_id = fossil_eval_args.eval_llm.split("-", 1)  # qwen25-14b
        model_path = model_path_mapping[model_name].format(model_id)
        if model_name != "api":
            self.mode = "local"
        else:
            self.mode = "api"
        self.sys_prompt = "You are a helpful assistant and always respond in JSON format."
        self.llm_generator = generator_mapping[model_name](
            model_path, temperature=0.7, max_tokens=8192, sys_prompt=self.sys_prompt
        )
        self.loaded_llm = True

        # Initialize prompt
        with open("eval/prompts/extract_system_prompt.txt", "r", encoding="utf8") as system_prompt_file:
            self.prompt = system_prompt_file.read()

    def reload_eval_mode(self):
        with open("eval/prompts/eval_system_prompt.txt", "r", encoding="utf8") as system_prompt_file:
            self.prompt = system_prompt_file.read()

    def get_messages(self, testee_batch):
        if self.mode == "local":
            messages = [
                [
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": self.prompt.replace("{input}", testee)},
                ]
                for testee in testee_batch
            ]
        else:
            messages = [
                [{"role": "user", "content": self.prompt.replace("{input}", testee)}]
                for testee in testee_batch
            ]
        return messages

    def generate_llm_responses(self, messages, desc, batch_size=1):
        if not messages:
            return [], False

        if self.mode == "api" and run_args.num_workers > 1:
            max_workers = min(run_args.num_workers, len(messages))
            logger.info(f"Calling API LLM with {max_workers} worker threads")
            get_one_response = getattr(self.llm_generator, "get_one_response")
            responses = [[""] for _ in messages]
            fail_flag = False
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(get_one_response, message): idx
                    for idx, message in enumerate(messages)
                }
                for future in tqdm(
                    as_completed(future_to_idx),
                    total=len(messages),
                    desc=desc,
                ):
                    idx = future_to_idx[future]
                    try:
                        responses[idx] = [future.result()]
                    except Exception as e:
                        responses[idx] = [""]
                        record_err(messages[idx], "", e, idx, desc)
                        fail_flag = True
            return responses, fail_flag

        responses = []
        with tqdm(total=len(messages), desc=desc) as pbar:
            for batch in self.llm_generator(messages, batch_size=batch_size):
                for response in batch:
                    responses.append([response])
                    pbar.update(1)
        return responses, False

    def parse_extraction_batch(self, batch):
        process_json_batch = []
        for response in batch:
            json_block, remaining_text = find_first_json_block(response)
            json_item = json.loads(json_block)
            validate_extracted_json_item(json_item)
            process_json_batch.append(json_item)
        return process_json_batch

    def extract_one_with_retries(self, message, initial_batch, image_path, idx, mode):
        max_retries = max(0, int(getattr(fossil_eval_args, "eval_extract_max_retries", 3)))
        batch = initial_batch
        last_error = None
        last_output = batch[0] if batch else ""

        for attempt in range(max_retries + 1):
            try:
                process_json_batch = self.parse_extraction_batch(batch)
                return [
                    normalize_extracted_item(json_item, image_path=image_path)
                    for json_item in process_json_batch
                ], False
            except Exception as e:
                last_error = e
                last_output = batch[0] if batch else ""
                if attempt >= max_retries:
                    break

                logger.warning(
                    f"Retrying extraction for {image_path or f'entry[{idx}]'} "
                    f"({mode}), attempt {attempt + 1}/{max_retries}; reason: {e}"
                )
                retry_responses, _ = self.generate_llm_responses(
                    [message],
                    desc=f"Eval: Retrying {mode}",
                    batch_size=1,
                )
                batch = retry_responses[0] if retry_responses else [""]

        process_json_batch = [normalize_extracted_item({}, image_path=image_path)]
        record_err(message, last_output, last_error, idx, mode)
        return process_json_batch, True

    def extract(self, entry_batch, mode):
        testee = [entry[mode] for entry in entry_batch]
        messages = self.get_messages(testee)
        responses, _ = self.generate_llm_responses(
            messages, desc=f"Eval: Extracting {mode}"
        )
        fail_flag = False
        outputs = []
        for idx, batch in enumerate(responses):
            process_json_batch, item_failed = self.extract_one_with_retries(
                messages[idx],
                batch,
                image_path=entry_batch[idx].get("img", ""),
                idx=idx,
                mode=mode,
            )
            fail_flag = fail_flag or item_failed
            outputs.extend(process_json_batch)

        return outputs, fail_flag

    def make_eval_prompt(self, eval_pair):
        one_eval_pair = []
        for char in characteristics:
            if char in rule_based_eval_features:
                continue

            try:
                A_content = eval_pair[0][char]
            except KeyError:
                A_content = ""
            try:
                B_content = eval_pair[1][char]
            except KeyError:
                B_content = ""
            one_eval_pair.append(f"- {char}\nGenerated:{A_content}\nReference:{B_content}")
        return "\n".join(one_eval_pair)

    def evaluate(self, outputs: list[dict], references: list[dict], caption_batch: list[dict]):
        assert len(outputs) == len(references)
        fail_flag = False
        detailed_scores = []

        prompts = [self.make_eval_prompt(eval_pair) for eval_pair in zip(outputs, references)]
        prompts = self.get_messages(prompts)
        responses, fail_flag = self.generate_llm_responses(prompts, desc="Eval: Evaluating", batch_size=1)

        for idx, batch in enumerate(responses):
            score_dict = {"image_path": caption_batch[idx].get("img", "")}
            for char in characteristics:
                score_dict[char] = {"reason": "", "rating": 0}

            try:
                for response in batch:
                    json_block, remaining_text = find_first_json_block(response)
                    detailed_score = json.loads(json_block)
                    for char in detailed_score.keys():
                        score_dict[char] = detailed_score[char]

                for char in characteristics:
                    if len(references[idx].get(char, "")) == 0:
                        score_dict[char] = {"reason": "Reference is empty, skipped evaluation", "rating": -1}
            except Exception as e:
                record_err(prompts[idx], batch[0], e, idx, "evaluation")
                fail_flag = True
            detailed_scores.append(score_dict)

        return detailed_scores, fail_flag


def main():
    evaluater = Evaluater()
    evaluater.load_llm_generator()
    caption_batch = load_caption_batch()
    if not os.path.exists(fossil_eval_args.eval_result_dir):
        os.makedirs(fossil_eval_args.eval_result_dir, exist_ok=True)

    # Extract-only runs do not need reference descriptions. This is used for the
    # 882-image analysis set, where the extracted model features are merged into
    # the visual expert's numeric CSV template rather than scored against labels.
    if fossil_eval_args.extract_only:
        if fossil_eval_args.read_extractions_from_file:
            with open(f"{fossil_eval_args.eval_result_dir}/extracted_output_info.json", "r") as f:
                normalize_extracted_batch(json.load(f))
        else:
            ex_o, fail = evaluater.extract(caption_batch, mode="output")
            with open(f"{fossil_eval_args.eval_result_dir}/extracted_output_info.json", "w") as f:
                json.dump(ex_o, f, indent=4)
            if fail:
                print("Fail Detected, check log file; output extraction is incomplete")
        return

    # Read reference features info
    if not os.path.exists(fossil_eval_args.eval_reference_file):
        if not fossil_eval_args.eval_reference_desc_file and fossil_eval_args.eval_output_file:
            raise ValueError(
                "eval_reference_desc_file is required when eval_reference_file does not point to "
                "an existing extracted reference JSON"
            )
        ex_r, fail = evaluater.extract(caption_batch, mode="reference")
        reference_dir = os.path.dirname(fossil_eval_args.eval_reference_file)
        if reference_dir:
            os.makedirs(reference_dir, exist_ok=True)
        with open(fossil_eval_args.eval_reference_file, "w") as f:
            json.dump(ex_r, f, indent=4)
        if fail:
            print(
                "Fail Detected, check log file; program aborted due to unabling to carry on until this error is fixed manually"
            )
            return
    else:
        with open(fossil_eval_args.eval_reference_file, "r") as f:
            ex_r = normalize_extracted_batch(
                align_extracted_reference(caption_batch, json.load(f))
            )
        logger.info(f"Loaded reference features info from {fossil_eval_args.eval_reference_file}")

    # Read output features info
    if fossil_eval_args.read_extractions_from_file:
        with open(f"{fossil_eval_args.eval_result_dir}/extracted_output_info.json", "r") as f:
            ex_o = normalize_extracted_batch(json.load(f))
    else:
        # Extract output feature info
        ex_o, fail = evaluater.extract(caption_batch, mode="output")
        with open(f"{fossil_eval_args.eval_result_dir}/extracted_output_info.json", "w") as f:
            json.dump(ex_o, f, indent=4)
        if fail:
            print("Fail Detected, check log file; carry on to independent reference extraction")
            return

    evaluater.reload_eval_mode()
    assert len(ex_r) == len(
        ex_o
    ), f"Failed extraction valid test, some extractions are not at correct length: ex_o:{len(ex_o)}, ex_r:{len(ex_r)}, caption_batch:{len(caption_batch)}"
    # Evaluation
    detailed, fail = evaluater.evaluate(ex_o, ex_r, caption_batch)
    with open(f"{fossil_eval_args.eval_result_dir}/detailed_score_list.json", "w") as f:
        json.dump(detailed, f, indent=4)

    # ---------debug---------
    # with open(f"{fossil_eval_args.eval_result_dir}/detailed_score_list.json", "r") as f:
    #     detailed = json.load(f)
    #     fail = False

    # Replace the numerical features with manually caculated ones
    detailed = rule_based_eval(detailed, ex_o, ex_r)

    with open(f"{fossil_eval_args.eval_result_dir}/detailed_score_list.json", "w") as f:
        json.dump(detailed, f, indent=4)
    if fail:
        print("Fail Detected, check log file; program aborted")
        return
    statistics()


def rule_based_eval(detailed_score_list, extracted_output_info, extracted_reference_info):
    new_detailed = []
    for detail, output, reference in zip(
        detailed_score_list, extracted_output_info, extracted_reference_info
    ):
        new_detail = detail.copy()

        for feature in rule_based_eval_features:
            if new_detail[feature].get("rating") == -1:
                continue

            # Calculate scores for numerical features
            if feature in ["length", "width", "ratio", "number_of_volutions", "proloculus", "tunnel_angles"]:
                if not isinstance(reference[feature], str):
                    reference[feature] = str(reference[feature])
                if not isinstance(output[feature], str):
                    output[feature] = str(output[feature])
                ref_range = extract_range_or_num(reference[feature])
                pred_range = extract_range_or_num(output[feature])

                score = calculate_score(ref_range, pred_range)
                new_detail[feature][
                    "reason"
                ] = f"Rule-based eval with output:{output[feature]}->{pred_range}, reference:{reference[feature]}->{ref_range}"
                new_detail[feature]["rating"] = score

            # Calculate scores for chomata
            elif feature == "chomata":
                rating = calculate_chomata_rating(output["chomata"], reference["chomata"])
                new_detail["chomata"]["rating"] = rating
                new_detail["chomata"][
                    "reason"
                ] = f"Rule-based eval with output:{output['chomata']}, reference:{reference['chomata']}"

            # Calculate scores for tunnel shape
            ### 已使用LLM评估，不使用
            # elif feature == "tunnel_shape":
            #     height_output, width_output = extract_tunnel_shape(
            #         output["tunnel_shape"], default_value="none"
            #     )
            #     height_reference, width_reference = extract_tunnel_shape(
            #         reference["tunnel_shape"], default_value="moderate"
            #     )
            #     rating = 0
            #     if height_reference == height_output:
            #         rating += 5
            #     elif height_output == "none":
            #         rating += 0
            #     elif height_reference == "moderate":
            #         rating += 2

            #     if width_reference == width_output:
            #         rating += 5
            #     elif width_output == "none":
            #         rating += 0
            #     elif width_reference == "moderate":
            #         rating += 2

            #     new_detail["tunnel_shape"]["rating"] = rating
            #     new_detail["tunnel_shape"][
            #         "reason"
            #     ] = f"Rule-based eval with output:{output['tunnel_shape']}, reference:{reference['tunnel_shape']}"

            # Calculate scores for axis shape
            elif feature == "axis_shape":
                rating = calculate_axis_shape_rating(output["axis_shape"], reference["axis_shape"])
                new_detail["axis_shape"]["rating"] = rating
                new_detail["axis_shape"][
                    "reason"
                ] = f"Rule-based eval with output:{output['axis_shape']}, reference:{reference['axis_shape']}"

        new_detailed.append(new_detail)
    return new_detailed


if __name__ == "__main__":
    main()
