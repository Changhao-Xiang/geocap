import json
import sys

from common.args import caption_args, data_args, run_args
from common.llm import generator_mapping, model_path_mapping


class Paraphraser:
    def __init__(self) -> None:
        self.loaded_llm = False

    def load_llm_generator(self):
        """Initialize the LLM generator for paraphrasing"""
        assert not self.loaded_llm
        # Initialize llm
        model_name, model_id = caption_args.caption_llm.split("-", 1)
        model_path = model_path_mapping[model_name].format(model_id)
        self.llm_generator = generator_mapping[model_name](model_path, max_tokens=4096, temperature=1.3)
        self.model_name = model_name
        self.loaded_llm = True

        # Initialize prompt
        self.sys_prompt = "You are a helpful assistant skilled in text paraphrasing."
        with open(caption_args.paraphrase_prompt_dir, "r") as f:
            self.user_prompt = f.read()

    def _generate_paraphrase_prompt(self, texts: list[str]) -> list[list[dict[str, str]]]:
        """Generate the prompt for paraphrasing"""
        user_prompts = [f"{self.user_prompt.replace('{text}', text)}" for text in texts]
        if "api" not in self.model_name:
            messages = [
                [{"role": "system", "content": self.sys_prompt}, {"role": "user", "content": user_prompt}]
                for user_prompt in user_prompts
            ]
        else:
            messages = [[{"role": "user", "content": user_prompt}] for user_prompt in user_prompts]
        return messages

    def __call__(self, texts: list[str]) -> list[str]:
        """Paraphrase a single text input"""
        if not self.loaded_llm:
            self.load_llm_generator()

        messages = self._generate_paraphrase_prompt(texts)
        responses = self.llm_generator(messages, batch_size=min(len(texts), caption_args.caption_batchsize))

        outputs = []
        for batch in responses:
            outputs.extend(batch)

        return outputs


def main():
    paraphraser = Paraphraser()
    # Read original captions
    with open(data_args.caption_path, "r") as f:
        file_ext = data_args.caption_path.split(".")[-1]
        if file_ext == "jsonl":
            captions = [json.loads(line) for line in f]
        elif file_ext == "json":
            captions = json.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {data_args.caption_path}")

    output_path = data_args.caption_path.replace(f".{file_ext}", f"_paraphrased.{file_ext}")
    if run_args.end_pos != sys.maxsize:
        captions = captions[run_args.start_pos : run_args.end_pos]
        output_path = output_path.replace(
            f".{file_ext}", f"_{run_args.start_pos}_{run_args.end_pos}.{file_ext}"
        )

    # Extract and paraphrase outputs
    original_outputs = [caption["output"] for caption in captions]
    paraphrased_outputs = paraphraser(original_outputs)

    # Replace original outputs with paraphrased ones
    for caption, paraphrased_output in zip(captions, paraphrased_outputs):
        caption["output"] = paraphrased_output

    with open(output_path, "w") as f:
        for caption in captions:
            f.write(json.dumps(caption) + "\n")


if __name__ == "__main__":
    main()
