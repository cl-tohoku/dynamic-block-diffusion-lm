import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, default="data/sample.txt")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--thresholds", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    parser.add_argument("--output_dir", type=Path, default="outputs")
    return parser.parse_args()


def compute_surprisal(token_ids: list[int], model):
    input_ids = torch.tensor(token_ids).to(model.device)
    with torch.no_grad():
        logits = model(input_ids).logits

    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs[:-1, :].gather(dim=1, index=input_ids[1:].unsqueeze(-1)).squeeze(-1)  # [seq_len, vocab_size]を想定
    surprisals = -target_log_probs
    total_surprisal = surprisals.sum().item()
    return total_surprisal


def segment_token_ids(tokens: list[int], tokenizer, model, threshold: float):
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    start = 0
    blocks = []
    while start < len(token_ids):
        for end in range(start + 1, len(token_ids)):
            block_ids = token_ids[start: end + 1]
            surprisal = compute_surprisal(token_ids=block_ids, model=model)
            if surprisal > threshold:
                blocks.append(tokens[start: end])
                start = end
                break
        else:
            blocks.append(tokens[start:])
            break
    return blocks


def replace_space_alias_to_space(text):
    return text.replace("Ġ", " ")


def print_colored_segments(text: str, prefix: str):
    segments = text.split("|")
    colors = ["\033[94m", "\033[92m"]  # 青と緑
    reset = "\033[0m"

    print(f"{prefix}", end=" ")
    for i, segment in enumerate(segments):
        color = colors[i % len(colors)]
        print(f"{color}{segment}{reset}", end="")
    print()


def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    with open(args.input_path, mode="r") as f:
        data = [line.strip() for line in f.readlines()]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    model.eval()

    for threshold in args.thresholds:
        print(f"{threshold=}")
        results = []
        for idx, datum in enumerate(data):
            tokens = tokenizer.tokenize(datum)
            blocks = segment_token_ids(tokens=tokens, tokenizer=tokenizer, model=model, threshold=threshold)
            joined_blocks = "|".join(["".join(block) for block in blocks])
            joined_blocks = replace_space_alias_to_space(text=joined_blocks)
            results.append(joined_blocks)
            print_colored_segments(joined_blocks, prefix=f"({idx})")

        output_path = args.output_dir / f"{threshold}.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, mode="w") as f:
            f.writelines([result + "\n" for result in results])


if __name__ == "__main__":
    args = parse_args()
    main(args)
