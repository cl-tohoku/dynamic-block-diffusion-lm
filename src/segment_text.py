import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_path", type=Path, default="data/sample.txt")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--threshold", type=float, default=30.0)
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


def segment_token_ids(token_ids: list[int], model, threshold: float):
    start = 0
    blocks = []
    while start < len(token_ids):
        for end in range(start + 1, len(token_ids)):
            block_ids = token_ids[start: end + 1]
            surprisal = compute_surprisal(token_ids=block_ids, model=model)
            if surprisal > threshold:
                blocks.append(token_ids[start: end])
                start = end
                break
        else:
            blocks.append(token_ids[start:])
            break
    return blocks


def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    with open(args.txt_path, mode="r") as f:
        data = [line.strip() for line in f.readlines()]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    model.eval()

    for datum in data:
        token_ids = tokenizer(datum)["input_ids"]
        decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids)  # デバッグ用

        blocks = segment_token_ids(token_ids=token_ids, model=model, threshold=args.threshold)
        blocks = [tokenizer.convert_ids_to_tokens(block) for block in blocks]
        joined_blocks = "|".join(["".join(block) for block in blocks]).replace("Ġ", " ")
        print(joined_blocks)


if __name__ == "__main__":
    args = parse_args()
    main(args)
