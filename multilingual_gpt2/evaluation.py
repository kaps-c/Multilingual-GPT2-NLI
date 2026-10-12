"""Batched NLI evaluation and single-prompt greedy generation."""

import torch


@torch.no_grad()
def evaluate_nli(model, tokenizer, loader, device="cpu"):
    model.eval()
    label_tokens = [
        tokenizer.encode(str(i), add_special_tokens=False) for i in range(3)
    ]
    if any(len(ids) != 1 for ids in label_tokens):
        raise ValueError("NLI evaluation requires single-token digit labels")
    label_ids = [ids[0] for ids in label_tokens]
    predictions = []
    for batch in loader:
        hidden = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )["last_token"]
        logits = model.hidden_state_to_token(hidden).float()
        generated = logits.argmax(-1).cpu().tolist()
        probabilities = logits[:, label_ids].softmax(-1).cpu().tolist()
        for record, target, token, probability in zip(
            batch["records"],
            batch["label_strs"],
            generated,
            probabilities,
            strict=True,
        ):
            prediction = tokenizer.decode([token]).strip()
            predictions.append(
                {
                    **record,
                    "gold": target,
                    "prediction": prediction,
                    "correct": prediction == target,
                    "label_probabilities": probability,
                }
            )
    if not predictions:
        raise ValueError("evaluation loader is empty")
    return {
        "accuracy": sum(row["correct"] for row in predictions)
        / len(predictions),
        "coverage": sum(
            row["prediction"] in {"0", "1", "2"} for row in predictions
        )
        / len(predictions),
        "count": len(predictions),
        "predictions": predictions,
    }


@torch.no_grad()
def generate_gpt2(model, tokenizer, prompt, max_new_tokens=50, device="cpu"):
    """Return the continuation only; never parse labels from echoed prompts."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        ids = [tokenizer.eos_token_id]
    budget = model.config.max_position_embeddings - len(ids)
    if budget < max_new_tokens:
        raise ValueError(
            "prompt plus requested generation exceeds context length"
        )
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []
    for _ in range(max_new_tokens):
        hidden = model(inputs)["last_token"]
        token = model.hidden_state_to_token(hidden).argmax(-1).item()
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        inputs = torch.cat((inputs, inputs.new_tensor([[token]])), dim=1)
    return tokenizer.decode(generated, skip_special_tokens=True)
