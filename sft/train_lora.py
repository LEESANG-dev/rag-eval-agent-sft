#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
연구기관 규정 QA — Qwen3-8B LoRA SFT

의존성 최소화: transformers.Trainer + peft 만 사용 (trl/datasets 불필요).
vLLM cu129 컨테이너(torch 2.11, sm_120) 안에서 실행한다.

  pip install --no-index /sft/wheels/peft-*.whl
  python3 train_lora.py --model /models/Qwen3-8B \
      --train /sft/sft_train_data.jsonl --val /sft/sft_train_data_val.jsonl \
      --out /sft/out/lora-r16

손실은 assistant 응답 토큰에만 걸리게 프롬프트를 -100 마스킹한다.
"""
import os, json, argparse, math, random

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model

random.seed(42)


def encode_chat(tok, msgs, add_gen):
    """apply_chat_template 반환형이 버전마다 다름 — 전부 흡수.
    (v5: BatchEncoding(UserDict) / 구버전: list / Encoding / str)"""
    out = tok.apply_chat_template(msgs, add_generation_prompt=add_gen,
                                  tokenize=True, enable_thinking=False)
    if isinstance(out, str):
        return tok(out, add_special_tokens=False)["input_ids"]
    try:
        ids = out["input_ids"]                     # BatchEncoding / dict
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)
    except (TypeError, KeyError, IndexError):
        pass
    if hasattr(out, "ids"):                        # tokenizers.Encoding
        return list(out.ids)
    return list(out)                               # plain list


class ChatSFTDataset(Dataset):
    """JSONL(messages) -> input_ids/labels. 프롬프트 구간은 -100."""

    def __init__(self, path, tokenizer, max_len):
        self.tok = tokenizer
        self.max_len = max_len
        self.rows = []
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = json.loads(line)["messages"]
                prompt_ids = encode_chat(tokenizer, m[:-1], True)
                full_ids = encode_chat(tokenizer, m, False)
                if len(full_ids) > max_len or len(full_ids) <= len(prompt_ids):
                    skipped += 1
                    continue
                labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
                self.rows.append((full_ids, labels))
        print(f"{os.path.basename(path)}: {len(self.rows)}건 사용, "
              f"{skipped}건 제외(길이 초과 등)")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    mx = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = mx - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return {"input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("모델 로드 (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lcfg = LoraConfig(
        r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    train_ds = ChatSFTDataset(a.train, tok, a.max_len)
    val_ds = ChatSFTDataset(a.val, tok, a.max_len)

    args = TrainingArguments(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.accum,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=args,
                      train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id))

    print(f"학습 시작: train {len(train_ds)} / val {len(val_ds)} / "
          f"유효 배치 {a.batch * a.accum}")
    trainer.train()

    model.save_pretrained(os.path.join(a.out, "final"))
    tok.save_pretrained(os.path.join(a.out, "final"))
    m = trainer.evaluate()
    print("최종 val loss:", round(m.get("eval_loss", float("nan")), 4),
          "| ppl:", round(math.exp(m.get("eval_loss", 0)), 2))
    print("어댑터 저장:", os.path.join(a.out, "final"))


if __name__ == "__main__":
    main()
