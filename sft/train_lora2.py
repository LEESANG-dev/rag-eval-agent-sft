#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2차 LoRA 학습 — 에이전트 궤적(도구 호출 포함) SFT

1차와의 차이:
  - 샘플이 다중 턴: system / user / assistant(tool_calls) / tool / assistant
  - 손실은 '모든 assistant 턴'에만 건다 (도구 호출 결정 + 최종 답변 둘 다 학습)
    system·user·tool 턴은 -100 마스킹 — 도구 출력을 생성하도록 배우면 안 되므로
  - chat template 에 tools 스펙을 전달해 서빙(vLLM hermes)과 동일한 렌더링을 보장

스팬 계산은 generation prompt 없이 누적 렌더 길이로 잡는다:
  start_i = render(messages[:i]) 길이 / end_i = render(messages[:i+1]) 길이
  -> [start_i, end_i) 가 assistant 턴 i (헤더 포함, 관례상 무해)
"""
import os, json, argparse, torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model


def unwrap_ids(out):
    """apply_chat_template 반환형(list/BatchEncoding/Encoding) 흡수"""
    if isinstance(out, list):
        return out if out and isinstance(out[0], int) else out[0]
    if hasattr(out, "input_ids"):
        ids = out.input_ids
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return ids[0] if ids and isinstance(ids[0], list) else ids
    if hasattr(out, "ids"):
        return out.ids
    raise TypeError(f"unknown template output: {type(out)}")


class TraceDataset(Dataset):
    def __init__(self, path, tokenizer, max_len, tool_cap=8000):
        self.samples = []
        skipped_long, skipped_bad = 0, 0
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            msgs, tools = r["messages"], r.get("tools")
            # 도구 출력이 과도하게 길면 시퀀스 예산을 위해 뒤를 자른다
            for m in msgs:
                if m["role"] == "tool" and len(m.get("content", "")) > tool_cap:
                    m["content"] = m["content"][:tool_cap] + "\n...(생략)"
            try:
                ids, labels = self.build(msgs, tools, tokenizer)
            except Exception:
                skipped_bad += 1
                continue
            if ids is None or len(ids) > max_len:
                skipped_long += 1
                continue
            self.samples.append((ids, labels))
        print(f"{os.path.basename(path)}: {len(self.samples)}건 사용, "
              f"길이 초과 {skipped_long} / 렌더 실패 {skipped_bad} 제외")

    @staticmethod
    def build(msgs, tools, tok):
        # 누적 렌더 길이로 스팬을 잡으면 Qwen3 템플릿이 '마지막 assistant 턴'에만
        # 빈 think 블록을 붙이는 비대칭 때문에 경계가 어긋난다(다음 턴 헤더로 누수).
        # 대신 전체 토큰열에서 <|im_start|>assistant ... <|im_end|> 블록을 직접 스캔한다.
        full = unwrap_ids(tok.apply_chat_template(
            msgs, tools=tools, tokenize=True, add_generation_prompt=False))
        im_start = tok.convert_tokens_to_ids("<|im_start|>")
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        role_ids = tok.encode("assistant", add_special_tokens=False)

        labels = [-100] * len(full)
        trained, i, n = 0, 0, len(full)
        while i < n:
            if full[i] == im_start and full[i + 1:i + 1 + len(role_ids)] == role_ids:
                j = i
                while j < n and full[j] != im_end:
                    j += 1
                j = min(j, n - 1)
                for k in range(i, j + 1):        # <|im_end|>까지 포함해 종료도 학습
                    labels[k] = full[k]
                trained += j + 1 - i
                i = j + 1
            else:
                i += 1
        if trained == 0:
            return None, None
        return full, labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ids, labels = self.samples[i]
        return {"input_ids": torch.tensor(ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.ones(len(ids), dtype=torch.long)}


def collate(batch):
    ml = max(len(b["input_ids"]) for b in batch)
    pad = lambda t, v: torch.cat([t, torch.full((ml - len(t),), v, dtype=t.dtype)])
    return {"input_ids": torch.stack([pad(b["input_ids"], 0) for b in batch]),
            "labels": torch.stack([pad(b["labels"], -100) for b in batch]),
            "attention_mask": torch.stack([pad(b["attention_mask"], 0) for b in batch])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--check", action="store_true", help="첫 샘플 마스킹 육안 검증 후 종료")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)

    if a.check:
        line = open(a.train, encoding="utf-8").readline()
        r = json.loads(line)
        ids, labels = TraceDataset.build(r["messages"], r.get("tools"), tok)
        print("=== 학습되는 토큰(assistant 턴)만 디코드 ===")
        chunk, out = [], []
        for t, l in zip(ids, labels):
            if l != -100:
                chunk.append(t)
            elif chunk:
                out.append(tok.decode(chunk)); chunk = []
        if chunk:
            out.append(tok.decode(chunk))
        for i, c in enumerate(out):
            print(f"--- 스팬 {i+1} ---")
            print(c[:400])
        return

    train_ds = TraceDataset(a.train, tok, a.max_len)
    val_ds = TraceDataset(a.val, tok, a.max_len)

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=a.accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=15,
        bf16=True, logging_steps=10,
        eval_strategy="steps", eval_steps=50, per_device_eval_batch_size=1,
        save_strategy="no", report_to=[], seed=42,
        remove_unused_columns=False, dataloader_num_workers=2)

    tr = Trainer(model=model, args=args, train_dataset=train_ds,
                 eval_dataset=val_ds, data_collator=collate)
    print(f"학습 시작: train {len(train_ds)} / val {len(val_ds)} / 유효 배치 {a.accum}")
    tr.train()

    m = tr.evaluate()
    import math
    print(f"최종 val loss: {m['eval_loss']:.4f} | ppl: {math.exp(m['eval_loss']):.2f}")
    final = os.path.join(a.out, "final")
    model.save_pretrained(final)
    tok.save_pretrained(final)
    print(f"어댑터 저장: {final}")


if __name__ == "__main__":
    main()
