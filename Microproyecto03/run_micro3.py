"""
Pipeline optimizado de alto desempeno para MAPLN Microproyecto 3 - Dayana Vega
Mejoras clave implementadas:
1. Prompt enriquecido con test_list[0] para condicionar la firma y nombre exacto de la funcion.
2. Adaptador LoRA en todas las capas lineales (atencion + MLP: q, k, v, o, gate, up, down).
3. Rango aumentado a r=32, lora_alpha=64 para maxima capacidad representacional.
4. Entrenamiento a 5 epocas con optimizador AdamW y enmascaramiento estricto de prompt (-100).
5. Evaluacion pass@1 por lotes con proteccion de ejecucion en namespace aislado.
"""

import os
import gc
import json
import time
import random
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset

def format_prompt(instruction: str, example_test: str = "") -> str:
    """Prompt enriquecido con el caso de prueba guía para guiar la firma de la función."""
    inst = instruction.strip()
    if example_test:
        return f"# Task: {inst}\n# Example test: {example_test.strip()}\n# Solution in Python:\n"
    return f"# Task: {inst}\n# Solution in Python:\n"

class MBPPFinetuningDataset(Dataset):
    def __init__(self, data_split, tokenizer, max_length: int = 512):
        self.examples = []
        for item in data_split:
            first_test = item["test_list"][0] if item.get("test_list") else ""
            prompt = format_prompt(item["text"], first_test)
            code = item["code"].strip() + tokenizer.eos_token
            
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            code_ids = tokenizer.encode(code, add_special_tokens=False)
            
            input_ids = prompt_ids + code_ids
            labels = [-100] * len(prompt_ids) + code_ids
            
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
                
            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn_pad(batch, pad_token_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    labels = []
    attention_mask = []
    
    for x in batch:
        cur_len = len(x["input_ids"])
        pad_len = max_len - cur_len
        
        input_ids.append(torch.cat([x["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        labels.append(torch.cat([x["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([torch.ones(cur_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
        
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask)
    }

@torch.no_grad()
def generate_batched(model, tokenizer, prompts: List[str], batch_size: int = 16, max_new_tokens: int = 256, max_length: int = 512) -> List[str]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.eval()
    solutions = []
    device_model = next(model.parameters()).device
    
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device_model)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        gen = out[:, inputs["input_ids"].shape[1]:]
        solutions.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return solutions

def passes_tests(generated_code: str, test_list: List[str], setup_code: str = "") -> bool:
    code_cleaned = generated_code
    if "```python" in code_cleaned:
        code_cleaned = code_cleaned.split("```python")[1].split("```")[0]
    elif "```" in code_cleaned:
        code_cleaned = code_cleaned.split("```")[1].split("```")[0]
        
    env = {
        "input": lambda *args: "",
        "print": lambda *args, **kwargs: None
    }
    try:
        exec(setup_code + "\n" + code_cleaned, env)
        for assertion in test_list:
            exec(assertion, env)
        return True
    except Exception:
        return False

def evaluate_pass_at_1(model, tokenizer, test_data, batch_size: int = 16) -> Dict[str, Any]:
    prompts = [format_prompt(item["text"], item["test_list"][0] if item.get("test_list") else "") for item in test_data]
    t0 = time.time()
    solutions = generate_batched(model, tokenizer, prompts, batch_size=batch_size)
    eval_time = time.time() - t0
    
    correct = 0
    results = []
    for item, sol in zip(test_data, solutions):
        is_correct = passes_tests(sol, item["test_list"], setup_code=item.get("test_setup_code", ""))
        if is_correct:
            correct += 1
        results.append({"task_id": item["task_id"], "correct": is_correct, "solution": sol})
        
    score = (correct / len(test_data)) * 100.0
    print(f"  -> pass@1: {score:.2f}% ({correct}/{len(test_data)}) evaluado en {eval_time:.1f} s")
    return {"pass@1": score, "correct": correct, "total": len(test_data), "details": results, "time": eval_time}

def train_lora_model(model_name: str, tokenizer, train_raw, val_raw, output_dir: str, epochs: int = 5, lr: float = 2e-4, batch_size: int = 4, grad_accum_steps: int = 4):
    print(f"\n[Finetuning LoRA Optimizado] Modelo: {model_name}")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map="auto")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Ajuste fino completo en todas las proyecciones lineales (atencion + feedforward)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    train_ds = MBPPFinetuningDataset(train_raw, tokenizer)
    val_ds = MBPPFinetuningDataset(val_raw, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn_pad(b, tokenizer.pad_token_id))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn_pad(b, tokenizer.pad_token_id))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()
        t_ep = time.time()
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            
            total_train_loss += loss.item() * grad_accum_steps
            
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
        avg_train = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"].to(model.device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()
                
        avg_val = total_val_loss / len(val_loader)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        print(f"  Epoca [{epoch}/{epochs}] - Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} ({time.time()-t_ep:.1f} s)")
        
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"  Checkpoint LoRA guardado en: {output_dir}")
    
    del model, base_model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return history

def main():
    print("=" * 70)
    print("MAPLN MICROPROYECTO 3: PIPELINE MEJORADO EXPONENCIAL")
    print("=" * 70)
    
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Dispositivo: {device} | Tipo de datos: {dtype}")
    
    print("\n[1] Cargando dataset MBPP...")
    mbpp = load_dataset("google-research-datasets/mbpp", "full")
    train_raw = mbpp["train"]
    val_raw = mbpp["validation"]
    test_raw = mbpp["test"]
    print(f"Dataset listo: {len(train_raw)} train, {len(val_raw)} val, {len(test_raw)} test.")
    
    ckpt_dir = r"c:\Repo\mp1_equipo_21\Microproyecto03\checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 2. Qwen2.5-1.5B Zero-Shot con Prompt enriquecido
    print("\n[2] Evaluando Qwen2.5-1.5B Zero-Shot en Test (500 problemas)...")
    qwen_name = "Qwen/Qwen2.5-1.5B"
    tok_qwen = AutoTokenizer.from_pretrained(qwen_name)
    model_qwen = AutoModelForCausalLM.from_pretrained(qwen_name, dtype=dtype, device_map="auto")
    res_qwen_zero = evaluate_pass_at_1(model_qwen, tok_qwen, test_raw, batch_size=16)
    del model_qwen
    torch.cuda.empty_cache()
    gc.collect()
    
    # 3. Finetuning LoRA: Qwen2.5-1.5B (r=32, todas las capas lineales, 5 epocas)
    print("\n[3] Entrenando adaptador LoRA de alta capacidad para Qwen2.5-1.5B...")
    qwen_lora_dir = os.path.join(ckpt_dir, "qwen_lora_adapter")
    hist_qwen = train_lora_model(qwen_name, tok_qwen, train_raw, val_raw, qwen_lora_dir, epochs=5, lr=2e-4)
    
    # 4. Evaluacion Qwen2.5-1.5B + LoRA
    print("\n[4] Evaluando Qwen2.5-1.5B + LoRA en Test (500 problemas)...")
    base_qwen = AutoModelForCausalLM.from_pretrained(qwen_name, dtype=dtype, device_map="auto")
    model_qwen_lora = PeftModel.from_pretrained(base_qwen, qwen_lora_dir)
    res_qwen_lora = evaluate_pass_at_1(model_qwen_lora, tok_qwen, test_raw, batch_size=16)
    del model_qwen_lora, base_qwen
    torch.cuda.empty_cache()
    gc.collect()
    
    # 5. SmolLM2-1.7B Zero-Shot
    print("\n[5] Evaluando SmolLM2-1.7B Zero-Shot en Test (500 problemas)...")
    smol_name = "HuggingFaceTB/SmolLM2-1.7B"
    tok_smol = AutoTokenizer.from_pretrained(smol_name)
    model_smol = AutoModelForCausalLM.from_pretrained(smol_name, dtype=dtype, device_map="auto")
    res_smol_zero = evaluate_pass_at_1(model_smol, tok_smol, test_raw, batch_size=16)
    del model_smol
    torch.cuda.empty_cache()
    gc.collect()
    
    # 6. Finetuning LoRA: SmolLM2-1.7B (r=32, todas las capas lineales, 5 epocas)
    print("\n[6] Entrenando adaptador LoRA de alta capacidad para SmolLM2-1.7B...")
    smol_lora_dir = os.path.join(ckpt_dir, "smollm2_lora_adapter")
    hist_smol = train_lora_model(smol_name, tok_smol, train_raw, val_raw, smol_lora_dir, epochs=5, lr=2e-4)
    
    # 7. Evaluacion SmolLM2-1.7B + LoRA
    print("\n[7] Evaluando SmolLM2-1.7B + LoRA en Test (500 problemas)...")
    base_smol = AutoModelForCausalLM.from_pretrained(smol_name, dtype=dtype, device_map="auto")
    model_smol_lora = PeftModel.from_pretrained(base_smol, smol_lora_dir)
    res_smol_lora = evaluate_pass_at_1(model_smol_lora, tok_smol, test_raw, batch_size=16)
    del model_smol_lora, base_smol
    torch.cuda.empty_cache()
    gc.collect()
    
    # 8. Consolidacion de resultados
    print("\n" + "=" * 70)
    print("RESUMEN GENERAL OPTIMIZADO: pass@1 EN MBPP TEST (500 PROBLEMAS)")
    print("=" * 70)
    summary = [
        {"Configuracion": "1. Qwen2.5-1.5B (Zero-Shot)", "pass@1 (%)": round(res_qwen_zero["pass@1"], 2), "Correctos": res_qwen_zero["correct"], "Total": 500},
        {"Configuracion": "2. Qwen2.5-1.5B + LoRA", "pass@1 (%)": round(res_qwen_lora["pass@1"], 2), "Correctos": res_qwen_lora["correct"], "Total": 500},
        {"Configuracion": "3. SmolLM2-1.7B (Zero-Shot)", "pass@1 (%)": round(res_smol_zero["pass@1"], 2), "Correctos": res_smol_zero["correct"], "Total": 500},
        {"Configuracion": "4. SmolLM2-1.7B + LoRA", "pass@1 (%)": round(res_smol_lora["pass@1"], 2), "Correctos": res_smol_lora["correct"], "Total": 500},
    ]
    for row in summary:
        print(f"{row['Configuracion']:<32} | pass@1: {row['pass@1 (%)']:<6}% ({row['Correctos']}/{row['Total']})")
        
    res_path = r"c:\Repo\mp1_equipo_21\Microproyecto03\experiment_results.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "hist_qwen": hist_qwen,
            "hist_smol": hist_smol,
            "samples_qwen_lora": [res_qwen_lora["details"][i] for i in [0, 5, 12]],
            "samples_smol_lora": [res_smol_lora["details"][i] for i in [0, 5, 12]]
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados exitosamente en: {res_path}")

if __name__ == "__main__":
    main()
