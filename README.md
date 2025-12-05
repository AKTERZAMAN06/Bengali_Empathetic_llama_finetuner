# Llama 3.1 Bengali Empathetic Fine-Tuner

This project implements an efficient, object-oriented fine-tuning pipeline for the **Llama 3.1 8B Instruct** model, specifically adapted for the **Bengali Empathetic Conversations** dataset.

The goal of this repository is to demonstrate how to fine-tune Large Language Models (LLMs) on resource-constrained environments (e.g., free-tier T4 GPUs) while maintaining high architectural standards and solving hardware synchronization challenges.

## Project Overview

- **Model:** `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
- **Domain:** Bengali NLP (Empathetic Dialogue Generation)
- **Technique:** QLoRA (Quantized Low-Rank Adaptation) via Unsloth
- **Infrastructure:** optimized for Dual Tesla T4 GPUs (Model Parallelism)

## Key Engineering Features

The codebase is structured to demonstrate robust software engineering principles in AI development:

### 1. Advanced Architecture
- **Strategy Pattern:** Implemented a flexible `FineTuningStrategy` interface to allow dynamic switching between training backends (e.g., Unsloth vs. standard PEFT).
- **Object-Oriented Design (OOD):** Modular classes (`DatasetProcessor`, `LLAMAFineTuner`, `Evaluator`) ensure the code is scalable and easy to maintain.

### 2. Hardware Optimization & Model Parallelism
One of the core challenges addressed in this project is running an 8B parameter model on limited hardware.
- **Challenge:** Standard Data Parallelism on dual T4 GPUs often leads to communication deadlocks and OOM (Out of Memory) errors with quantized models.
- **Solution:** Implemented a **Model Parallelism** strategy using `device_map="auto"`. This intelligently shards the model layers across multiple GPUs (e.g., Layers 0-8 on GPU0, Layers 9-32 on GPU1), utilizing the collective VRAM without the overhead of inter-GPU gradient synchronization.

## Technical Stack

- **Frameworks:** PyTorch, Transformers, TRL (Transformer Reinforcement Learning)
- **Optimization:** Unsloth (Fast Llama patching), Bitsandbytes (4-bit quantization), PEFT (LoRA)
- **Evaluation:** BLEU, ROUGE, and Perplexity metrics

## Repository Structure

- `Bengali_Empathetic_Llama_Finetuner.ipynb`: The end-to-end execution pipeline.
- `LLAMAExperiments.csv`: Automated logging of training loss and validation metrics.
- `GeneratedResponses.csv`: Qualitative log of model inputs and generated Bengali responses.
- `lora_model/`: Saved Low-Rank Adaptation (LoRA) weights.

## Performance

The fine-tuning process demonstrated rapid convergence:
- **Training Loss:** Reduced from ~18.4 to ~5.6 within 30 steps.
- **Inference:** The model successfully adapted to the Bengali language, generating contextually relevant and empathetic advice.

## How to Run

1. Open the notebook `Bengali_Empathetic_Llama_Finetuner.ipynb` in a Kaggle or Colab environment.
2. Select an accelerator with at least 15GB VRAM (e.g., T4 x2).
3. Upload the `BengaliEmpatheticConversationsCorpus.csv` dataset.
4. Run the pipeline sequentially.

## Dependencies

- unsloth
- transformers
- trl
- peft
- accelerate
- bitsandbytes
- evaluate
