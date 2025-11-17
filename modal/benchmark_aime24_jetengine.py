"""
Modal script for benchmarking SDAR models on AIME24 dataset using JetEngine.

Usage:
    # AIME 2024 (30 problems), pass@1
    modal run modal/benchmark_aime24_jetengine.py

    # AIME 2024, pass@32
    modal run modal/benchmark_aime24_jetengine.py --num-runs 32

    # Historical AIME (600+ problems)
    modal run modal/benchmark_aime24_jetengine.py --dataset-name di-zhang-fdu/AIME_1983_2024

    # Help
    modal run modal/benchmark_aime24_jetengine.py --help
"""
import modal
import json
from pathlib import Path

# Define Modal app
app = modal.App("sdar-aime24-jetengine")

CUDA_TAG = "12.9.1-devel-ubuntu22.04"  # CUDA 12.9 devel image

# Create Modal image with NVIDIA CUDA devel base for flash-attn compilation
# (Reuse same image setup as GSM8K benchmark)
image = (
    modal.Image
    # 1) Start from NVIDIA CUDA "devel" image so we have nvcc + full toolkit
    .from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.11")
    .entrypoint([])  # optional: silence base image entrypoint
    # 2) Basic build tooling + git
    .apt_install("git", "build-essential")
    # Make sure CUDA_HOME is explicitly set (usually already correct, but safe)
    .env({"CUDA_HOME": "/usr/local/cuda"})
    # 3) Install PyTorch 2.8.0 compiled for CUDA 12.9 (cu129 wheels)
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        extra_index_url="https://download.pytorch.org/whl/cu129",
        extra_options="--index-strategy unsafe-best-match",
    )
    # 4) Your other Python deps
    .uv_pip_install(
        "transformers==4.52.4",
        "datasets>=2.16.0",
        "accelerate>=1.3.0",
        "huggingface_hub",
        "einops",
        "numpy<2.0.0",
        "tqdm",
        "packaging",
        "ninja",
        "wheel",
    )
    # 5) Install flash-attn AFTER torch is in place
    .uv_pip_install(
        "flash-attn==2.8.3",       # pin to a recent, torch2.8-compatible version
        extra_options="--no-build-isolation",
    )
    # Optional: quick sanity check during build
    .run_commands(
        "python -c \"import torch; print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda); import flash_attn; print('flash_attn OK')\""
    )
    # 6) Install JetEngine from local directory (copy=True to allow subsequent build steps)
    .add_local_dir("thirdparty/JetEngine", remote_path="/root/JetEngine", copy=True)
    .run_commands(
        "pip install /root/JetEngine",
    )
    # 7) Your local helper file (added last for fast redeployment without rebuild)
    .add_local_file("modal/benchmark_utils.py", remote_path="/root/benchmark_utils.py")
)

# Volumes for caching models and storing results
models_volume = modal.Volume.from_name("sdar-models-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("sdar-results", create_if_missing=True)

# Constants
MODELS_DIR = "/models"
RESULTS_DIR = "/results"


@app.function(
    image=image,
    gpu="H100",
    volumes={
        MODELS_DIR: models_volume,
        RESULTS_DIR: results_volume,
    },
    timeout=14400,  # 4 hours (AIME needs more time for multiple runs)
    memory=65536,  # 64GB RAM
)
def run_aime24_benchmark(
    model_name: str = "JetLM/SDAR-4B-Chat",
    dataset_name: str = "HuggingFaceH4/aime_2024",
    num_samples: int | None = None,
    num_runs: int = 1,
    block_length: int = 4,
    denoising_steps: int = 4,
    temperature: float = 1.0,
    remasking_strategy: str = "low_confidence_dynamic",
    confidence_threshold: float = 0.9,
    max_gen_length: int = 2048,
    save_results: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run AIME24 benchmark on SDAR model using JetEngine.

    Args:
        model_name: HuggingFace model ID or local path
        dataset_name: HuggingFace dataset ID for AIME problems
        num_samples: Number of problems to evaluate (None = all)
        num_runs: Number of runs per problem (1, 8, or 32 for pass@k)
        block_length: SDAR block length (default: 4)
        denoising_steps: Number of denoising steps (default: 4)
        temperature: Sampling temperature (0.0 = greedy, 1.0 = sampling)
        remasking_strategy: Token remasking strategy (default: low_confidence_dynamic)
        confidence_threshold: Threshold for dynamic remasking (default: 0.9)
        max_gen_length: Maximum generation length in tokens (2048 for AIME)
        save_results: Whether to save detailed results to JSON
        verbose: Show detailed generation logs

    Returns:
        Dict with evaluation metrics and results
    """
    from jetengine import LLM, SamplingParams
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from tqdm import tqdm
    import time
    import os
    from datetime import datetime

    # Import helper functions
    import sys
    sys.path.append("/root")
    from benchmark_utils import (
        extract_answer_aime,
        normalize_answer_aime,
        evaluate_answer_aime,
        format_prompt_aime,
        compute_aime_metrics,
    )

    # Capture start time
    run_datetime = datetime.now()
    timestamp_str = run_datetime.strftime("%Y%m%d_%H%M%S")

    # Print configuration
    print("=" * 80)
    print("SDAR AIME24 Benchmark (JetEngine)")
    print("=" * 80)
    print(f"Run datetime: {run_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Number of runs per problem: {num_runs}")
    print(f"Block length: {block_length}")
    print(f"Denoising steps: {denoising_steps}")
    print(f"Temperature: {temperature}")
    print(f"Remasking strategy: {remasking_strategy}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Max generation length: {max_gen_length}")
    print("=" * 80)

    # Download model from HuggingFace to local cache or use existing cached model
    print("\n[1/5] Loading model...")

    # Check if model_name is a local cache name or a HuggingFace model ID
    if "/" in model_name:
        # HuggingFace model ID - download if needed
        model_local_name = model_name.replace("/", "_")
        model_path = f"{MODELS_DIR}/{model_local_name}"

        if not os.path.exists(model_path):
            print(f"  Downloading {model_name} to {model_path}...")
            snapshot_download(
                repo_id=model_name,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )
        else:
            print(f"  Using cached model at {model_path}")
    else:
        # Local cache name - use directly
        model_path = f"{MODELS_DIR}/{model_name}"
        if os.path.exists(model_path):
            print(f"  Using local cached model: {model_name}")
        else:
            raise FileNotFoundError(
                f"Model not found in cache: {model_path}\n"
                f"Available models in {MODELS_DIR}: {os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else 'none'}"
            )

    # Load tokenizer
    print("\n[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Initialize JetEngine LLM
    print("\n[3/5] Initializing JetEngine...")
    start_time = time.time()

    llm = LLM(
        model_path,  # Use local path instead of HF model ID
        enforce_eager=True,
        tensor_parallel_size=1,
        mask_token_id=151669,  # SDAR mask token ID
        block_length=block_length
    )

    print(f"JetEngine initialized in {time.time() - start_time:.2f}s")

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        topk=0,
        topp=1.0,
        max_tokens=max_gen_length,
        remasking_strategy=remasking_strategy,
        block_length=block_length,
        denoising_steps=denoising_steps,
        dynamic_threshold=confidence_threshold
    )

    # Load AIME dataset
    print("\n[4/5] Loading AIME dataset...")
    dataset = load_dataset(dataset_name, split="train")

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} problems")
    print(f"Running {num_runs} attempts per problem")

    # Run evaluation with multiple attempts per problem
    print("\n[5/5] Running evaluation...")
    problem_results = []
    total_gen_time = 0.0

    # Batching configuration (smaller than GSM8K due to longer generations)
    batch_size = 8  # Process 8 problems at a time

    # Create progress bar
    total_evals = len(dataset) * num_runs
    pbar = tqdm(
        total=total_evals,
        desc="AIME Eval",
        unit="run",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    # Process each problem
    for problem_idx, example in enumerate(dataset):
        problem_id = example.get("id", problem_idx)
        problem_text = example.get("problem", "")
        ground_truth = example.get("answer", "")

        if verbose:
            print(f"\n{'='*80}")
            print(f"Problem {problem_idx + 1}/{len(dataset)} (ID: {problem_id})")
            print(f"{'='*80}")
            print(f"Problem: {problem_text[:200]}..." if len(problem_text) > 200 else f"Problem: {problem_text}")
            print(f"Ground truth: {ground_truth}")

        # Multiple runs for this problem
        attempts = []

        for run_idx in range(num_runs):
            try:
                # Format prompt
                prompt_text = format_prompt_aime(problem_text, include_instruction=True)
                messages = [{"role": "user", "content": prompt_text}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )

                # Generate
                gen_start = time.time()
                outputs = llm.generate_streaming(
                    [formatted_prompt],
                    sampling_params,
                    max_active=256,
                    use_tqdm=False
                )
                gen_time = time.time() - gen_start
                total_gen_time += gen_time

                # Extract output
                output = outputs[0]
                output_text = output['text']
                cleaned_text = output_text.replace('<|MASK|>', '')

                # Extract answer (integer 0-999)
                predicted_answer = extract_answer_aime(cleaned_text)

                # Evaluate
                is_correct = evaluate_answer_aime(predicted_answer, ground_truth)

                # Calculate token metrics
                output_len = len(tokenizer.encode(output_text))
                tokens_per_sec = output_len / gen_time if gen_time > 0 else 0

                if verbose:
                    print(f"\n  Run {run_idx + 1}/{num_runs}:")
                    print(f"    Generated ({output_len} tokens in {gen_time:.2f}s = {tokens_per_sec:.1f} tok/s)")
                    print(f"    Predicted: {predicted_answer}")
                    print(f"    Correct: {'✓' if is_correct else '✗'}")

                # Store attempt
                attempt = {
                    "run_number": run_idx,
                    "predicted_answer": predicted_answer,
                    "correct": is_correct,
                    "generated_text": cleaned_text,
                    "generation_time_seconds": gen_time,
                    "tokens_per_second": tokens_per_sec,
                    "output_tokens": output_len,
                }
                attempts.append(attempt)

                pbar.update(1)

            except Exception as e:
                print(f"\n  ✗ Error on run {run_idx}: {e}")
                import traceback
                if verbose:
                    traceback.print_exc()

                attempts.append({
                    "run_number": run_idx,
                    "predicted_answer": None,
                    "correct": False,
                    "error": str(e),
                })
                pbar.update(1)

        # Store problem result
        problem_result = {
            "problem_id": problem_id,
            "problem_text": problem_text,
            "ground_truth": ground_truth,
            "attempts": attempts,
            "passes_at_1": any(a.get('correct', False) for a in attempts[:1]),
            "passes_at_8": any(a.get('correct', False) for a in attempts[:8]),
            "passes_at_32": any(a.get('correct', False) for a in attempts[:32]),
        }
        problem_results.append(problem_result)

    pbar.close()

    # Compute final metrics
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    metrics = compute_aime_metrics(problem_results)
    print(f"Pass@1:  {metrics['pass@1_percentage']} ({int(metrics['pass@1'] * len(problem_results))}/{len(problem_results)})")
    if num_runs >= 8:
        print(f"Pass@8:  {metrics['pass@8_percentage']} ({int(metrics['pass@8'] * len(problem_results))}/{len(problem_results)})")
    if num_runs >= 32:
        print(f"Pass@32: {metrics['pass@32_percentage']} ({int(metrics['pass@32'] * len(problem_results))}/{len(problem_results)})")
    print()
    print("Performance:")
    avg_gen_time = total_gen_time / (len(problem_results) * num_runs)
    print(f"  Average generation time: {avg_gen_time:.2f}s per run")
    print(f"  Total generation time: {total_gen_time:.1f}s ({total_gen_time/60:.1f} minutes)")
    print("=" * 80)

    # Get model folder name for filename
    model_folder = os.path.basename(model_path)

    # Prepare output
    output = {
        "config": {
            "run_datetime": run_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": timestamp_str,
            "model_name": model_name,
            "model_path": model_path,
            "model_folder": model_folder,
            "dataset_name": dataset_name,
            "num_problems": len(problem_results),
            "num_runs": num_runs,
            "block_length": block_length,
            "denoising_steps": denoising_steps,
            "temperature": temperature,
            "remasking_strategy": remasking_strategy,
            "confidence_threshold": confidence_threshold,
            "max_gen_length": max_gen_length,
        },
        "metrics": metrics,
        "problem_results": problem_results,
    }

    # Save results if requested
    if save_results:
        # Create model-specific results directory
        model_results_dir = f"{RESULTS_DIR}/{model_folder}"
        os.makedirs(model_results_dir, exist_ok=True)

        # Create filename with model name and timestamp
        safe_model_name = model_folder.replace("/", "_").replace("\\", "_")
        filename = f"aime24_{safe_model_name}_runs{num_runs}_{timestamp_str}.json"
        output_file = f"{model_results_dir}/{filename}"

        # Save results to file
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {output_file}")

        # Commit the volume to persist changes
        results_volume.commit()
        print(f"Results committed to volume: sdar-results")

    return output


@app.local_entrypoint()
def main(
    model_name: str = "JetLM/SDAR-4B-Chat",
    dataset_name: str = "HuggingFaceH4/aime_2024",
    num_samples: int | None = None,
    num_runs: int = 1,
    block_length: int = 4,
    denoising_steps: int = 4,
    temperature: float = 1.0,
    remasking_strategy: str = "low_confidence_dynamic",
    confidence_threshold: float = 0.9,
    max_gen_length: int = 2048,
    verbose: bool = True,
):
    """
    Run AIME24 benchmark locally via Modal.
    """
    result = run_aime24_benchmark.remote(
        model_name=model_name,
        dataset_name=dataset_name,
        num_samples=num_samples,
        num_runs=num_runs,
        block_length=block_length,
        denoising_steps=denoising_steps,
        temperature=temperature,
        remasking_strategy=remasking_strategy,
        confidence_threshold=confidence_threshold,
        max_gen_length=max_gen_length,
        save_results=True,
        verbose=verbose,
    )

    print("\n✅ Benchmark complete!")
    print(f"📊 Pass@1: {result['metrics']['pass@1_percentage']}")
    if num_runs >= 8:
        print(f"📊 Pass@8: {result['metrics']['pass@8_percentage']}")
    if num_runs >= 32:
        print(f"📊 Pass@32: {result['metrics']['pass@32_percentage']}")
