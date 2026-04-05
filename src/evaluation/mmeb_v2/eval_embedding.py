import os
import sys
import time
import yaml
import torch
import random
import datetime
import pickle
import json
import logging
import numpy as np
import torch.distributed as dist

from datetime import timedelta
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from datasets import concatenate_datasets
from datasets.distributed import split_dataset_by_node
from .arguments import ModelArguments, DataArguments, EvalArguments
from .utils.basic_utils import print_rank
from .utils.eval_utils.metrics import RankingMetrics
from .models import MMEBEmbeddingModel
from .data.datasets.base_eval_dataset import AutoEvalPairDataset, generate_cand_dataset
from .data.collator import MultimodalEvalDataCollator

logger = logging.getLogger(__name__)


def setup_logging(output_path: str, rank: int = 0):
    """Configure logging to both console and file."""
    log_file = os.path.join(output_path, "eval.log")

    # Clear existing handlers on root logger to avoid duplicates
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — always show INFO+
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler — only on rank 0
    if rank == 0:
        os.makedirs(output_path, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

def pad_dataset_to_divisible(dataset, world_size):
    num_samples = len(dataset)
    if num_samples % world_size == 0:
        return dataset, num_samples

    num_to_add = world_size - (num_samples % world_size)
    padded_size = num_samples + num_to_add

    padding_data = dataset.select([i % len(dataset) for i in range(num_to_add)])
    padded_dataset = concatenate_datasets([dataset, padding_data])
    return padded_dataset, padded_size


@torch.no_grad()
def encode_embeddings(
    model: MMEBEmbeddingModel,
    loader: DataLoader,
    encode_side: str,  # 'qry' or 'cand'
    full_dataset_len: int,
    description: str = "Encoding"
) -> tuple[np.ndarray, list]:
    """
    Generate embeddings using MMEBEmbeddingModel's encode_input method.
    Supports DDP distributed gathering.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    local_embeds = []
    local_gt_infos = []

    model.eval()
    
    # Show tqdm progress bar only on main process
    progress_bar = tqdm(
        loader, 
        desc=f"{description} (rank {rank})", 
        disable=local_rank > 0,
        ncols=120
    )

    for batch_inputs, dataset_info in progress_bar:
        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            reps = model.encode_input(batch_inputs)  # Returns [Batch, Dim]
            reps = reps.detach()

        local_embeds.append(reps)
        
        # Process metadata
        if encode_side == "qry":
            local_gt_infos.extend(dataset_info)
        else:
            local_gt_infos.extend([info.get("cand_name", "") for info in dataset_info])

    if not local_embeds:
        return np.array([]), []
    
    local_embeds_tensor = torch.cat(local_embeds, dim=0).contiguous()

    # DDP synchronization logic
    if dist.is_initialized():
        # Gather tensors
        gathered_embeds = [torch.zeros_like(local_embeds_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_embeds, local_embeds_tensor)
        
        # Gather metadata
        gathered_infos = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_infos, local_gt_infos)
        
        if rank == 0:
            final_embeddings = torch.cat(gathered_embeds, dim=0).cpu().float().numpy()
            final_infos = [info for rank_list in gathered_infos for info in rank_list]
            
            # Truncate potential DDP padding to match original dataset size
            return final_embeddings[:full_dataset_len], final_infos[:full_dataset_len]
        else:
            return None, None
    else:
        return local_embeds_tensor.cpu().float().numpy(), local_gt_infos

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if "RANK" in os.environ and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Parse arguments early so we can set up logging with the output path
    parser = HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    eval_args: EvalArguments
    os.makedirs(data_args.encode_output_path, exist_ok=True)

    # Set up logging to console + file
    setup_logging(data_args.encode_output_path, rank=rank)

    logger.info("=" * 60)
    logger.info("Qwen3-VL-Embedding Evaluation")
    logger.info("=" * 60)
    logger.info(f"Model:        {model_args.model_name_or_path}")
    logger.info(f"Config:       {data_args.dataset_config}")
    logger.info(f"Output:       {data_args.encode_output_path}")
    logger.info(f"Data basedir: {data_args.data_basedir}")
    logger.info(f"Batch size:   {eval_args.per_device_eval_batch_size}")
    logger.info(f"Normalize:    {model_args.normalize}")
    logger.info(f"Device:       {eval_args.device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(local_rank)
        gpu_mem = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
        logger.info(f"GPU:          {gpu_name} ({gpu_mem:.1f} GB)")
    logger.info(f"World size:   {world_size}")
    logger.info(f"Rank:         {rank}, Local rank: {local_rank}")
    logger.info("-" * 60)

    # DDP-safe model loading
    t0 = time.time()
    if rank == 0:
        logger.info(f"Loading model: {model_args.model_name_or_path} ...")
        model = MMEBEmbeddingModel.load(
            model_name_or_path=model_args.model_name_or_path,
            normalize=model_args.normalize,
            instruction=model_args.instruction,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
        )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if rank != 0:
        logger.info(f"[rank={rank}] Loading model from cache...")
        time.sleep(random.randint(2 * rank, 3 * rank))
        model = MMEBEmbeddingModel.load(
            model_name_or_path=model_args.model_name_or_path,
            normalize=model_args.normalize,
            instruction=model_args.instruction,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
        )

    model.eval()
    model = model.to(eval_args.device, dtype=torch.bfloat16)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_configs = yaml.safe_load(yaml_file)
    logger.info(f"Datasets to evaluate: {list(dataset_configs.keys())}")

    # Main evaluation loop
    for dataset_idx, (dataset_name, task_config) in enumerate(dataset_configs.items()):
        dataset_t0 = time.time()
        if dist.is_initialized():
            dist.barrier()

        logger.info("")
        logger.info(f"{'=' * 60}")
        logger.info(f"[{dataset_idx + 1}/{len(dataset_configs)}] Evaluating: {dataset_name}")
        logger.info(f"{'=' * 60}")
        logger.info(f"  parser:     {task_config.get('dataset_parser')}")
        logger.info(f"  eval_type:  {task_config.get('eval_type', 'global')}")
        logger.info(f"  num_frames: {task_config.get('num_frames')}")
        logger.info(f"  video_root: {task_config.get('video_root')}")
        logger.info(f"  frame_root: {task_config.get('frame_root')}")

        query_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_qry")
        cand_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_tgt")
        dataset_info_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_info.jsonl")

        do_query = not os.path.exists(query_embed_path) or not os.path.exists(dataset_info_path)
        do_cand = not os.path.exists(cand_embed_path)

        if do_query or do_cand:
            logger.info(f"  Cached embeddings: queries={'found' if not do_query else 'MISSING'}, "
                        f"candidates={'found' if not do_cand else 'MISSING'}")

            if data_args.data_basedir is not None:
                for key in ["image_root", "video_root", "frame_root", "clip_root", "data_path"]:
                    if task_config.get(key):
                        task_config[key] = os.path.join(data_args.data_basedir, task_config[key])

            try:
                logger.info(f"  Loading dataset from HuggingFace...")
                full_eval_qry_dataset, corpus = AutoEvalPairDataset.instantiate(
                    model_args=model_args, data_args=data_args, **task_config
                )
                full_eval_cand_dataset = generate_cand_dataset(full_eval_qry_dataset, corpus)
                eval_qry_dataset, eval_cand_dataset = full_eval_qry_dataset, full_eval_cand_dataset
                logger.info(f"  Dataset loaded: {len(full_eval_qry_dataset)} queries, "
                            f"{len(full_eval_cand_dataset)} candidates, "
                            f"{len(corpus) if corpus else 0} corpus classes")

                # Pad datasets to be divisible by world_size before splitting
                if dist.is_initialized():
                    padded_qry_dataset, _ = pad_dataset_to_divisible(full_eval_qry_dataset, world_size)
                    padded_cand_dataset, _ = pad_dataset_to_divisible(full_eval_cand_dataset, world_size)
                    eval_qry_dataset = split_dataset_by_node(padded_qry_dataset, rank=rank, world_size=world_size)
                    eval_cand_dataset = split_dataset_by_node(padded_cand_dataset, rank=rank, world_size=world_size)
                else:
                    padded_qry_dataset, padded_cand_dataset = full_eval_qry_dataset, full_eval_cand_dataset
            except Exception as e:
                logger.error(f"  Failed to load dataset {dataset_name}: {e}", exc_info=True)
                raise e
        else:
            logger.info(f"  Cached embeddings: queries=found, candidates=found (skipping encoding)")

        # 1. Compute query embeddings
        if do_query:
            logger.info(f"  Encoding {len(full_eval_qry_dataset)} queries (batch_size={eval_args.per_device_eval_batch_size})...")
            t_enc = time.time()
            eval_qry_collator = MultimodalEvalDataCollator(encode_side="qry")
            eval_qry_loader = DataLoader(
                eval_qry_dataset,
                batch_size=eval_args.per_device_eval_batch_size,
                collate_fn=eval_qry_collator,
                num_workers=eval_args.dataloader_num_workers,
                pin_memory=True,
                shuffle=False
            )
            query_embeds, gt_infos = encode_embeddings(
                model=model,
                loader=eval_qry_loader,
                encode_side="qry",
                full_dataset_len=len(full_eval_qry_dataset),
                description=f"Queries: {dataset_name}"
            )
            if rank == 0:
                os.makedirs(os.path.dirname(query_embed_path), exist_ok=True)
                with open(query_embed_path, 'wb') as f:
                    pickle.dump(query_embeds, f)
                with open(dataset_info_path, 'w', encoding='utf-8') as f:
                    for info in gt_infos:
                        f.write(json.dumps(info, ensure_ascii=False) + '\n')
                logger.info(f"  Query encoding done: {len(query_embeds)} embeddings in {time.time() - t_enc:.1f}s "
                            f"-> {query_embed_path}")

            if dist.is_initialized():
                dist.barrier()

        # 2. Compute candidate embeddings
        if do_cand:
            logger.info(f"  Encoding {len(full_eval_cand_dataset)} candidates...")
            t_enc = time.time()
            eval_cand_collator = MultimodalEvalDataCollator(encode_side="cand")
            eval_cand_loader = DataLoader(
                eval_cand_dataset,
                batch_size=eval_args.per_device_eval_batch_size,
                collate_fn=eval_cand_collator,
                num_workers=eval_args.dataloader_num_workers,
                pin_memory=True,
                shuffle=False
            )
            cand_embeds, all_cand_ids = encode_embeddings(
                model=model,
                loader=eval_cand_loader,
                encode_side="cand",
                full_dataset_len=len(full_eval_cand_dataset),
                description=f"Candidates: {dataset_name}"
            )
            if rank == 0:
                os.makedirs(os.path.dirname(cand_embed_path), exist_ok=True)
                cand_embed_dict = {
                    cand_id: embed for cand_id, embed in zip(all_cand_ids, cand_embeds)
                }
                with open(cand_embed_path, 'wb') as f:
                    pickle.dump(cand_embed_dict, f)
                logger.info(f"  Candidate encoding done: {len(cand_embed_dict)} unique embeddings in {time.time() - t_enc:.1f}s "
                            f"-> {cand_embed_path}")

            if dist.is_initialized():
                dist.barrier()
        
        # 3. Compute scores (rank 0 only)
        if rank == 0:
            score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_score.json")
            pred_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_pred.jsonl")

            # Skip computation only if both files exist and are valid
            need_compute = True
            if os.path.exists(score_path) and os.path.exists(pred_path):
                try:
                    with open(score_path, "r") as f:
                        score_dict = json.load(f)
                    if "num_pred" in score_dict:
                        logger.info(f"  Scores cached, skipping computation.")
                        need_compute = False
                except Exception as e:
                    logger.warning(f"  Score cache corrupted ({e}), re-computing...")

            if need_compute:
                logger.info(f"  Computing similarity scores...")
                t_score = time.time()

                # Load persisted embeddings and metadata
                with open(query_embed_path, 'rb') as f:
                    qry_embeds = pickle.load(f)  # np.ndarray [Nq, D]
                with open(cand_embed_path, 'rb') as f:
                    cand_embed_dict = pickle.load(f)  # Dict {id: [D]}
                gt_infos = [json.loads(l) for l in open(dataset_info_path, encoding='utf-8')]

                logger.info(f"  Loaded: {len(qry_embeds)} query embeds (dim={qry_embeds.shape[1]}), "
                            f"{len(cand_embed_dict)} candidate embeds")

                device = model.device
                pred_dicts = []
                qry_tensor = torch.from_numpy(qry_embeds).to(device)

                rank_against_all_candidates = task_config.get("eval_type", "global") == "global"
                if rank_against_all_candidates:
                    logger.info(f"  Ranking mode: GLOBAL (each query ranked against all {len(cand_embed_dict)} candidates)")
                    cand_keys = list(cand_embed_dict.keys())
                    cand_embeds = np.stack([cand_embed_dict[key] for key in cand_keys])
                    cand_tensor = torch.from_numpy(cand_embeds).to(device)

                    with torch.no_grad():
                        scores = model.compute_similarity(qry_tensor, cand_tensor)
                        _, ranked_indices = torch.sort(scores, dim=1, descending=True)
                        ranked_indices = ranked_indices.cpu().numpy().astype(int)

                    del cand_tensor
                    torch.cuda.empty_cache()

                    for qid, (ranked_idx, gt_info) in tqdm(
                        enumerate(zip(ranked_indices, gt_infos)),
                        total=len(gt_infos), desc=f"Global Ranking: {dataset_name}",
                        disable=local_rank > 0, ncols=120,
                    ):
                        rel_docids = gt_info["label_name"] if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]]
                        rel_scores = gt_info.get("rel_scores", None)
                        pred_dicts.append({
                            "prediction": [cand_keys[i] for i in ranked_idx],
                            "label": rel_docids,
                            "rel_scores": rel_scores,
                        })
                else:
                    logger.info(f"  Ranking mode: LOCAL (per-query candidate set)")
                    for qid, (qry_vec, gt_info) in tqdm(
                        enumerate(zip(qry_tensor, gt_infos)),
                        total=len(gt_infos), desc=f"Local Ranking: {dataset_name}",
                        disable=local_rank > 0, ncols=120
                    ):
                        cand_names = gt_info["cand_names"]
                        cand_embeds = np.stack([cand_embed_dict[name] for name in cand_names])
                        cand_tensor = torch.from_numpy(cand_embeds).to(device)

                        with torch.no_grad():
                            sim_scores = model.compute_similarity(qry_vec.unsqueeze(0), cand_tensor).squeeze(0)
                            _, ranked_idx = torch.sort(sim_scores, descending=True)
                            ranked_idx = ranked_idx.cpu().numpy().astype(int)

                        rel_docids = gt_info["label_name"] if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]]
                        rel_scores = gt_info.get("rel_scores", None)
                        pred_dicts.append({
                            "prediction": [cand_names[i] for i in ranked_idx],
                            "label": rel_docids,
                            "rel_scores": rel_scores,
                        })

                    torch.cuda.empty_cache()

                # Compute metrics
                metrics_to_report = task_config.get("metrics", ["hit", "ndcg", "precision", "recall", "f1", "map", "mrr"])
                metrics = RankingMetrics(metrics_to_report)
                score_dict = metrics.evaluate(pred_dicts)

                score_dict["num_pred"] = len(pred_dicts)
                score_dict["num_data"] = len(gt_infos)

                with open(score_path, "w") as f:
                    json.dump(score_dict, f, indent=4)
                with open(pred_path, "w", encoding='utf-8') as f:
                    for pred in pred_dicts:
                        f.write(json.dumps(pred, ensure_ascii=False) + '\n')

                logger.info(f"  Scoring done in {time.time() - t_score:.1f}s")

            # Log results (both cached and freshly computed)
            with open(score_path, "r") as f:
                score_dict = json.load(f)

            logger.info(f"")
            logger.info(f"  Results for {dataset_name}:")
            logger.info(f"  {'─' * 40}")
            key_metrics = ['hit@1', 'hit@5', 'hit@10', 'mrr@10', 'ndcg_linear@10', 'map@10']
            for k in key_metrics:
                if k in score_dict:
                    logger.info(f"    {k:<25s} {score_dict[k]:.4f}")
            logger.info(f"  {'─' * 40}")
            logger.info(f"    {'num_queries':<25s} {score_dict.get('num_data', 'N/A')}")
            logger.info(f"    {'num_predictions':<25s} {score_dict.get('num_pred', 'N/A')}")
            logger.info(f"  Score file: {score_path}")
            logger.info(f"  Pred file:  {pred_path}")
            logger.info(f"  Dataset total time: {time.time() - dataset_t0:.1f}s")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluation complete.")
    logger.info("=" * 60)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()