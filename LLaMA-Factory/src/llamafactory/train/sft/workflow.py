# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Optional
import os

import torch
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import TrainerCallback

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.packages import is_transformers_version_greater_than
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer
from ...extras import global_vars


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


# [New] Hook Function Definition
def deepseek_router_hook(layer_idx, top_k=6):
    def hook(module, inputs, outputs):
        # outputs is the router logits: [batch*seq, num_experts]
        tags = global_vars.CURRENT_BATCH_TAGS
        mask = global_vars.CURRENT_ATTENTION_MASK
        
        if tags is None:
            return

        if isinstance(outputs, tuple):
            topk_idx = outputs[0].detach()
        else:
            return
        
        batch_size = tags.size(0)
        
        # Handle dimensions, ensure [batch, seq, top_k]
        if topk_idx.dim() == 2:
            total_tokens = topk_idx.size(0)
            if total_tokens % batch_size != 0:
                return # Mismatch
            seq_len = total_tokens // batch_size
            topk_idx = topk_idx.view(batch_size, seq_len, -1)
        
        top_k_actual = topk_idx.size(-1)
        seq_len = topk_idx.size(1)
        
        # Expand tags: [batch] -> [batch, seq, topk]
        tags_expanded = tags.view(batch_size, 1, 1).expand(batch_size, seq_len, top_k_actual)
        
        # Expand mask if available: [batch, seq] -> [batch, seq, topk]
        mask_expanded = None
        if mask is not None:
            if mask.size(0) == batch_size and mask.size(1) == seq_len:
                 mask_expanded = mask.view(batch_size, seq_len, 1).expand(batch_size, seq_len, top_k_actual)
            else:
                 # Dimension mismatch (maybe due to packing or other reasons), ignore mask or warn
                 pass

        # Statistics logic
        if layer_idx not in global_vars.ROUTER_STATS:
            global_vars.ROUTER_STATS[layer_idx] = {}
            
        # Convert to CPU list for statistics (avoid GPU sync blocking)
        flat_tags = tags_expanded.reshape(-1).cpu().tolist()
        flat_experts = topk_idx.reshape(-1).cpu().tolist()
        
        flat_mask = None
        if mask_expanded is not None:
            flat_mask = mask_expanded.reshape(-1).cpu().tolist()
        
        if flat_mask is not None:
            for t, e, m in zip(flat_tags, flat_experts, flat_mask):
                if m == 0: continue # Skip padding
                if t == -1: continue # Skip untagged data
                
                if t not in global_vars.ROUTER_STATS[layer_idx]:
                    global_vars.ROUTER_STATS[layer_idx][t] = {}
                if e not in global_vars.ROUTER_STATS[layer_idx][t]:
                    global_vars.ROUTER_STATS[layer_idx][t][e] = 0
                global_vars.ROUTER_STATS[layer_idx][t][e] += 1
        else:
            # Fallback if no mask
            for t, e in zip(flat_tags, flat_experts):
                if t == -1: continue # Skip untagged data
                if t not in global_vars.ROUTER_STATS[layer_idx]:
                    global_vars.ROUTER_STATS[layer_idx][t] = {}
                if e not in global_vars.ROUTER_STATS[layer_idx][t]:
                    global_vars.ROUTER_STATS[layer_idx][t][e] = 0
                global_vars.ROUTER_STATS[layer_idx][t][e] += 1
            
    return hook


# [New] Callback Definition
class MoEVisCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Only run on main process to avoid write conflicts
        if not state.is_world_process_zero:
            return

        # Plot every 50 steps
        if state.global_step % 5 == 0 and state.global_step > 0:
            # Create subdirectories for data and visualization
            moe_data_dir = os.path.join(args.output_dir, "moe_data")
            moe_vis_dir = os.path.join(args.output_dir, "moe_vis")
            os.makedirs(moe_data_dir, exist_ok=True)
            os.makedirs(moe_vis_dir, exist_ok=True)

            # --- 1. Save CSV for ALL layers (Independent file per step) ---
            all_data = []
            for layer_idx, layer_stats in global_vars.ROUTER_STATS.items():
                for tag, experts in layer_stats.items():
                    for expert_id, count in experts.items():
                        all_data.append({
                            "Step": state.global_step,
                            "Layer": layer_idx,
                            "Tag": tag,
                            "Expert": expert_id,
                            "Count": count
                        })
            
            if all_data:
                df_all = pd.DataFrame(all_data)
                csv_path = os.path.join(moe_data_dir, f"router_stats_step_{state.global_step}.csv")
                df_all.to_csv(csv_path, index=False)

            # --- 2. Visualization for Specific Layers ---
            # Define the layers you want to visualize
            target_layers = [5, 10, 15, 20, 25]
            
            for target_layer in target_layers:
                stats = global_vars.ROUTER_STATS.get(target_layer, {})
                
                if stats:
                    data = []
                    for tag, experts in stats.items():
                        for expert_id, count in experts.items():
                            data.append({"Tag": tag, "Expert": expert_id, "Count": count})
                    
                    if data:
                        df = pd.DataFrame(data)
                        # Normalized heatmap (Row: Expert, Col: Tag)
                        df_pivot = df.pivot_table(index="Expert", columns="Tag", values="Count", fill_value=0)
                        
                        # 归一化: 每个 Tag 下的专家激活比例 (列和为1)
                        # 防止除以0
                        column_sums = df_pivot.sum(axis=0)
                        column_sums[column_sums == 0] = 1.0 
                        df_norm = df_pivot / column_sums
                        
                        plt.figure(figsize=(12, 6))
                        sns.heatmap(df_norm, cmap="viridis")
                        plt.title(f"Layer {target_layer} Routing (Step {state.global_step})")
                        
                        # Save to local file
                        save_path = os.path.join(moe_vis_dir, f"layer_{target_layer}_step_{state.global_step}.png")
                        plt.savefig(save_path)
                        
                        # Send to Wandb (使用独立的 Key，方便在 Wandb 面板中区分)
                        if wandb.run is not None:
                            wandb.log({f"layer_{target_layer}_routing": wandb.Image(plt)})
                        
                        plt.close() # Close plot to free memory
            
            # Clear statistics
            global_vars.ROUTER_STATS.clear()


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # [New] Register Hook
    layer_cnt = 0
    top_k = getattr(model.config, "num_experts_per_tok", 6)
    for name, module in model.named_modules():
        # Targeting DeepSeek V2 Gate module
        if "gate" in name and "mlp" in name and "gate_proj" not in name: 
             module.register_forward_hook(deepseek_router_hook(layer_cnt, top_k=top_k))
             layer_cnt += 1

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Metric utils
    metric_module = {}
    if model_args.use_kt:
        if training_args.predict_with_generate:
            raise NotImplementedError("`predict_with_generate` is not supported in KTransformers SFT yet.")
        elif finetuning_args.compute_accuracy:
            raise NotImplementedError("`compute_accuracy` is not supported in KTransformers SFT yet.")
    
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)

    # Compatible with Transformers v4 and Transformers v5
    if is_transformers_version_greater_than("4.58.0"):
        extra_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
        if not isinstance(extra_ids, list):
            extra_special_tokens = getattr(tokenizer, "_extra_special_tokens", [])
            string_tokens = [str(t) for t in extra_special_tokens]
            extra_ids = tokenizer.convert_tokens_to_ids(string_tokens)
        all_eos_ids = [tokenizer.eos_token_id] + [i for i in extra_ids if i != -1]
        unique_eos_ids = list(dict.fromkeys(all_eos_ids))
        gen_kwargs["eos_token_id"] = unique_eos_ids
    else:
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # Initialize our Trainer
    if callbacks is None:
        callbacks = []
    callbacks.append(MoEVisCallback())

    if model_args.use_kt:
        from ktransformers.util.globals import GLOBAL_CONFIG
        from ktransformers.sft.lora import KTrainer

        GLOBAL_CONFIG._config["mod"] = "sft"

        trainer = KTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer_module,
            data_collator=data_collator,
            callbacks=callbacks,
            **dataset_module,
            **metric_module,
        )
        trainer.model_accepts_loss_kwargs = False
        model.config.use_cache = False

    else:
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            callbacks=callbacks,
            gen_kwargs=gen_kwargs,
            **dataset_module,
            **tokenizer_module,
            **metric_module,
        )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]

            plot_loss(training_args.output_dir, keys=keys)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
