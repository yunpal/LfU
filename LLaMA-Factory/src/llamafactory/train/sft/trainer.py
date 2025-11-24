# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from torch.utils.data import DataLoader, RandomSampler    
import torch.nn.functional as F

from transformers.trainer_utils import (
    seed_worker
)

from transformers.utils import (
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_apollo_torch_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_in_notebook,
    is_ipex_available,
    is_liger_kernel_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_schedulefree_available,
    is_torch_compile_available,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    logging,
    strtobool,
)

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments

import wandb

logger = logging.get_logger(__name__)



class Lfu_consistency_asc_rep(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        train_dataset=None,
        data_collator=None,
        train_dataset_b=None,  
        data_collator_b=None,  
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__( train_dataset=train_dataset,data_collator=data_collator,**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)
        self.train_dataset_b = train_dataset_b  
        self.data_collator_b = data_collator_b  
        self.harmful_dataloader = self.get_harmful_dataloader(self.train_dataset_b)
        self.harmful_data_iter = iter(self.harmful_dataloader)


    def get_harmful_dataloader(self, harmful_dataset):
        if self.finetuning_args.disable_shuffling:
            sampler = SequentialSampler(harmful_dataset)
        else:
            sampler = RandomSampler(harmful_dataset)

        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": self.data_collator_b,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "sampler": sampler,
            "drop_last": self.args.dataloader_drop_last,
            "worker_init_fn": seed_worker,
        }

        return self.accelerator.prepare(DataLoader(harmful_dataset, **dataloader_params))



    def sample_from_harmful(self):
        try:
            batch = next(self.harmful_data_iter)
        except StopIteration:
            self.harmful_data_iter = iter(self.harmful_dataloader)
            batch = next(self.harmful_data_iter)
        return batch

 

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate: 
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")


    def _grad_norm(self,poison_grads_representation):
        norm = torch.norm(
                torch.stack([

                    ( poison_grads_representation[name] ).norm(p=2)

                    for name in poison_grads_representation
                ]),
                p=2
               )

        return norm


    def compute_kl_consistency_loss(
        self,
        perturbed_logits,               
        target_logits,                
        labels,                          
        label_pad_token_id: int = -100   
    ):

    
        mask = (labels != label_pad_token_id).float() 


        log_probs = F.log_softmax(perturbed_logits, dim=-1)  
        probs = F.softmax(target_logits, dim=-1)              


        kl = F.kl_div(log_probs, probs, reduction='none').sum(-1)  

        loss = (kl * mask).sum() / (mask.sum() + 1e-8)

        return loss



    import torch.nn.functional as F


    def compute_mse_consistency_loss(self, perturbed_logits, target_logits, labels, label_pad_token_id=-100):

        assert isinstance(perturbed_logits, tuple)
        assert isinstance(target_logits, tuple)
        assert len(perturbed_logits) == len(target_logits)

        total_loss = 0.0
        count = 0

        valid_token_mask = (labels != label_pad_token_id)

        start_layer = self.finetuning_args.start_layer 
        finish_layer = self.finetuning_args.finish_layer 

        for i, (p, t) in enumerate(zip(perturbed_logits, target_logits)):
            if not (start_layer <= i <= finish_layer):
                continue 

            t = t.detach()
            assert p.shape == t.shape, f"Shape mismatch: {p.shape} vs {t.shape}"

            p_flat = p[valid_token_mask]
            t_flat = t[valid_token_mask]

            if p_flat.numel() == 0:
                continue

            layer_loss = F.mse_loss(p_flat, t_flat, reduction='mean')
            total_loss += layer_loss
            count += 1

        return total_loss / count if count > 0 else torch.tensor(0.0, device=labels.device)


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
             labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        outputs = model(**inputs, use_cache=False, output_hidden_states=True)

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        logits = outputs["hidden_states"]
        return loss, logits



    def training_step(self, model, inputs, num_items_in_batch=None):


        inputs = self._prepare_inputs(inputs)
        harmful_inputs = self.sample_from_harmful()
        harmful_inputs = self._prepare_inputs(harmful_inputs)


        saved_grads = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                saved_grads[name] = param.grad.clone()


        loss,logits = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        reversed_loss, _ = self.compute_loss(model, harmful_inputs, num_items_in_batch=num_items_in_batch)



        model.zero_grad()
        self.accelerator.backward(reversed_loss)

        stored_grads = {}
        param_names = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                stored_grads[name] = param.grad.detach().clone()
                param_names.append(name)

        original_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                original_params[name] = param.data.clone()

        with torch.no_grad():
            grad_norm = self._grad_norm(stored_grads) 


        for name, param in model.named_parameters():
            if param.requires_grad and name in stored_grads:
                param.data += self.finetuning_args.alpha * stored_grads[name] / grad_norm

        _ , perturbed_logits = self.compute_loss(model, inputs,num_items_in_batch=num_items_in_batch)

        model.zero_grad()

        div_loss = self.compute_mse_consistency_loss(
            perturbed_logits=perturbed_logits,
            target_logits=logits,
            labels=inputs["labels"],
            label_pad_token_id=-100
        )/self.args.gradient_accumulation_steps



        for name, param in model.named_parameters():
            if param.requires_grad and name in original_params:
                param.data = original_params[name]


        for name, param in model.named_parameters():
            if param.requires_grad and name in saved_grads:
                param.grad = saved_grads[name]

        final_loss=loss+self.finetuning_args.beta*div_loss
        wandb.log({"train/div_loss": (self.finetuning_args.beta *div_loss).item()}, step=self.state.global_step)
        wandb.log({"train/final_loss": final_loss.item()}, step=self.state.global_step)

        self.accelerator.backward(final_loss)

        return final_loss.detach()





class Lfu_consistency_asc_rep_adv_train_rep_add(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        train_dataset=None,
        data_collator=None,
        train_dataset_b=None,  
        data_collator_b=None,  
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__( train_dataset=train_dataset,data_collator=data_collator,**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)
        self.train_dataset_b = train_dataset_b  
        self.data_collator_b = data_collator_b  
        self.harmful_dataloader = self.get_harmful_dataloader(self.train_dataset_b)
        self.harmful_data_iter = iter(self.harmful_dataloader)


    def get_harmful_dataloader(self, harmful_dataset):
        if self.finetuning_args.disable_shuffling:
            sampler = SequentialSampler(harmful_dataset)
        else:
            sampler = RandomSampler(harmful_dataset)

        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": self.data_collator_b,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "sampler": sampler,
            "drop_last": self.args.dataloader_drop_last,
            "worker_init_fn": seed_worker,
        }

        return self.accelerator.prepare(DataLoader(harmful_dataset, **dataloader_params))



    def sample_from_harmful(self):
        try:
            batch = next(self.harmful_data_iter)
        except StopIteration:
            self.harmful_data_iter = iter(self.harmful_dataloader)
            batch = next(self.harmful_data_iter)
        return batch

 

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")


    def _grad_norm(self,poison_grads_representation):
        norm = torch.norm(
                torch.stack([
                    ( poison_grads_representation[name] ).norm(p=2)
                    for name in poison_grads_representation
                ]),
                p=2
               )

        return norm


    def compute_kl_consistency_loss(
        self,
        perturbed_logits,               
        target_logits,                
        labels,                          
        label_pad_token_id: int = -100   
    ):


        mask = (labels != label_pad_token_id).float()  


        log_probs = F.log_softmax(perturbed_logits, dim=-1) 
        probs = F.softmax(target_logits, dim=-1)             


        kl = F.kl_div(log_probs, probs, reduction='none').sum(-1)  

        loss = (kl * mask).sum() / (mask.sum() + 1e-8)

        return loss



    import torch.nn.functional as F

    import torch.nn.functional as F

    def compute_mse_consistency_loss_last_only(self, target_logits, perturbed_logits, labels, label_pad_token_id=-100):

        batch_size, target_seq_len, hidden_dim = target_logits[0].shape


        if isinstance(perturbed_logits, torch.Tensor):
            if perturbed_logits.shape[1] == 1 or perturbed_logits.shape[1] != target_seq_len:
                perturbed_logits = perturbed_logits[:, -1:, :].expand(batch_size, target_seq_len, hidden_dim)
            perturbed_logits = tuple(perturbed_logits.clone() for _ in range(len(target_logits)))


        elif isinstance(perturbed_logits, (list, tuple)):
            new_perturbed = []
            for p in perturbed_logits:
                if p.shape[1] == 1 or p.shape[1] != target_seq_len:
                    p_expanded = p[:, -1:, :].expand(batch_size, target_seq_len, hidden_dim)
                else:
                    p_expanded = p
                new_perturbed.append(p_expanded)
            perturbed_logits = tuple(new_perturbed)

        else:
            raise TypeError(f"Expected Tensor, list, or tuple for perturbed_logits, but got {type(perturbed_logits)}")

        assert isinstance(target_logits, tuple)
        assert isinstance(perturbed_logits, tuple)

        assert len(target_logits) == len(perturbed_logits)

        total_loss = 0.0
        count = 0

        start_layer = self.finetuning_args.start_layer 
        finish_layer = self.finetuning_args.finish_layer 

        for i, (t, p) in enumerate(zip(target_logits, perturbed_logits)):
            if not (start_layer <= i <= finish_layer):
                continue


            t_last = t[:, -1:, :]  
            p_last = p[:, -1:, :].detach()  

            assert t_last.shape == p_last.shape, f"Shape mismatch: {t_last.shape} vs {p_last.shape}"

            layer_loss = F.mse_loss(t_last, p_last, reduction='mean')
            total_loss += layer_loss
            count += 1

        return total_loss / count if count > 0 else torch.tensor(0.0, device=target_logits[0].device)

    def compute_mse_consistency_loss_perturb_detach(self, target_logits, perturbed_logits, labels, label_pad_token_id=-100):

        batch_size, target_seq_len, hidden_dim = target_logits[0].shape


        if isinstance(perturbed_logits, torch.Tensor):
            if perturbed_logits.shape[1] == 1 or perturbed_logits.shape[1] != target_seq_len:
                perturbed_logits = perturbed_logits[:, -1:, :].expand(batch_size, target_seq_len, hidden_dim)
            perturbed_logits = tuple(perturbed_logits.clone() for _ in range(len(target_logits)))


        elif isinstance(perturbed_logits, (list, tuple)):
            new_perturbed = []
            for p in perturbed_logits:
                if p.shape[1] == 1 or p.shape[1] != target_seq_len:
                    p_expanded = p[:, -1:, :].expand(batch_size, target_seq_len, hidden_dim)
                else:
                    p_expanded = p
                new_perturbed.append(p_expanded)
            perturbed_logits = tuple(new_perturbed)

        else:
            raise TypeError(f"Expected Tensor, list, or tuple for perturbed_logits, but got {type(perturbed_logits)}")


        assert isinstance(target_logits, tuple)
        assert isinstance(perturbed_logits, tuple)
        assert len(target_logits) == len(perturbed_logits)

        total_loss = 0.0
        count = 0
        valid_token_mask = (labels != label_pad_token_id)

        start_layer = self.finetuning_args.start_layer 
        finish_layer = self.finetuning_args.finish_layer 

        for i, (t, p) in enumerate(zip(target_logits, perturbed_logits)):
            if not (start_layer <= i <= finish_layer):
                continue

            assert p.shape == t.shape, f"Shape mismatch: {p.shape} vs {t.shape}"

            t_flat = t[valid_token_mask]
            p_flat = p[valid_token_mask].detach()

            if p_flat.numel() == 0:
                continue

            layer_loss = F.mse_loss(t_flat, p_flat, reduction='mean')
            total_loss += layer_loss
            count += 1

        return total_loss / count if count > 0 else torch.tensor(0.0, device=labels.device)



    def compute_mse_consistency_loss_perturb_learn(self, target_logits, perturbed_logits, labels, label_pad_token_id=-100):


        if isinstance(perturbed_logits, torch.Tensor):

            seq_len = target_logits[0].shape[1]
            batch_size = target_logits[0].shape[0]
            if perturbed_logits.shape[1] == 1:
                perturbed_logits = perturbed_logits.expand(batch_size, seq_len, -1)
            perturbed_logits = tuple(perturbed_logits.clone() for _ in range(len(target_logits)))

        assert isinstance(target_logits, tuple)
        assert isinstance(perturbed_logits, tuple)
        assert len(target_logits) == len(perturbed_logits)

        total_loss = 0.0
        count = 0
        valid_token_mask = (labels != label_pad_token_id)

        start_layer = self.finetuning_args.start_layer 
        finish_layer = self.finetuning_args.finish_layer 

        for i, (t, p) in enumerate(zip(target_logits, perturbed_logits)):
            if not (start_layer <= i <= finish_layer):
                continue

            assert p.shape == t.shape, f"Shape mismatch: {p.shape} vs {t.shape}"


            t_flat = t[valid_token_mask].detach()
            p_flat = p[valid_token_mask]

            if p_flat.numel() == 0:
                continue

            layer_loss = F.mse_loss(p_flat, t_flat, reduction='mean')
            total_loss += layer_loss
            count += 1

        return total_loss / count if count > 0 else torch.tensor(0.0, device=labels.device)



    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
             labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        outputs = model(**inputs, use_cache=False, output_hidden_states=True)

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        logits = outputs["hidden_states"]
        return loss, logits

    def compute_repulsion_loss(self, reversed_logit, target_logits, labels, label_pad_token_id=-100):
        """
        reversed_logit: [B, 1, H]  (fixed)
        target_logits: tuple of [B, T, H]  (T varies per step)
        """
        valid_token_mask = (labels != label_pad_token_id)  # [B, T]
        total_cosine_loss = 0.0
        count = 0

        for layer_logits in target_logits:
            B, T, H = layer_logits.shape
            reversed_expanded = reversed_logit.expand(B, T, H)  # [B, T, H]

  
            tgt_flat = layer_logits[valid_token_mask].detach()
            rev_flat = reversed_expanded[valid_token_mask]

            if rev_flat.numel() == 0:
                continue

            cos_loss = (F.cosine_similarity(rev_flat, tgt_flat, dim=-1) ** 2).mean()

            total_cosine_loss += cos_loss
            count += 1

        return total_cosine_loss / count if count > 0 else torch.tensor(0.0, device=labels.device)




    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step on a batch of inputs.
        """

        inputs = self._prepare_inputs(inputs)
        harmful_inputs = self.sample_from_harmful()
        harmful_inputs = self._prepare_inputs(harmful_inputs)




        loss,logits = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)


        if not hasattr(self, "reversed_logits"):
            device = logits[0].device
            num_layers = len(get_layers(model))
            last_logit = logits[-1]  # [B, T, H]
            init_logit = last_logit.mean(dim=1, keepdim=True).mean(dim=0, keepdim=True)  # [1, 1, H]

            self.reversed_logits = [
                torch.nn.Parameter(torch.randn_like(init_logit), requires_grad=True)
                for _ in range(num_layers)
            ]


        if not hasattr(self, "rev_optimizer"):
            self.rev_optimizer = torch.optim.Adam(self.reversed_logits, lr=self.args.learning_rate * self.finetuning_args.lr_weight )
        

        if not hasattr(self, "rev_inner_step"):
            self.rev_inner_step = 0


        num_layers = len(get_layers(model))
        alpha = [self.finetuning_args.alpha for _ in range(num_layers)]
        add_layers(model, self.reversed_logits, alpha)

        model.eval()
        harmful_loss,harmful_logit = self.compute_loss(model, harmful_inputs, num_items_in_batch=num_items_in_batch)
        model.train()
        remove_layers(model)

        
        repulsion_loss = harmful_loss * (-1) 
        grads = torch.autograd.grad(
            outputs=repulsion_loss,
            inputs=self.reversed_logits
        )

        for param, grad in zip(self.reversed_logits, grads):
            if grad is not None:
                param.grad = grad.detach()
            else:
                print(f"[WARN] Grad is None for a reversed_logit parameter")


        self.rev_inner_step += 1
        if self.rev_inner_step % self.args.gradient_accumulation_steps == 0:

            self.rev_optimizer.step()
            self.rev_optimizer.zero_grad()


        div_loss = self.compute_mse_consistency_loss_perturb_detach(
            target_logits=logits,                   
            perturbed_logits=harmful_logit,    
            labels=inputs["labels"],
            label_pad_token_id=-100
        ) / self.args.gradient_accumulation_steps



        final_loss=loss+self.finetuning_args.beta*div_loss

        wandb.log({"train/div_loss": (self.finetuning_args.beta *div_loss).item()}, step=self.state.global_step)
        wandb.log({"train/final_loss": final_loss.item()}, step=self.state.global_step)
        wandb.log({"train/loss": loss.item()}, step=self.state.global_step)
        wandb.log({"train/harmful_loss": harmful_loss.item()}, step=self.state.global_step)



        self.accelerator.backward(final_loss)
  

        return final_loss.detach()


import torch
from torch.nn import functional as F
from torch import nn
from transformers import PreTrainedModel
from torch import Tensor
import numpy as np



class REPLayer(nn.Module):
    def __init__(self, direction, lam):
        super().__init__()
        self.direction = direction  
        self.lam = lam                      

    def forward(self, x):
        if self.direction is not None:
            norm = torch.norm(x, dim=-1, keepdim=True)  
            
            direction = F.normalize(self.direction, dim=-1)  
            direction = direction.expand(x.size(0), x.size(1), -1)  

            x = F.normalize(x.float(), dim=-1) + self.lam * direction  
            x = F.normalize(x, dim=-1) * norm 

        return x 


def get_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 10:
            return module
    raise ValueError("No valid layer list found")


def find_module(block, keywords):
    for name, module in block.named_modules():
        if any(k in name.lower() for k in keywords):
            return module
    raise ValueError(f"Cannot find keyword {keywords} in block: {block}")


def add_layers(model: PreTrainedModel, directions: list[torch.Tensor], alphas: list[float]):

    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    assert len(directions) == len(layers)
    assert len(alphas) == len(layers)

    for i, layer in enumerate(layers):
        original_mlp = find_module(layer, mlp_keywords)
        repe = REPLayer(directions[i], alphas[i])
        layer.mlp = nn.Sequential(original_mlp, repe)


def remove_layers(model):
    layers = get_layers(model)
    for layer in layers:
        if isinstance(layer.mlp, nn.Sequential):
            layer.mlp = layer.mlp[0]