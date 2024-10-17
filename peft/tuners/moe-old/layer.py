# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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

import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose

rms_norm = None

class LoraMoeLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ["lora_moe_A", "lora_moe_B", "lora_moe_embedding_A", "lora_moe_embedding_B"]

    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_moe_dropout = nn.ModuleDict({})
        self.lora_moe_A = nn.ModuleDict({})
        self.lora_moe_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_moe_embedding_A = nn.ParameterDict({})
        self.lora_moe_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def _init_empty_weights(self, cls, *args, **kwargs) -> None:
        # A helper method that allows to initialize the layer of the given class without spending time to initialize the
        # model weights. The implementation is inspired by
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.skip_init.html but this function cannot be used
        # directly.
        # Instead of this approach, it would be possible to bypass the __init__ of the class but that runs the risk of
        # omitting important logic inside that __init__.
        kwargs = kwargs.copy()
        final_device = kwargs.pop("device", "cpu")
        cls.__init__(self, *args, device="meta", **kwargs)
        self.to_empty(device=final_device)

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_moe_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_moe_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
            self.lora_moe_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        # print(self.weight)
        weight = getattr(self, "weight", None)
        
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def update_layer_conv2d(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_moe_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        if r > 0:
            kernel_size = self.kwargs["kernel_size"]
            stride = self.kwargs["stride"]
            padding = self.kwargs["padding"]
            self.lora_moe_A[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
            self.lora_moe_B[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(self.weight.device, dtype=weight.dtype)

    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_moe_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        if r > 0:
            weight_A = torch.randn((r, self.in_features))
            weight_B = torch.randn((self.out_features, r))
            self.lora_moe_embedding_A[adapter_name] = nn.Parameter(weight_A)
            self.lora_moe_embedding_B[adapter_name] = nn.Parameter(weight_B)
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(self.weight.device, dtype=weight.dtype)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_moe_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_moe_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_moe_B[adapter_name].weight)
        if adapter_name in self.lora_moe_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_moe_embedding_A[adapter_name])
            nn.init.normal_(self.lora_moe_embedding_B[adapter_name])

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_moe_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_moe_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale


class MlpMoeLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ["mlp_moe", "mlp_moe_embedding"]

    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.mlp_moe_dropout = nn.ModuleDict({})
        self.mlp_moe = nn.ModuleDict({})
        # For Embedding layer
        self.mlp_moe_embedding = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def _init_empty_weights(self, cls, *args, **kwargs) -> None:
        # A helper method that allows to initialize the layer of the given class without spending time to initialize the
        # model weights. The implementation is inspired by
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.skip_init.html but this function cannot be used
        # directly.
        # Instead of this approach, it would be possible to bypass the __init__ of the class but that runs the risk of
        # omitting important logic inside that __init__.
        kwargs = kwargs.copy()
        final_device = kwargs.pop("device", "cpu")
        cls.__init__(self, *args, device="meta", **kwargs)
        self.to_empty(device=final_device)

    def update_layer(self, adapter_name, mlp_dropout, init_mlp_weights):
        if mlp_dropout > 0.0:
            mlp_dropout_layer = nn.Dropout(p=mlp_dropout)
        else:
            mlp_dropout_layer = nn.Identity()

        self.mlp_moe_dropout.update(nn.ModuleDict({adapter_name: mlp_dropout_layer}))
        # Actual trainable parameters
        self.mlp_moe[adapter_name] = nn.Linear(self.in_features, self.out_features, bias=True)
        if init_mlp_weights:
            self.reset_mlp_parameters(adapter_name)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def update_layer_conv2d(self, adapter_name, mlp_dropout, init_mlp_weights):
        if mlp_dropout > 0.0:
            mlp_dropout_layer = nn.Dropout(p=mlp_dropout)
        else:
            mlp_dropout_layer = nn.Identity()

        self.mlp_moe_dropout[adapter_name] = mlp_dropout_layer
        # Actual trainable parameters
        kernel_size = self.kwargs["kernel_size"]
        stride = self.kwargs["stride"]
        padding = self.kwargs["padding"]
        self.mlp_moe[adapter_name] = nn.Conv2d(self.in_features, self.out_features, kernel_size, stride, padding, bias=False)
        if init_mlp_weights:
            self.reset_mlp_parameters(adapter_name)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(self.weight.device, dtype=weight.dtype)

    def update_layer_embedding(self, adapter_name, mlp_dropout, init_mlp_weights):
        if mlp_dropout > 0.0:
            mlp_dropout_layer = nn.Dropout(p=mlp_dropout)
        else:
            mlp_dropout_layer = nn.Identity()

        self.mlp_moe_dropout[adapter_name] = mlp_dropout_layer
        # Actual trainable parameters
        weight = torch.randn((self.out_features, self.in_features))
        self.mlp_moe_embedding[adapter_name] = nn.Parameter(weight)
        if init_mlp_weights:
            self.reset_mlp_parameters(adapter_name)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(self.weight.device, dtype=weight.dtype)

    def reset_mlp_parameters(self, adapter_name):
        if adapter_name in self.mlp_moe.keys():
            nn.init.zeros_(self.mlp_moe[adapter_name].weight)
            nn.init.zeros_(self.mlp_moe[adapter_name].bias)
        if adapter_name in self.mlp_moe_embedding.keys():
            nn.init.normal_(self.mlp_moe_embedding[adapter_name])


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class LoraLinear(nn.Linear, LoraMoeLayer):
    # Lora Moe implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        multiple_loras: bool = False,
        g_enable = False,
        noise_std: float = 0.1,
        gates_tmp: float = 1.0,
        topk = 1,
        num_experts = 4,
        loss_coef = 0.001,
        token=True,
        freeze_gate=False,
        **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        # this gets the init from nn.Linear's super perspective, i.e.
        # nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        # Note that we don't use self._init_empty_weights() for Linear because it is a bit slower and the benefit of
        # added robustness is not big enough for Linear.

        LoraMoeLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix

        self.fan_in_fan_out = fan_in_fan_out

        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.set_adapter(adapter_name)
        self.multiple_loras = multiple_loras
        self.g_enable = g_enable
        if self.multiple_loras:
            self.noise_std = noise_std
            self.gates_tmp = gates_tmp
            self.topk = topk
            self.num_experts = num_experts
            self.w_gate = nn.Linear(in_features, num_experts) 
            
            '''
            nn.Sequential(nn.Linear(in_features, in_features),
                nn.ReLU(),
                nn.Linear(in_features, num_experts))
            '''
            
            self.ln = RMSNorm(
                in_features,
                eps=1e-6,
            )
            self.loss_coef = loss_coef
        self.token=token
        self.freeze_gate=freeze_gate
        if self.freeze_gate:
            self.w_gate.requires_grad_(False)
            print('freeze gate now!')
        else:
            print('the gate is trainable now!')

    def merge(self, safe_merge: bool = False) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )
        for active_adapter in self.active_adapters:
            if active_adapter in self.lora_moe_A.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = self.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.weight.data = orig_weights
                else:
                    self.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_moe_A.keys():
                self.weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_moe_B[adapter].weight.device
        dtype = self.lora_moe_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_moe_A[adapter].weight
        weight_B = self.lora_moe_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_moe_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_moe_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    ### begin of additional functions for LoRA MoE
    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)
    
    def _prob_in_top_k(
            self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        # print(noisy_values.shape)  #bs,n
        # print(clean_values.shape)  #bs,n
        # print(noise_stddev.shape)
        # print(noisy_top_values.shape)  # bs,topk
        batch = clean_values.size(0)
        m = noisy_top_values.size(1) # (B*50, top_k+1)
        top_values_flat = noisy_top_values.flatten()
        # print(top_values_flat.shape)  # bs*topk
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.topk
        # print(threshold_positions_if_in.shape)  # bs
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1) # (B*50, 1)
        # print(threshold_if_in.shape)  # bs,1
        is_in = torch.gt(noisy_values, threshold_if_in)  # (B*50, 4) similar to gate results.
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(
            torch.tensor([0.0], device=clean_values.device),
            torch.tensor([1.0], device=clean_values.device),
        )
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_expert = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean() ** 2 + eps)
    # end of additional functions for LoRA MoE
    
    def forward(self, x: torch.Tensor, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._linear(x)
        elif self.merged:
            result = self._linear(x)
        else:
            result = self._linear(x)
            if self.multiple_loras:
                if self.token:
                    # all-token start
                    bs, sl, _ = x.size()
                    x = x.view(bs * sl, -1)
                    clean_logits = self.w_gate(self.ln(inputs_embeds))
                    
                    # all-token end
                # elif self.sentence:
                else:
                    # all-sentence 
                    ### embeding average
                    new_embeds = [] # bs,dim
                    for i in range(attention_mask.size(0)):
                        # prompt_embeds = inputs_embeds[i][attention_mask[i][0]: attention_mask[i][1]] # pl,dim
                        prompt_embeds = inputs_embeds[i][attention_mask[i][0]: attention_mask[i][1]] # imgl+pl,dim
                        prompt_embeds = prompt_embeds.mean(0)  # 1,dim

                        new_embeds.append(prompt_embeds) 

                    prompt_embeds = torch.stack(new_embeds, dim=0) # bs,dim
                    # print(prompt_embeds.shape)
                    clean_logits = self.w_gate(self.ln(prompt_embeds)) # bs, n
                    # print(clean_logits.shape)
                    

                    ### logits average
                    # clean_logits = self.w_gate(self.ln(inputs_embeds))#.mean(-2)  # bs,sl,num_gate ---calculate embedding for gating for each token 
                    
                    # new_logits = []
                    # for i in range(attention_mask.size(0)):
                    #     logit = clean_logits[i][attention_mask[i][0]: attention_mask[i][1]]   # pl,num_expert---fetch gating embedding for peompt tokens(only text prompt, except for image tokens) 
                    #     logit = logit.mean(0)  # 1, num_expert---on average
                        
                    #     new_logits.append(logit) # bs,1,num_expert
                    
                    
                    # clean_logits = torch.stack(new_logits, dim=0) # bs, num_expert

                    ### 整句average
                    # clean_logits = clean_logits * attention_mask.unsqueeze(-1)
                    # clean_logits = clean_logits.mean(-2)
                    

                    # all-sentence end
                # elif self.hierarchiacl_sentence:
                #     # 根据当前token位置
                #     bs, sl, _ = x.size()
                #     x = x.view(bs * sl, -1)
                #     clean_logits = self.w_gate(self.ln(inputs_embeds))
                #     for i in range(bs):
                #         current_data_logit = []
                #         for j in range(sl):  
                #             logit = clean_logits[i][0: j]



                raw_noise_stddev = self.noise_std
                noise_stddev = raw_noise_stddev * self.training
                noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
                logits = noisy_logits
                logits = F.softmax(logits / self.gates_tmp, dim=1, dtype=torch.float16)
                top_logits, top_indices = logits.topk(min(self.topk + 1, self.num_experts), dim=1)
                top_k_logits = top_logits[:, :self.topk]
                top_k_indices = top_indices[:, :self.topk]
                top_k_gates = top_k_logits

                zeros = torch.zeros_like(logits, requires_grad=True)
                gates = zeros.scatter(1, top_k_indices, top_k_gates)
                gate_load = gates.gt(0).sum(0)
                
                if self.training:
                    importance = gates.sum(0)
                    load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
                    load_loss = (self.cv_squared(importance) + self.cv_squared(load)) * self.loss_coef
                else:
                    load_loss = None
                try :
                    dispatcher = SparseDispatcher(self.num_experts, gates, token=self.token)
                    expert_inputs = list(dispatcher.dispatch(x))
                    gates_ = dispatcher.expert_to_gates()
                except:
                    #print(inputs_embeds)
                    print("clean_logits : ", clean_logits)
                    print("logit : ", logit)
                    exit()

                if self.g_enable:
                    expert_outputs = []
                    for i, adpt in enumerate(list(self.r.keys())[:-1]):
                        expert_inputs[i] = expert_inputs[i].to(self.lora_moe_A[adpt].weight.dtype)
                        expert_outputs.append(self.lora_moe_B[adpt](self.lora_moe_A[adpt](self.lora_moe_dropout[adpt](expert_inputs[i]))) * self.scaling[adpt])
                    y_e = dispatcher.combine(expert_outputs)

                    # universal expert
                    lora_moe_A = self.lora_moe_A['g']
                    lora_moe_B = self.lora_moe_B['g']
                    dropout = self.lora_moe_dropout['g']
                    scaling = self.scaling['g']
                    x = x.to(lora_moe_A.weight.dtype)
                    y_g = lora_moe_B(lora_moe_A(dropout(x))) * scaling

                    if self.token:
                        w = gates.max(dim=1)[0].unsqueeze(-1)
                    else:
                        w = gates.max(dim=1)[0].unsqueeze(-1).unsqueeze(-1)
                    y = y_e * w + y_g * (1-w)
                else:
                    expert_outputs = []
                    for i, adpt in enumerate(list(self.r.keys())):
                        expert_inputs[i] = expert_inputs[i].to(self.lora_moe_A[adpt].weight.dtype)
                        expert_outputs.append(self.lora_moe_B[adpt](self.lora_moe_A[adpt](self.lora_moe_dropout[adpt](expert_inputs[i]))) * self.scaling[adpt])
                    y = dispatcher.combine(expert_outputs)

                if self.token:
                    # all-token start
                    y = y.view(bs, sl, -1)
                    # all-token end
                result += y
               
            else:
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.lora_moe_A.keys():
                        continue
                    lora_moe_A = self.lora_moe_A[active_adapter]
                    lora_moe_B = self.lora_moe_B[active_adapter]
                    dropout = self.lora_moe_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]
                    x = x.to(lora_moe_A.weight.dtype)
                    result += lora_moe_B(lora_moe_A(dropout(x))) * scaling

        result = result.to(previous_dtype)
        if self.freeze_gate:
            return result, None, gates
        else:
            return result, load_loss, gates
        #return result, load_loss, gates
        # return result


class LoraLinear_noGate(nn.Linear, LoraMoeLayer):
    # Lora Moe implemented in a dense layer without Gate
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        multiple_loras: bool = False,
        g_enable = False,
        noise_std: float = 0.1,
        topk = 1,
        num_experts = 4,
        token=True,
        freeze_gate=False,
        **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        # this gets the init from nn.Linear's super perspective, i.e.
        # nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        # Note that we don't use self._init_empty_weights() for Linear because it is a bit slower and the benefit of
        # added robustness is not big enough for Linear.

        LoraMoeLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix

        self.fan_in_fan_out = fan_in_fan_out

        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.set_adapter(adapter_name)
        self.multiple_loras = multiple_loras
        self.g_enable = g_enable
        if self.multiple_loras:
            self.noise_std = noise_std
            self.topk = topk
            self.num_experts = num_experts
            #self.w_gate = nn.Linear(in_features, num_experts)
        self.token=token
        self.freeze_gate=freeze_gate

    def merge(self, safe_merge: bool = False) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )
        for active_adapter in self.active_adapters:
            if active_adapter in self.lora_moe_A.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = self.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.weight.data = orig_weights
                else:
                    self.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_moe_A.keys():
                self.weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_moe_B[adapter].weight.device
        dtype = self.lora_moe_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_moe_A[adapter].weight
        weight_B = self.lora_moe_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_moe_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_moe_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    ### begin of additional functions for LoRA MoE
    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)
    
    def _prob_in_top_k(
            self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1) # (B*50, top_k+1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.topk
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1) # (B*50, 1)
        is_in = torch.gt(noisy_values, threshold_if_in)  # (B*50, 4) similar to gate results.
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(
            torch.tensor([0.0], device=clean_values.device),
            torch.tensor([1.0], device=clean_values.device),
        )
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_expert = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean() ** 2 + eps)
    # end of additional functions for LoRA MoE
    
    def forward(self, x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        assert gates is not None, "Gate can not be None!"
    
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._linear(x)
        elif self.merged:
            result = self._linear(x)
        else:
            result = self._linear(x)
            if self.multiple_loras:
                if self.token:
                    # all-token start
                    bs, sl, _ = x.size()
                    x = x.view(bs * sl, -1)
                gate_load = gates.gt(0).sum(0)

                load_loss = None
                
                dispatcher = SparseDispatcher(self.num_experts, gates, token=self.token)
                expert_inputs = list(dispatcher.dispatch(x))
                gates_ = dispatcher.expert_to_gates()
                
                if self.g_enable:
                    expert_outputs = []
                    for i, adpt in enumerate(list(self.r.keys())[:-1]):
                        expert_inputs[i] = expert_inputs[i].to(self.lora_moe_A[adpt].weight.dtype)
                        expert_outputs.append(self.lora_moe_B[adpt](self.lora_moe_A[adpt](self.lora_moe_dropout[adpt](expert_inputs[i]))) * self.scaling[adpt])
                    y_e = dispatcher.combine(expert_outputs)

                    lora_moe_A = self.lora_moe_A['g']
                    lora_moe_B = self.lora_moe_B['g']
                    dropout = self.lora_moe_dropout['g']
                    scaling = self.scaling['g']
                    x = x.to(lora_moe_A.weight.dtype)
                    y_g = lora_moe_B(lora_moe_A(dropout(x))) * scaling

                    if self.token:
                        w = gates.max(dim=1)[0].unsqueeze(-1)
                    else:
                        w = gates.max(dim=1)[0].unsqueeze(-1).unsqueeze(-1)
                    y = y_e * w + y_g * (1-w)
                else:
                    expert_outputs = []
                    for i, adpt in enumerate(list(self.r.keys())):
                        expert_inputs[i] = expert_inputs[i].to(self.lora_moe_A[adpt].weight.dtype)
                        expert_outputs.append(self.lora_moe_B[adpt](self.lora_moe_A[adpt](self.lora_moe_dropout[adpt](expert_inputs[i]))) * self.scaling[adpt])
                    y = dispatcher.combine(expert_outputs)

                if self.token:
                    # all-token start
                    y = y.view(bs, sl, -1)
                    # all-token end
                result += y
               
            else:
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.lora_moe_A.keys():
                        continue
                    lora_moe_A = self.lora_moe_A[active_adapter]
                    lora_moe_B = self.lora_moe_B[active_adapter]
                    dropout = self.lora_moe_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]
                    x = x.to(lora_moe_A.weight.dtype)
                    result += lora_moe_B(lora_moe_A(dropout(x))) * scaling
            
        result = result.to(previous_dtype)
        if self.freeze_gate:
            return result, None
        else:
            return result, load_loss
        # return result

class MlpLinear(nn.Linear, MlpMoeLayer):
    # Mlp Moe implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        mlp_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        multiple_mlps: bool = False,
        g_enable = False,
        noise_std: float = 0.1,
        gates_tmp: float = 1.0,
        topk = 1,
        num_experts = 4,
        loss_coef = 0.001,
        token=True,
        freeze_gate=False,
        **kwargs,
    ) -> None:
        init_mlp_weights = kwargs.pop("init_mlp_weights", True)
        # this gets the init from nn.Linear's super perspective, i.e.
        # nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        # Note that we don't use self._init_empty_weights() for Linear because it is a bit slower and the benefit of
        # added robustness is not big enough for Linear.

        MlpMoeLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix

        self.fan_in_fan_out = fan_in_fan_out

        self.update_layer(adapter_name, mlp_dropout, init_mlp_weights)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.set_adapter(adapter_name)
        self.multiple_mlps = multiple_mlps
        self.g_enable = g_enable
        if self.multiple_mlps:
            self.noise_std = noise_std
            self.gates_tmp = gates_tmp
            self.topk = topk
            self.num_experts = num_experts
            self.w_gate = nn.Linear(in_features, num_experts)
            '''
            nn.Sequential(nn.Linear(in_features, in_features),
                nn.ReLU(),
                nn.Linear(in_features, num_experts))
            '''
            self.loss_coef = loss_coef
        self.token=token
        self.freeze_gate=freeze_gate
        if self.freeze_gate:
            self.w_gate.requires_grad_(False)
            print('freeze gate now!')

    def merge(self, safe_merge: bool = False) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )
        for active_adapter in self.active_adapters:
            if active_adapter in self.mlp_moe.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = self.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.weight.data = orig_weights
                else:
                    self.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.mlp_moe.keys():
                self.weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.mlp_moe[adapter].weight.device
        dtype = self.mlp_moe[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight = self.mlp_moe[adapter].weight

        if cast_to_fp32:
            weight = weight.float()

        output_tensor = transpose(weight, self.fan_in_fan_out)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.mlp_moe[adapter].weight.data = weight.to(dtype)

        return output_tensor

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    ### begin of additional functions for MoE
    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)
    
    def _prob_in_top_k(
            self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1) # (B*50, top_k+1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.topk
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1) # (B*50, 1)
        is_in = torch.gt(noisy_values, threshold_if_in)  # (B*50, 4) similar to gate results.
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(
            torch.tensor([0.0], device=clean_values.device),
            torch.tensor([1.0], device=clean_values.device),
        )
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_expert = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean() ** 2 + eps)
    # end of additional functions for MoE
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._linear(x)
        elif self.merged:
            result = self._linear(x)
        else:
            result = None
            if self.multiple_mlps:
                if self.token:
                    # all-token start
                    bs, sl, _ = x.size()
                    x = x.view(bs * sl, -1)
                    clean_logits = self.w_gate(x)
                    # all-token end
                else:
                    # all-sentence start
                    ### embeding average
                    x_mean = x.mean(-2)
                    clean_logits = self.w_gate(x_mean)
                    ### logits average
                    # clean_logits = self.w_gate(x).mean(-2)
                    #clean_logits = clean_logits * attention_mask.unsqueeze(-1)
                    #clean_logits = clean_logits.mean(-2)
                    # all-sentence end
                    
                    
                
                raw_noise_stddev = self.noise_std
                noise_stddev = raw_noise_stddev * self.training
                noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
                logits = noisy_logits
                logits = F.softmax(logits / self.gates_tmp, dim=1, dtype=torch.float16)
                top_logits, top_indices = logits.topk(min(self.topk + 1, self.num_experts), dim=1)
                top_k_logits = top_logits[:, :self.topk]
                top_k_indices = top_indices[:, :self.topk]
                top_k_gates = top_k_logits

                zeros = torch.zeros_like(logits, requires_grad=True)
                gates = zeros.scatter(1, top_k_indices, top_k_gates)
                gate_load = gates.gt(0).sum(0)  # 每个gate有多少个数据经过

                if self.training:
                    importance = gates.sum(0)  
                    load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
                    load_loss = (self.cv_squared(importance) + self.cv_squared(load)) * self.loss_coef
                else:
                    load_loss = None

                dispatcher = SparseDispatcher(self.num_experts, gates, token=self.token)
                expert_inputs = list(dispatcher.dispatch(x))  # 把x分给experts
                gates_ = dispatcher.expert_to_gates()
                
                if self.g_enable:
                    expert_outputs = []
                    for i, adpt in enumerate(list(self.mlp_moe.keys())[:-1]):
                        expert_inputs[i] = expert_inputs[i].to(self.mlp_moe[adpt].weight.dtype)
                        expert_outputs.append(self.mlp_moe[adpt](self.mlp_moe_dropout[adpt](expert_inputs[i])))
                    y_e = dispatcher.combine(expert_outputs)

                    mlp_moe = self.mlp_moe['g']  # universal expert
                    dropout = self.mlp_moe_dropout['g']
                    x = x.to(mlp_moe.weight.dtype)
                    y_g = mlp_moe(dropout(x))

                    if self.token:
                        w = gates.max(dim=1)[0].unsqueeze(-1)
                    else:
                        w = gates.max(dim=1)[0].unsqueeze(-1).unsqueeze(-1)
                    y = y_e * w + y_g * (1-w)
                else:
                    expert_outputs = []
                    for i, adpt in enumerate(list(self.mlp_moe.keys())):
                        expert_inputs[i] = expert_inputs[i].to(self.mlp_moe[adpt].weight.dtype)
                        expert_outputs.append(self.mlp_moe[adpt](self.mlp_moe_dropout[adpt](expert_inputs[i])))
                    y = dispatcher.combine(expert_outputs)

                if self.token:
                    # all-token start
                    y = y.view(bs, sl, -1)
                    # all-token end
                if result is None:
                    result = y
                else:
                    result += y
               
            else:
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.mlp_moe.keys():
                        continue
                    mlp_moe = self.mlp_moe[active_adapter]
                    dropout = self.mlp_moe_dropout[active_adapter]
                    x = x.to(mlp_moe.weight.dtype)
                    if result is None:
                        result = mlp_moe(dropout(x))
                    else:
                        result += mlp_moe(dropout(x))
                load_loss = None

        result = result.to(previous_dtype)
        
        if self.freeze_gate:
            return result, None
        else:
            return result, load_loss
        # return result


class Embedding(nn.Embedding, LoraMoeLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        adapter_name: str,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self._init_empty_weights(nn.Embedding, num_embeddings, embedding_dim, **kwargs)
        LoraMoeLayer.__init__(self, in_features=num_embeddings, out_features=embedding_dim)
        self.update_layer_embedding(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.set_adapter(adapter_name)

    def merge(self, safe_merge: bool = False) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )
        for active_adapter in self.active_adapters:
            if active_adapter in self.lora_embedding_A.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = self.weight.data.copy()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.weight.data = orig_weights
                else:
                    self.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_embedding_A.keys():
                self.weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_embedding_A[adapter] = weight_A.to(dtype)
            self.lora_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor

    def _embed(self, input: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = self.weight if weight is None else weight
        return F.embedding(
            input,
            weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._embed(x)
        elif self.merged:
            result = self._embed(x)
        else:
            result = self._embed(x)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter].T
                embedding_B = self.lora_embedding_B[active_adapter].T
                scaling = self.scaling[active_adapter]
                after_A = self._embed(x, embedding_A)
                result += (after_A @ embedding_B) * scaling

        return result


class Conv2d(nn.Conv2d, LoraMoeLayer):
    # Lora implemented in a conv2d layer
    def __init__(
        self,
        adapter_name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self._init_empty_weights(nn.Conv2d, in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        LoraMoeLayer.__init__(
            self,
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.update_layer_conv2d(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.set_adapter(adapter_name)

    def merge(self, safe_merge: bool = False) -> None:
        """
        Merge the active adapter weights inside the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )
        for active_adapter in self.active_adapters:
            if active_adapter in self.lora_A.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = self.weight.data.copy()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    self.weight.data = orig_weights
                else:
                    self.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                self.weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_A[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
        if self.weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                3
            ) * self.scaling[adapter]
        else:
            # conv2d 3x3
            output_tensor = (
                F.conv2d(
                    weight_A.permute(1, 0, 2, 3),
                    weight_B,
                ).permute(1, 0, 2, 3)
                * self.scaling[adapter]
            )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def _conv2d(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            input,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._conv2d(x)
        elif self.merged:
            result = self._conv2d(x)
        else:
            result = self._conv2d(x)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                result += lora_B(lora_A(dropout(x))) * scaling

        result = result.to(previous_dtype)
        return result
    

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates, token=False):
        """Create a SparseDispatcher."""

        self._gates = gates     # dyq: [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0.5, 0, 0.5]]
        self._num_experts = num_experts
        self.token = token
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)      # dyq: [[0],[0],[1],[2],[2]]
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]     # dyq: [1, 3, 0, 2, 3]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()      # dyq: [2, 1, 2]
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]      # dyq: [[1, 0, 0],[0.5, 0, 0.5],[0, 1, 0],[0, 0, 1],[0.5, 0, 0.5]]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)    # dyq: [[1],[0.5],[1],[1],[0.5]]


    def dispatch(self, inp):    # dyq: [[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        # inp_exp = inp[self._batch_index].squeeze(1)

        # if self.token:
        #     inp_exp = inp[self._batch_index].squeeze(1)
        # else:
        #     inp_exp = inp[self._batch_index]

        # dyq start
        
        inp_exp = inp[self._batch_index]
        # dyq end
        
        return torch.split(inp_exp, self._part_sizes, dim=0)    # dyq: ([[2,2,2],[4,4,4]], [[1,1,1]], [[3,3,3],[4,4,4]])

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        # stitched = torch.cat(expert_out, 0).exp()
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            if self.token:
                stitched = stitched.mul(self._nonzero_gates)
            else:
                stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1))
        zeros = torch.zeros(self._gates.size(0), *expert_out[-1].shape[1:], requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts

        try:
            combined = zeros.index_add(0, self._batch_index, stitched.float())
        except RuntimeError:
            import ipdb
            ipdb.set_trace()
        # add eps to all zero values in order to avoid nans when going back to log space
        # combined[combined == 0] = np.finfo(float).eps
        # back to log space
        # return combined.log()
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if rms_norm is not None and x.is_cuda:
            return rms_norm(x, self.weight, self.eps)
        else:
            output = self._norm(x.float()).type_as(x)
            return output * self.weight
