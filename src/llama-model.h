#pragma once

#include "llama.h"
#include "llama-arch.h"
#include "llama-hparams.h"
#include "llama-vocab.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_model_loader;

// available models
enum llm_type {
    LLM_TYPE_UNKNOWN,
    LLM_TYPE_14M,
    LLM_TYPE_17M,
    LLM_TYPE_22M,
    LLM_TYPE_33M,
    LLM_TYPE_60M,
    LLM_TYPE_70M,
    LLM_TYPE_80M,
    LLM_TYPE_109M,
    LLM_TYPE_137M,
    LLM_TYPE_160M,
    LLM_TYPE_220M,
    LLM_TYPE_250M,
    LLM_TYPE_270M,
    LLM_TYPE_335M,
    LLM_TYPE_410M,
    LLM_TYPE_450M,
    LLM_TYPE_770M,
    LLM_TYPE_780M,
    LLM_TYPE_0_5B,
    LLM_TYPE_1B,
    LLM_TYPE_1_3B,
    LLM_TYPE_1_4B,
    LLM_TYPE_1_5B,
    LLM_TYPE_1_6B,
    LLM_TYPE_2B,
    LLM_TYPE_2_8B,
    LLM_TYPE_3B,
    LLM_TYPE_4B,
    LLM_TYPE_6B,
    LLM_TYPE_6_9B,
    LLM_TYPE_7B,
    LLM_TYPE_8B,
    LLM_TYPE_9B,
    LLM_TYPE_11B,
    LLM_TYPE_12B,
    LLM_TYPE_13B,
    LLM_TYPE_14B,
    LLM_TYPE_15B,
    LLM_TYPE_16B,
    LLM_TYPE_20B,
    LLM_TYPE_30B,
    LLM_TYPE_32B,
    LLM_TYPE_34B,
    LLM_TYPE_35B,
    LLM_TYPE_40B,
    LLM_TYPE_65B,
    LLM_TYPE_70B,
    LLM_TYPE_236B,
    LLM_TYPE_314B,
    LLM_TYPE_671B,
    LLM_TYPE_SMALL,
    LLM_TYPE_MEDIUM,
    LLM_TYPE_LARGE,
    LLM_TYPE_XL,
    LLM_TYPE_A1_7B,
    LLM_TYPE_A2_7B,
    LLM_TYPE_8x7B,
    LLM_TYPE_8x22B,
    LLM_TYPE_16x12B,
    LLM_TYPE_16x3_8B,
    LLM_TYPE_10B_128x3_66B,
    LLM_TYPE_57B_A14B,
    LLM_TYPE_27B,
};

struct llama_layer_posnet {
    // resnet
    struct ggml_tensor * norm1   = nullptr;
    struct ggml_tensor * norm1_b = nullptr;

    struct ggml_tensor * conv1   = nullptr;
    struct ggml_tensor * conv1_b = nullptr;

    struct ggml_tensor * norm2   = nullptr;
    struct ggml_tensor * norm2_b = nullptr;

    struct ggml_tensor * conv2   = nullptr;
    struct ggml_tensor * conv2_b = nullptr;

    // attention
    struct ggml_tensor * attn_norm   = nullptr;
    struct ggml_tensor * attn_norm_b = nullptr;

    struct ggml_tensor * attn_q   = nullptr;
    struct ggml_tensor * attn_q_b = nullptr;

    struct ggml_tensor * attn_k   = nullptr;
    struct ggml_tensor * attn_k_b = nullptr;

    struct ggml_tensor * attn_v   = nullptr;
    struct ggml_tensor * attn_v_b = nullptr;

    struct ggml_tensor * attn_o   = nullptr;
    struct ggml_tensor * attn_o_b = nullptr;

    // normalize
    struct ggml_tensor * norm   = nullptr;
    struct ggml_tensor * norm_b = nullptr;
};

struct llama_layer_convnext {
    struct ggml_tensor * dw   = nullptr;
    struct ggml_tensor * dw_b = nullptr;

    struct ggml_tensor * norm   = nullptr;
    struct ggml_tensor * norm_b = nullptr;

    struct ggml_tensor * pw1   = nullptr;
    struct ggml_tensor * pw1_b = nullptr;

    struct ggml_tensor * pw2   = nullptr;
    struct ggml_tensor * pw2_b = nullptr;

    struct ggml_tensor * gamma = nullptr;
};

struct llama_layer {
    // Normalization
    struct ggml_tensor * attn_norm       = nullptr; // Input RMS norm (already present)
    struct ggml_tensor * attn_norm_b     = nullptr; // Bias for attn_norm (already present)
    struct ggml_tensor * attn_norm_2     = nullptr; // (already present, optional for DeepSeek V3)
    struct ggml_tensor * attn_norm_2_b   = nullptr; // (already present)
    struct ggml_tensor * attn_q_norm     = nullptr; // (already present, optional)
    struct ggml_tensor * attn_q_norm_b   = nullptr; // (already present)
    struct ggml_tensor * attn_k_norm     = nullptr; // (already present, optional)
    struct ggml_tensor * attn_k_norm_b   = nullptr; // (already present)
    struct ggml_tensor * attn_out_norm   = nullptr; // (already present, optional)
    struct ggml_tensor * attn_out_norm_b = nullptr; // (already present)
    struct ggml_tensor * attn_q_a_norm   = nullptr; // Query LoRA A norm (needed for DeepSeek V3)
    struct ggml_tensor * attn_kv_a_norm  = nullptr; // KV A norm (needed for DeepSeek V3, renamed from attn_kv_a_norm for clarity)
    struct ggml_tensor * attn_sub_norm   = nullptr; // (already present, optional)
    struct ggml_tensor * attn_post_norm  = nullptr; // (already present, optional)
    struct ggml_tensor * ffn_sub_norm    = nullptr; // (already present, optional)
    struct ggml_tensor * attn_norm_cross = nullptr; // (already present, optional)
    struct ggml_tensor * attn_norm_enc   = nullptr; // (already present, optional)
    struct ggml_tensor * ffn_norm        = nullptr; // Post-attention RMS norm (already present)
    struct ggml_tensor * ffn_norm_b      = nullptr; // Bias for ffn_norm (already present)
    struct ggml_tensor * ffn_post_norm    = nullptr; // (already present, optional)
    struct ggml_tensor * layer_out_norm   = nullptr; // (already present, optional)
    struct ggml_tensor * layer_out_norm_b = nullptr; // (already present)
    struct ggml_tensor * ffn_norm_exps    = nullptr; // (already present, optional for MoE)
    struct ggml_tensor * ffn_norm_enc     = nullptr; // (already present, optional)

    // Attention
    struct ggml_tensor * wq        = nullptr; // Query projection (already present, used if q_lora_rank == 0)
    struct ggml_tensor * wk        = nullptr; // Key projection (already present)
    struct ggml_tensor * wv        = nullptr; // Value projection (already present)
    struct ggml_tensor * wo        = nullptr; // Output projection (already present)
    struct ggml_tensor * wqkv      = nullptr; // Combined QKV projection (already present, optional)
    struct ggml_tensor * wq_a      = nullptr; // Query LoRA A projection (already present, needed for DeepSeek V3)
    struct ggml_tensor * wq_b      = nullptr; // Query LoRA B projection (already present, needed for DeepSeek V3)
    struct ggml_tensor * wkv_a_mqa = nullptr; // KV A projection with MQA (already present, needed for DeepSeek V3)
    struct ggml_tensor * wkv_b     = nullptr; // KV B projection (already present, needed for DeepSeek V3)
    struct ggml_tensor * wq_cross  = nullptr; // (already present, optional)
    struct ggml_tensor * wk_cross  = nullptr; // (already present, optional)
    struct ggml_tensor * wv_cross  = nullptr; // (already present, optional)
    struct ggml_tensor * wo_cross  = nullptr; // (already present, optional)
    struct ggml_tensor * wq_enc    = nullptr; // (already present, optional)
    struct ggml_tensor * wk_enc    = nullptr; // (already present, optional)
    struct ggml_tensor * wv_enc    = nullptr; // (already present, optional)
    struct ggml_tensor * wo_enc    = nullptr; // (already present, optional)

    // Attention bias (optional, depending on DeepSeek V3 implementation)
    struct ggml_tensor * bq   = nullptr; // (already present)
    struct ggml_tensor * bk   = nullptr; // (already present)
    struct ggml_tensor * bv   = nullptr; // (already present)
    struct ggml_tensor * bo   = nullptr; // (already present)
    struct ggml_tensor * bqkv = nullptr; // (already present)

    // Relative position bias (optional)
    struct ggml_tensor * attn_rel_b       = nullptr; // (already present)
    struct ggml_tensor * attn_rel_b_enc   = nullptr; // (already present)
    struct ggml_tensor * attn_rel_b_cross = nullptr; // (already present)

    // Feed-forward (FF) standard components
    struct ggml_tensor * ffn_gate     = nullptr; // w1 (already present)
    struct ggml_tensor * ffn_down     = nullptr; // w2 (already present)
    struct ggml_tensor * ffn_up       = nullptr; // w3 (already present)
    struct ggml_tensor * ffn_gate_enc = nullptr; // (already present, optional)
    struct ggml_tensor * ffn_down_enc = nullptr; // (already present, optional)
    struct ggml_tensor * ffn_up_enc   = nullptr; // (already present, optional)

    // FF MoE components
    struct ggml_tensor * ffn_gate_inp  = nullptr; // Gating linear layer for MoE (already present, needed for DeepSeek V3)
    struct ggml_tensor * ffn_gate_exps = nullptr; // Expert gate projections (already present, needed for DeepSeek V3)
    struct ggml_tensor * ffn_down_exps = nullptr; // Expert down projections (already present, needed for DeepSeek V3)
    struct ggml_tensor * ffn_up_exps   = nullptr; // Expert up projections (already present, needed for DeepSeek V3)

    // FF shared expert (shexp) components
    struct ggml_tensor * ffn_gate_inp_shexp = nullptr; // (already present, needed if n_shared_experts > 0)
    struct ggml_tensor * ffn_gate_shexp     = nullptr; // (already present, needed if n_shared_experts > 0)
    struct ggml_tensor * ffn_down_shexp     = nullptr; // (already present, needed if n_shared_experts > 0)
    struct ggml_tensor * ffn_up_shexp       = nullptr; // (already present, needed if n_shared_experts > 0)

    // FF bias and additional MoE components
    struct ggml_tensor * ffn_gate_b = nullptr; // (already present, optional)
    struct ggml_tensor * ffn_down_b = nullptr; // b2 (already present, optional)
    struct ggml_tensor * ffn_up_b   = nullptr; // b3 (already present, optional)
    struct ggml_tensor * ffn_act    = nullptr; // (already present, optional)
    struct ggml_tensor * ffn_exp_probs_b = nullptr; // Expert score correction bias (already present, needed for DeepSeek V3)

    // Mamba, RWKV, and other fields remain unchanged as they are unrelated to DeepSeek V3
    struct ggml_tensor * ssm_in  = nullptr;
    struct ggml_tensor * ssm_x   = nullptr;
    struct ggml_tensor * ssm_dt  = nullptr;
    struct ggml_tensor * ssm_out = nullptr;
    struct ggml_tensor * ssm_conv1d = nullptr;
    struct ggml_tensor * ssm_a      = nullptr;
    struct ggml_tensor * ssm_d      = nullptr;
    struct ggml_tensor * ssm_conv1d_b = nullptr;
    struct ggml_tensor * ssm_dt_b     = nullptr;

    struct ggml_tensor * time_mix_w1         = nullptr;
    struct ggml_tensor * time_mix_w2         = nullptr;
    struct ggml_tensor * time_mix_lerp_x     = nullptr;
    struct ggml_tensor * time_mix_lerp_w     = nullptr;
    struct ggml_tensor * time_mix_lerp_k     = nullptr;
    struct ggml_tensor * time_mix_lerp_v     = nullptr;
    struct ggml_tensor * time_mix_lerp_r     = nullptr;
    struct ggml_tensor * time_mix_lerp_g     = nullptr;
    struct ggml_tensor * time_mix_lerp_fused = nullptr;

    struct ggml_tensor * time_mix_first        = nullptr;
    struct ggml_tensor * time_mix_decay        = nullptr;
    struct ggml_tensor * time_mix_decay_w1     = nullptr;
    struct ggml_tensor * time_mix_decay_w2     = nullptr;
    struct ggml_tensor * time_mix_key          = nullptr;
    struct ggml_tensor * time_mix_key_b        = nullptr;
    struct ggml_tensor * time_mix_value        = nullptr;
    struct ggml_tensor * time_mix_value_b      = nullptr;
    struct ggml_tensor * time_mix_receptance   = nullptr;
    struct ggml_tensor * time_mix_receptance_b = nullptr;
    struct ggml_tensor * time_mix_gate         = nullptr;

    struct ggml_tensor * time_mix_ln     = nullptr;
    struct ggml_tensor * time_mix_ln_b   = nullptr;
    struct ggml_tensor * time_mix_output = nullptr;

    struct ggml_tensor * channel_mix_lerp_k = nullptr;
    struct ggml_tensor * channel_mix_lerp_r = nullptr;

    struct ggml_tensor * channel_mix_key        = nullptr;
    struct ggml_tensor * channel_mix_receptance = nullptr;
    struct ggml_tensor * channel_mix_value      = nullptr;

    struct ggml_tensor * rope_long  = nullptr;
    struct ggml_tensor * rope_short = nullptr;
    struct ggml_tensor * rope_freqs = nullptr;

    struct ggml_tensor * wq_scale       = nullptr;
    struct ggml_tensor * wk_scale       = nullptr;
    struct ggml_tensor * wv_scale       = nullptr;
    struct ggml_tensor * wo_scale       = nullptr;
    struct ggml_tensor * ffn_gate_scale = nullptr;
    struct ggml_tensor * ffn_up_scale   = nullptr;
    struct ggml_tensor * ffn_down_scale = nullptr;

    struct llama_layer_posnet posnet;
    struct llama_layer_convnext convnext;
};

struct llama_model {
    llm_type type = LLM_TYPE_UNKNOWN;
    llm_arch arch = LLM_ARCH_UNKNOWN;

    std::string name = "n/a";

    llama_hparams hparams = {};
    llama_vocab   vocab;

    struct ggml_tensor * tok_embd   = nullptr;
    struct ggml_tensor * type_embd  = nullptr;
    struct ggml_tensor * pos_embd   = nullptr;
    struct ggml_tensor * tok_norm   = nullptr;
    struct ggml_tensor * tok_norm_b = nullptr;

    struct ggml_tensor * output_norm     = nullptr;
    struct ggml_tensor * output_norm_b   = nullptr;
    struct ggml_tensor * output          = nullptr;
    struct ggml_tensor * output_b        = nullptr;
    struct ggml_tensor * output_norm_enc = nullptr;

    // classifier
    struct ggml_tensor * cls       = nullptr;
    struct ggml_tensor * cls_b     = nullptr;
    struct ggml_tensor * cls_out   = nullptr;
    struct ggml_tensor * cls_out_b = nullptr;

    struct ggml_tensor * conv1d   = nullptr;
    struct ggml_tensor * conv1d_b = nullptr;

    std::vector<llama_layer> layers;

    llama_model_params params;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // list of devices used in this model
    std::vector<ggml_backend_dev_t> devices;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    int64_t t_load_us  = 0;
    int64_t t_start_us = 0;

    explicit llama_model(const struct llama_model_params & params);
    ~llama_model();

    void load_stats  (llama_model_loader & ml);
    void load_arch   (llama_model_loader & ml);
    void load_hparams(llama_model_loader & ml);
    void load_vocab  (llama_model_loader & ml);
    bool load_tensors(llama_model_loader & ml); // returns false if cancelled by progress_callback

    std::string arch_name() const;
    std::string type_name() const;

    std::string desc() const;

    size_t size() const;
    size_t max_nodes() const;
    size_t n_devices() const;

    // total number of parameters in the model
    uint64_t n_elements() const;

    void print_info() const;

    ggml_backend_dev_t dev_layer(int il) const;
    ggml_backend_dev_t dev_output() const;

    ggml_backend_buffer_type_t select_buft(int il) const;

    const struct ggml_tensor * get_tensor(const char * name) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

const char * llm_type_name(llm_type type);
