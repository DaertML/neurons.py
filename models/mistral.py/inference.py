from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json
#import matplotlib.pyplot as plt
import json

def load_tokenizer_DW():
  tokenizer_path = "./mistral/tokenizer.model"

  special_tokens_map = json.loads(open("./mistral/special_tokens_map.json", "r").read())

  special_tokens = []

  for tok in special_tokens_map.keys():
    special_tokens.append(special_tokens_map[tok]["content"])

  mergeable_ranks = load_tiktoken_bpe("./mistral/tokenizer.json")

  tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
  )

  tokenizer.decode(tokenizer.encode("hello world!"))


# Load tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./mistral")


# Load model
import torch
model = torch.load("./mistral/consolidated.0.mistral.pth")

for layer in model.keys():
  print(layer)

with open("./mistral/params.json", "r") as f:
    config = json.load(f)
print(config)

dim = config["dim"]
n_layers = config["n_layers"]
n_heads = config["n_heads"]
n_kv_heads = config["n_kv_heads"]
vocab_size = config["vocab_size"]
multiple_of = config["multiple_of"]
norm_eps = config["norm_eps"]
rope_theta = config["rope_theta"]

# Gen tokens
prompt = "the answer to the ultimate question of life, the universe, and everything is "
tokens = tokenizer.encode(prompt)
print(tokens)
tokens = torch.tensor(tokens)
prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens]
print(prompt_split_as_tokens)

# Gen embeddings
print("Vocab size", vocab_size)
print("Dim", dim)

embedding_layer = torch.nn.Embedding(vocab_size, dim)
embedding_layer.weight.data.copy_(model["model.embed_tokens.weight"])

print(model["model.embed_tokens.weight"].shape)

print("Tokens", tokens)
print("Tokens shape", tokens.shape)


if torch.any(tokens >= vocab_size) or torch.any(tokens < 0):
    raise ValueError("Input tensor contains out-of-range indices.")

token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)
print("Unnormalized tokens shape",token_embeddings_unnormalized.shape)

# Run RMSNorm
def rms_norm(tensor, norm_weights):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights

token_embeddings = rms_norm(token_embeddings_unnormalized, model["model.layers.0.input_layernorm.weight"])
print("Normalized tokens shape",token_embeddings.shape)

# Attention

wq = model["model.layers.0.self_attn.q_proj.weight"]
print("Wrapped Q", wq.shape)

wk = model["model.layers.0.self_attn.k_proj.weight"]
wv = model["model.layers.0.self_attn.v_proj.weight"]
wo = model["model.layers.0.self_attn.o_proj.weight"]

#  - unwrap the heads from the weight matrices of q, k, v
q_layer0 = model["model.layers.0.self_attn.q_proj.weight"]
k_layer0 = model["model.layers.0.self_attn.k_proj.weight"]
v_layer0 = model["model.layers.0.self_attn.v_proj.weight"]

head_dim = q_layer0.shape[0] // n_heads

qkv_attention_store = []

n_splits = int(n_kv_heads/2)
zero_to_one_split_into_n_parts = torch.tensor(range(n_splits))/n_splits
freqs = 1.0 / (rope_theta ** zero_to_one_split_into_n_parts)
freqs_for_each_token = torch.outer(torch.arange(17), freqs)
freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)

final_embedding = token_embeddings_unnormalized
for layer in range(n_layers):
    qkv_attention_store = []
    layer_embedding_norm = rms_norm(final_embedding, model[f"model.layers.{layer}.input_layernorm.weight"])
    q_layer = model[f"model.layers.{layer}.self_attn.q_proj.weight"]
    q_layer = q_layer.view(n_heads, q_layer.shape[0] // n_heads, dim)
    k_layer = model[f"model.layers.{layer}.self_attn.k_proj.weight"]
    k_layer = k_layer.view(n_kv_heads, k_layer.shape[0] // n_kv_heads, dim)
    v_layer = model[f"model.layers.{layer}.self_attn.v_proj.weight"]
    v_layer = v_layer.view(n_kv_heads, v_layer.shape[0] // n_kv_heads, dim)
    w_layer = model[f"model.layers.{layer}.self_attn.o_proj.weight"]
    for head in range(n_heads):
        q_layer_head = q_layer[head]
        k_layer_head = k_layer[head//n_splits]
        v_layer_head = v_layer[head//n_splits]
        q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
        k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
        v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)
        q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
        q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
        q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis)
        q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
        k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
        k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
        k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
        k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
        qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5
        mask = torch.full((len(token_embeddings_unnormalized), len(token_embeddings_unnormalized)), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        qk_per_token_after_masking = qk_per_token + mask
        qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
        qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
        qkv_attention_store.append(qkv_attention)

    stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
    w_layer = model[f"model.layers.{layer}.self_attn.o_proj.weight"]
    embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
    embedding_after_edit = final_embedding + embedding_delta
    embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"model.layers.{layer}.post_attention_layernorm.weight"])
    w1 = model[f"model.layers.{layer}.mlp.gate_proj.weight"]
    w2 = model[f"model.layers.{layer}.mlp.up_proj.weight"]
    w3 = model[f"model.layers.{layer}.mlp.down_proj.weight"]
    output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3), w2)
    final_embedding = embedding_after_edit+output_after_feedforward

final_embedding = rms_norm(final_embedding, model["model.norm.weight"])
print(final_embedding.shape)
logits = torch.matmul(final_embedding[-1], model["model.embed_tokens.weight"].T)
next_token = torch.argmax(logits, dim=-1)
print(  tokenizer.decode(next_token))
