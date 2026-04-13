"""
PhysioGraphSleep — Katman Katman Parametre Hesaplamaları
========================================================

Her modül, her katman, her ağırlık matrisinin boyutu ve parametre sayısı.
Sonuç: model.count_parameters() ile çapraz doğrulama yapılır.

Toplam beklenen: 687,946 parametre
"""

import sys

# ============================================================
# 1. WaveformStem
# ============================================================
print("=" * 70)
print("1. WaveformStem")
print("=" * 70)

# --- 1a. Initial Conv ---
# Conv1d(in_channels=1, out_channels=32, kernel_size=25, stride=2, padding=12, bias=False)
initial_conv = 1 * 32 * 25  # weight only, no bias
print(f"  Initial Conv1d(1→32, k=25, s=2, bias=False): {initial_conv}")

# GroupNorm(8, 32): gamma(32) + beta(32)
initial_gn = 32 + 32
print(f"  GroupNorm(8, 32):                             {initial_gn}")

initial_total = initial_conv + initial_gn
print(f"  Initial block subtotal:                       {initial_total}")
print()

# --- 1b. DSConvBranch ×3 (k=25, 75, 225) ---
# Her branch içinde:
#   DepthwiseSeparableConv ×2
#   SqueezeExcitation ×1

def calc_dsconv(channels, kernel_size):
    """Bir DepthwiseSeparableConv'un parametre sayısı."""
    # DWConv1d(ch, ch, k, groups=ch, bias=False): ch * 1 * k
    dw = channels * 1 * kernel_size
    # PWConv1d(ch, ch, 1, bias=False): ch * ch * 1
    pw = channels * channels * 1
    # GroupNorm(8, ch): gamma + beta
    gn = channels + channels
    total = dw + pw + gn
    return total, dw, pw, gn

def calc_se(channels, reduction):
    """SqueezeExcitation parametre sayısı."""
    mid = max(channels // reduction, 4)
    # Linear(ch, mid): weight + bias
    fc1 = channels * mid + mid
    # Linear(mid, ch): weight + bias
    fc2 = mid * channels + channels
    total = fc1 + fc2
    return total, mid, fc1, fc2

def calc_branch(channels, kernel_size, se_reduction):
    """Bir DSConvBranch parametre sayısı."""
    dsconv_total, dw, pw, gn = calc_dsconv(channels, kernel_size)
    se_total, mid, fc1, fc2 = calc_se(channels, se_reduction)
    # 2× DSConv + 1× SE
    branch_total = 2 * dsconv_total + se_total
    return branch_total, dsconv_total, se_total

channels = 32
se_reduction = 4
kernel_sizes = [25, 75, 225]
branch_totals = []

for k in kernel_sizes:
    branch_total, dsconv, se = calc_branch(channels, k, se_reduction)
    dsconv_detail, dw, pw, gn = calc_dsconv(channels, k)
    se_detail, mid, fc1, fc2 = calc_se(channels, se_reduction)
    
    print(f"  DSConvBranch(k={k}):")
    print(f"    DepthwiseSeparableConv ×2:")
    print(f"      DWConv1d({channels},{channels},k={k},groups={channels}): {dw}")
    print(f"      PWConv1d({channels},{channels},k=1):                   {pw}")
    print(f"      GroupNorm(8,{channels}):                              {gn}")
    print(f"      Per DSConv:                                     {dsconv_detail}")
    print(f"      ×2 DSConv:                                      {2 * dsconv_detail}")
    print(f"    SE(ch={channels}, red={se_reduction}, mid={mid}):")
    print(f"      Linear({channels}→{mid}): {fc1},  Linear({mid}→{channels}): {fc2}")
    print(f"      SE subtotal:                                    {se_detail}")
    print(f"    Branch total:                                     {branch_total}")
    print()
    branch_totals.append(branch_total)

all_branches = sum(branch_totals)
print(f"  Tüm branches toplam: {' + '.join(str(b) for b in branch_totals)} = {all_branches}")
print()

# --- 1c. Projection ---
# Conv1d(96, 96, 1, bias=False)
proj_conv = 96 * 96 * 1
proj_gn = 96 + 96
proj_total = proj_conv + proj_gn
print(f"  Projection Conv1d(96→96, k=1, bias=False): {proj_conv}")
print(f"  Projection GroupNorm(8, 96):                {proj_gn}")
print(f"  Projection subtotal:                        {proj_total}")
print()

# AdaptiveAvgPool1d(6) → 0 parametre
print(f"  AdaptiveAvgPool1d(6):                       0 (parametresiz)")
print()

waveform_total = initial_total + all_branches + proj_total
print(f"  *** WaveformStem TOPLAM: {initial_total} + {all_branches} + {proj_total} = {waveform_total}")
print()

# ============================================================
# 2. SpectralTokenEncoder
# ============================================================
print("=" * 70)
print("2. SpectralTokenEncoder")
print("=" * 70)

# BandMLP(42→64→96) × 5
in_f, hid, out_f = 42, 64, 96

linear1 = in_f * hid + hid       # Linear(42,64) weight+bias
ln = hid + hid                    # LayerNorm(64)
linear2 = hid * out_f + out_f    # Linear(64,96) weight+bias

per_mlp = linear1 + ln + linear2
print(f"  Per BandMLP(42→64→96):")
print(f"    Linear(42→64): weight={in_f}×{hid}={in_f*hid}, bias={hid} → {linear1}")
print(f"    LayerNorm(64): γ={hid} + β={hid} → {ln}")
print(f"    Linear(64→96): weight={hid}×{out_f}={hid*out_f}, bias={out_f} → {linear2}")
print(f"    Per MLP toplam: {linear1} + {ln} + {linear2} = {per_mlp}")
print()

spectral_total = 5 * per_mlp
print(f"  *** SpectralTokenEncoder TOPLAM: 5 × {per_mlp} = {spectral_total}")
print()

# ============================================================
# 3. HeteroGraphEncoder
# ============================================================
print("=" * 70)
print("3. HeteroGraphEncoder")
print("=" * 70)

dim = 96
num_heads = 4
head_dim = dim // num_heads  # 24
ff_mult = 2
num_layers = 2

# --- 3a. Summary token ---
summary_token = 1 * dim
print(f"  Summary token (learnable): 1 × {dim} = {summary_token}")
print()

# --- 3b. GraphTransformerBlock × 2 ---
# EdgeAwareAttention(dim=96, num_heads=4)
q_proj = dim * dim + dim   # Linear(96,96)
k_proj = dim * dim + dim
v_proj = dim * dim + dim
out_proj = dim * dim + dim
edge_bias = 4 * num_heads   # Embedding(4, 4)

attn_total = q_proj + k_proj + v_proj + out_proj + edge_bias
print(f"  Per GraphTransformerBlock (dim={dim}, heads={num_heads}):")
print(f"    EdgeAwareAttention:")
print(f"      Q_proj Linear({dim}→{dim}): {q_proj}")
print(f"      K_proj Linear({dim}→{dim}): {k_proj}")
print(f"      V_proj Linear({dim}→{dim}): {v_proj}")
print(f"      out_proj Linear({dim}→{dim}): {out_proj}")
print(f"      edge_bias Embedding(4, {num_heads}): {edge_bias}")
print(f"      Attention subtotal: {attn_total}")

# LayerNorm ×2
norm1 = dim + dim
norm2 = dim + dim
print(f"    norm1 LayerNorm({dim}): {norm1}")
print(f"    norm2 LayerNorm({dim}): {norm2}")

# FFN
ffn_l1 = dim * (dim * ff_mult) + (dim * ff_mult)  # Linear(96,192)
ffn_l2 = (dim * ff_mult) * dim + dim                # Linear(192,96)
ffn_total = ffn_l1 + ffn_l2
print(f"    FFN:")
print(f"      Linear({dim}→{dim*ff_mult}): {ffn_l1}")
print(f"      Linear({dim*ff_mult}→{dim}): {ffn_l2}")
print(f"      FFN subtotal: {ffn_total}")

per_block = attn_total + norm1 + norm2 + ffn_total
print(f"    Per block toplam: {attn_total} + {norm1} + {norm2} + {ffn_total} = {per_block}")
print()

all_blocks = num_layers * per_block
print(f"  {num_layers} blocks toplam: {num_layers} × {per_block} = {all_blocks}")
print()

# --- 3c. AttentiveReadout ---
readout_gate = dim * 1 + 1  # Linear(96,1)
print(f"  AttentiveReadout gate Linear({dim}→1): {readout_gate}")

# --- 3d. Projection ---
graph_out_dim = 128
proj_linear = (dim * 2) * graph_out_dim + graph_out_dim  # Linear(192,128)
proj_ln = graph_out_dim + graph_out_dim                    # LayerNorm(128)
proj_total_graph = proj_linear + proj_ln
print(f"  Projection Linear({dim*2}→{graph_out_dim}): {proj_linear}")
print(f"  Projection LayerNorm({graph_out_dim}): {proj_ln}")
print(f"  Projection subtotal: {proj_total_graph}")
print()

graph_total = summary_token + all_blocks + readout_gate + proj_total_graph
print(f"  *** HeteroGraphEncoder TOPLAM: {summary_token} + {all_blocks} + {readout_gate} + {proj_total_graph} = {graph_total}")
print()

# ============================================================
# 4. SequenceTransitionDecoder
# ============================================================
print("=" * 70)
print("4. SequenceTransitionDecoder")
print("=" * 70)

input_dim = 128
gru_hidden = 80
gru_layers = 2
gru_out = gru_hidden * 2  # 160, bidirectional
output_dim = 160

# --- 4a. BiGRU ---
# PyTorch GRU parametreleri per direction per layer:
#   weight_ih: (3*hidden, input_of_layer)
#   weight_hh: (3*hidden, hidden)
#   bias_ih:   (3*hidden)
#   bias_hh:   (3*hidden)

print(f"  BiGRU (input={input_dim}, hidden={gru_hidden}, layers={gru_layers}, bidirectional)")
print()

gru_total = 0

for layer in range(gru_layers):
    if layer == 0:
        layer_input = input_dim  # 128
    else:
        layer_input = gru_hidden * 2  # 160 (prev layer bidirectional output)
    
    for direction in ["forward", "reverse"]:
        w_ih = 3 * gru_hidden * layer_input
        w_hh = 3 * gru_hidden * gru_hidden
        b_ih = 3 * gru_hidden
        b_hh = 3 * gru_hidden
        dir_total = w_ih + w_hh + b_ih + b_hh
        
        print(f"    Layer {layer} {direction}:")
        print(f"      weight_ih ({3*gru_hidden}×{layer_input}): {w_ih}")
        print(f"      weight_hh ({3*gru_hidden}×{gru_hidden}): {w_hh}")
        print(f"      bias_ih ({3*gru_hidden}): {b_ih}")
        print(f"      bias_hh ({3*gru_hidden}): {b_hh}")
        print(f"      Direction toplam: {dir_total}")
        gru_total += dir_total

print(f"  GRU toplam: {gru_total}")
print()

# --- 4b. TemporalConvBlock ---
# Conv1d(160, 160, k=3, padding=1, bias=False)
tcn_conv = gru_out * gru_out * 3
tcn_ln = gru_out + gru_out  # LayerNorm(160)
tcn_total = tcn_conv + tcn_ln
print(f"  TemporalConvBlock:")
print(f"    Conv1d({gru_out}→{gru_out}, k=3, bias=False): {tcn_conv}")
print(f"    LayerNorm({gru_out}): {tcn_ln}")
print(f"    TCB toplam: {tcn_total}")
print()

# --- 4c. TransitionMemoryBlock ---
num_prototypes = 5
mha_heads = 4

# Prototypes
proto_params = num_prototypes * gru_out  # 5 × 160 = 800
print(f"  TransitionMemoryBlock:")
print(f"    Prototypes ({num_prototypes}×{gru_out}): {proto_params}")

# MultiheadAttention(embed_dim=160, num_heads=4)
# PyTorch MHA: in_proj_weight (3*embed, embed), in_proj_bias (3*embed), 
#              out_proj.weight (embed, embed), out_proj.bias (embed)
mha_in_proj_w = 3 * gru_out * gru_out   # 3×160×160
mha_in_proj_b = 3 * gru_out              # 3×160
mha_out_proj_w = gru_out * gru_out       # 160×160
mha_out_proj_b = gru_out                  # 160
mha_total = mha_in_proj_w + mha_in_proj_b + mha_out_proj_w + mha_out_proj_b

print(f"    MultiheadAttention(embed={gru_out}, heads={mha_heads}):")
print(f"      in_proj_weight (3×{gru_out}×{gru_out}): {mha_in_proj_w}")
print(f"      in_proj_bias (3×{gru_out}): {mha_in_proj_b}")
print(f"      out_proj.weight ({gru_out}×{gru_out}): {mha_out_proj_w}")
print(f"      out_proj.bias ({gru_out}): {mha_out_proj_b}")
print(f"      MHA toplam: {mha_total}")

tmb_ln = gru_out + gru_out  # LayerNorm(160)
print(f"    LayerNorm({gru_out}): {tmb_ln}")

tmb_total = proto_params + mha_total + tmb_ln
print(f"    TMB toplam: {tmb_total}")
print()

# --- 4d. Final Projection ---
proj_lin = gru_out * output_dim + output_dim  # Linear(160, 160)
proj_decoder_total = proj_lin
print(f"  Projection Linear({gru_out}→{output_dim}): {proj_lin}")
print()

decoder_total = gru_total + tcn_total + tmb_total + proj_decoder_total
print(f"  *** SequenceTransitionDecoder TOPLAM: {gru_total} + {tcn_total} + {tmb_total} + {proj_lin} = {decoder_total}")
print()

# ============================================================
# 5. MultiTaskHeads
# ============================================================
print("=" * 70)
print("5. MultiTaskHeads")
print("=" * 70)

head_input = 160
head_dropout = 0.3

def calc_head(in_dim, num_classes):
    """ClassificationHead: LayerNorm + Dropout + Linear."""
    ln = in_dim + in_dim             # LayerNorm
    linear = in_dim * num_classes + num_classes  # Linear
    return ln + linear, ln, linear

heads_info = [
    ("stage_head", 5),
    ("boundary_head", 1),
    ("prev_head", 5),
    ("next_head", 5),
    ("n1_head", 1),
]

heads_total = 0
for name, nc in heads_info:
    total, ln, linear = calc_head(head_input, nc)
    print(f"  {name}(in={head_input}, out={nc}):")
    print(f"    LayerNorm({head_input}): {ln}")
    print(f"    Linear({head_input}→{nc}): {linear}")
    print(f"    Head toplam: {total}")
    heads_total += total

print()
print(f"  *** MultiTaskHeads TOPLAM: {heads_total}")
print()

# ============================================================
# GRAND TOTAL
# ============================================================
print("=" * 70)
print("GRAND TOTAL")
print("=" * 70)

grand_total = waveform_total + spectral_total + graph_total + decoder_total + heads_total

print(f"  WaveformStem:              {waveform_total:>10,}")
print(f"  SpectralTokenEncoder:      {spectral_total:>10,}")
print(f"  HeteroGraphEncoder:        {graph_total:>10,}")
print(f"  SequenceTransitionDecoder: {decoder_total:>10,}")
print(f"  MultiTaskHeads:            {heads_total:>10,}")
print(f"  {'─' * 40}")
print(f"  TOPLAM (hesaplanan):       {grand_total:>10,}")
print()

expected = 687_946
if grand_total == expected:
    print(f"  ✅ DOĞRU! Hesaplanan ({grand_total:,}) == Beklenen ({expected:,})")
else:
    print(f"  ❌ UYUMSUZLUK! Hesaplanan ({grand_total:,}) != Beklenen ({expected:,})")
    print(f"     Fark: {grand_total - expected:+,}")

print()
print("=" * 70)
print("Model ile doğrulama (PyTorch gerektirir)")
print("=" * 70)

try:
    sys.path.insert(0, ".")
    import torch
    from physiographsleep.configs import ExperimentConfig
    from physiographsleep.models.physiographsleep import PhysioGraphSleep

    config = ExperimentConfig()
    model = PhysioGraphSleep(config.model)
    actual_counts = model.count_parameters()

    print(f"\n  model.count_parameters():")
    for name, count in actual_counts.items():
        print(f"    {name:30s}: {count:>10,}")

    # Detaylı karşılaştırma
    expected_map = {
        "waveform_stem": waveform_total,
        "spectral_encoder": spectral_total,
        "graph_encoder": graph_total,
        "sequence_decoder": decoder_total,
        "heads": heads_total,
        "total": grand_total,
    }

    print(f"\n  Karşılaştırma:")
    all_match = True
    for name in expected_map:
        calc = expected_map[name]
        actual = actual_counts[name]
        match = "✅" if calc == actual else "❌"
        if calc != actual:
            all_match = False
        print(f"    {name:30s}: hesaplanan={calc:>10,} | gerçek={actual:>10,} {match}")

    if all_match:
        print(f"\n  ✅ TÜM MODÜLLER EŞLEŞTI!")
    else:
        print(f"\n  ❌ UYUMSUZLUK TESPİT EDİLDİ!")

except ImportError as e:
    print(f"\n  [SKIP] PyTorch bulunamadı: {e}")
    print(f"  Manuel hesaplama sonucu: {grand_total:,} parametre")
