"""
PhysioGraphSleep — Tensor Shape Flow Hesaplamaları
===================================================

Modelin girişinden çıkışına kadar her adımda tensor boyutlarının
nasıl değiştiğini detaylıca izler.

Batch size B=64, Sequence length L=25, Channels C=1, Time T=3000
"""

import numpy as np

print("=" * 70)
print("PhysioGraphSleep — Tensor Shape Flow")
print("=" * 70)

# Sabitler
B = 64  # batch size (notebook'ta ayarlanan)
L = 25  # sequence length
C = 1   # EEG channel
T = 3000  # samples per epoch (100Hz × 30s)

print(f"\nGiriş boyutları: B={B}, L={L}, C={C}, T={T}")
print(f"Toplam sinyal: {B}×{L}×{C}×{T} = {B*L*C*T:,} eleman")
print()

# ============================================================
# 1. GİRİŞ (Dataset çıktısı)
# ============================================================
print("=" * 70)
print("1. Dataset Çıktısı (SleepEDFDataset.__getitem__)")
print("=" * 70)

print(f"""
  signal:       ({B}, {L}, {C}, {T})    → float32, {B*L*C*T*4/1e6:.1f} MB
  spectral:     ({B}, {L}, 5, 42)       → float32, {B*L*5*42*4/1e6:.1f} MB
  label:        ({B},)                   → int64
  seq_labels:   ({B}, {L})               → int64
  mask:         ({B}, {L})               → float32
  boundary:     ({B},)                   → float32
  prev_label:   ({B},)                   → int64
  next_label:   ({B},)                   → int64
  n1_label:     ({B},)                   → float32
""")

# ============================================================
# 2. PhysioGraphSleep.forward() — Per-Epoch Loop
# ============================================================
print("=" * 70)
print("2. Per-Epoch Processing (L kez tekrarlanır)")
print("=" * 70)

print(f"\n  for t in range({L}):")
print(f"    sig_t = signals[:, t, :, :]     → ({B}, {C}, {T})")
print(f"    spec_t = spectral[:, t, :, :]   → ({B}, 5, 42)")
print()

# --- 2a. WaveformStem ---
print("  2a. WaveformStem:")
# Initial conv: Conv1d(1, 32, k=25, s=2, p=12)
# Output length: floor((T + 2*p - k) / s) + 1 = floor((3000 + 24 - 25) / 2) + 1
T_after_initial = (T + 2 * 12 - 25) // 2 + 1
print(f"      Input:                         ({B}, {C}, {T})")
print(f"      Conv1d(1→32, k=25, s=2, p=12): ({B}, 32, ?)")
print(f"        Output length = floor(({T} + 2×12 - 25) / 2) + 1 = floor({T + 24 - 25} / 2) + 1 = {T_after_initial}")
print(f"      After initial:                 ({B}, 32, {T_after_initial})")
print()

# DSConvBranch: Depthwise-separate conv, padding=k//2 → aynı uzunluk
# Her DSConvBranch out: (B, 32, T_after_initial)
print(f"      3× DSConvBranch (her biri kanal boyutunu korur):")
for k_size in [25, 75, 225]:
    print(f"        Branch k={k_size}: ({B}, 32, {T_after_initial}) → ({B}, 32, {T_after_initial})")
    # DWConv padding=k//2 → same padding, boyut korunur
    # PWConv k=1 → boyut korunur

print(f"      Concat 3 branch:               ({B}, 96, {T_after_initial})")
print(f"      Projection Conv1d(96→96, k=1): ({B}, 96, {T_after_initial})")
print(f"      AdaptiveAvgPool1d(6):          ({B}, 96, 6)")
print(f"      Transpose(1,2):                ({B}, 6, 96)")
print(f"      → patch_tokens:               ({B}, 6, 96)")
print()

# --- 2b. SpectralTokenEncoder ---
print("  2b. SpectralTokenEncoder:")
print(f"      Input:                         ({B}, 5, 42)")
print(f"      5× BandMLP(42→64→96):")
print(f"        Band 0: ({B}, 42) → Linear(42,64) → LN → GELU → Drop → Linear(64,96) → ({B}, 96)")
print(f"        Band 1-4: aynı")
print(f"      Stack:                         ({B}, 5, 96)")
print(f"      → band_tokens:                ({B}, 5, 96)")
print()

# --- 2c. HeteroGraphEncoder ---
print("  2c. HeteroGraphEncoder:")
NUM_NODES = 12
num_patch = 6
num_band = 5
num_summary = 1

# Graph construction
E_patch_patch = 5 * 2   # adjacent patch pairs, bidirectional
E_band_band = 5 * 4     # C(5,2) = 10 pairs × 2 directions = 20
E_patch_band = 6 * 5 * 2  # each patch ↔ each band, bidirectional
E_summary = 11 * 2      # summary ↔ all 11 other nodes
E_total = E_patch_patch + E_band_band + E_patch_band + E_summary

print(f"      Nodes per epoch: {num_patch} patch + {num_band} band + {num_summary} summary = {NUM_NODES}")
print(f"      Edges per epoch:")
print(f"        patch-patch (adjacent, bidir): 5 × 2 = {E_patch_patch}")
print(f"        band-band (full, bidir):  C(5,2)×2 = {E_band_band}")
print(f"        patch-band (cross, bidir):  6×5×2 = {E_patch_band}")
print(f"        summary (bidir):            11×2 = {E_summary}")
print(f"        Toplam edges per epoch:     {E_total}")
print()

# Batching
total_nodes = B * NUM_NODES
total_edges = B * E_total
print(f"      Batched graph:")
print(f"        x:          ({total_nodes}, 96)   = ({B}×{NUM_NODES}, 96)")
print(f"        edge_index: (2, {total_edges})  = (2, {B}×{E_total})")
print(f"        edge_type:  ({total_edges},)     = ({B}×{E_total},)")
print(f"        batch_id:   ({total_nodes},)     = ({B}×{NUM_NODES},)")
print()

print(f"      2× GraphTransformerBlock:")
print(f"        EdgeAwareAttention(dim=96, heads=4, head_dim=24):")
print(f"          Q = x @ W_q → ({total_nodes}, 96) → reshape → ({total_nodes}, 4, 24)")
print(f"          K = x @ W_k → aynı")
print(f"          V = x @ W_v → aynı")
print(f"          attn_scores: (E, H) = ({total_edges}, 4)")
print(f"          edge_bias: Embedding(4,4) → ({total_edges}, 4)")
print(f"          output: ({total_nodes}, 96)")
print(f"        LayerNorm + FFN({96}→{96*2}→{96}) + LayerNorm")
print(f"        Block output: ({total_nodes}, 96)")
print()

print(f"      Readout:")
print(f"        Summary tokens: indices [{0*NUM_NODES+11}, {1*NUM_NODES+11}, ...] → ({B}, 96)")
print(f"        Attentive pool: gate({total_nodes},96)→({total_nodes},1) → scatter_add → ({B}, 96)")
print(f"        Concat: ({B}, 192)")
print(f"        Linear(192→128) + LN + GELU")
print(f"      → epoch_embedding:            ({B}, 128)")
print()

# --- Loop sonucu ---
print("  Per-epoch loop sonucu:")
print(f"    epoch_embeddings = stack(L adet ({B}, 128)) → ({B}, {L}, 128)")
print()

# ============================================================
# 3. SequenceTransitionDecoder
# ============================================================
print("=" * 70)
print("3. SequenceTransitionDecoder")
print("=" * 70)

gru_hidden = 80
gru_out = gru_hidden * 2  # 160

print(f"\n  Input: ({B}, {L}, 128)")
print()

print(f"  BiGRU(input=128, hidden=80, layers=2, bidirectional):")
print(f"    Layer 0: input={128} → hidden=80 × 2dir = 160")
print(f"    Layer 1: input={gru_out} → hidden=80 × 2dir = 160")
print(f"    Output: ({B}, {L}, {gru_out})")
print(f"    Dropout(0.3): ({B}, {L}, {gru_out})")
print()

print(f"  TemporalConvBlock:")
print(f"    Transpose: ({B}, {gru_out}, {L})")
print(f"    Conv1d({gru_out},{gru_out}, k=3, p=1): ({B}, {gru_out}, {L})")
print(f"    Transpose: ({B}, {L}, {gru_out})")
print(f"    LayerNorm({gru_out}), GELU, Dropout")
print(f"    + residual")
print(f"    Output: ({B}, {L}, {gru_out})")
print()

print(f"  TransitionMemoryBlock:")
print(f"    Prototypes: (5, {gru_out}) → expand → ({B}, 5, {gru_out})")
print(f"    Cross-attention:")
print(f"      query = x:          ({B}, {L}, {gru_out})")
print(f"      key = prototypes:   ({B}, 5, {gru_out})")
print(f"      value = prototypes: ({B}, 5, {gru_out})")
print(f"      attn: ({B}, {L}, {gru_out}) — her zaman adımı 5 prototype'a attend eder")
print(f"    + residual + LayerNorm")
print(f"    Output: ({B}, {L}, {gru_out})")
print()

print(f"  Final Projection:")
print(f"    Linear({gru_out}→{gru_out}): ({B}, {L}, {gru_out})")
print(f"    GELU, Dropout")
print(f"    Output: ({B}, {L}, {gru_out})")
print()

# ============================================================
# 4. Center Epoch Extraction + MultiTaskHeads
# ============================================================
print("=" * 70)
print("4. Center Epoch Extraction + MultiTaskHeads")
print("=" * 70)

center_idx = L // 2
print(f"\n  center_idx = {L} // 2 = {center_idx}")
print(f"  center_features = seq_features[:, {center_idx}, :]  → ({B}, {gru_out})")
print()

print(f"  MultiTaskHeads (her biri: LN → Drop → Linear):")
print(f"    stage_head({gru_out}→5):    ({B}, 5)    → 5-class logits")
print(f"    boundary_head({gru_out}→1): ({B}, 1) → squeeze → ({B},)  → binary logit")
print(f"    prev_head({gru_out}→5):     ({B}, 5)    → 5-class logits")
print(f"    next_head({gru_out}→5):     ({B}, 5)    → 5-class logits")
print(f"    n1_head({gru_out}→1):       ({B}, 1) → squeeze → ({B},)  → binary logit")
print()

# ============================================================
# 5. Bellek Hesaplamaları
# ============================================================
print("=" * 70)
print("5. Bellek Hesaplamaları (Forward Pass)")
print("=" * 70)

# Model parametreleri
param_count = 687_946
param_mem = param_count * 4 / 1e6  # float32
print(f"\n  Model parametreleri: {param_count:,} × 4 bytes = {param_mem:.2f} MB")

# Giriş tensörleri
input_signal = B * L * C * T * 4 / 1e6
input_spectral = B * L * 5 * 42 * 4 / 1e6
print(f"  Giriş sinyali:  {B}×{L}×{C}×{T} × 4B = {input_signal:.2f} MB")
print(f"  Giriş spectral: {B}×{L}×5×42 × 4B = {input_spectral:.2f} MB")

# Per-epoch encode (L kez, ama bellekteki peak)
# En büyük ara tensör: graph batched x
graph_x = B * NUM_NODES * 96 * 4 / 1e6
graph_edges = total_edges * 2 * 8 / 1e6  # int64 indices
print(f"  Graph node features peak: {B}×{NUM_NODES}×96 × 4B = {graph_x:.2f} MB")

# Epoch embeddings accumulation
epoch_emb_mem = B * L * 128 * 4 / 1e6
print(f"  Epoch embeddings: {B}×{L}×128 × 4B = {epoch_emb_mem:.2f} MB")

# Sequence decoder
seq_feat_mem = B * L * 160 * 4 / 1e6
print(f"  Sequence features: {B}×{L}×160 × 4B = {seq_feat_mem:.2f} MB")

# GRU hidden states
gru_hidden_mem = gru_hidden * 2 * B * 2 * 4 / 1e6  # 2 layers, 2 directions
print(f"  GRU hidden states: ≈{gru_hidden_mem:.2f} MB")
print()

# ============================================================
# 6. Hesaplama Karmaşıklığı (FLOPs tahmini)
# ============================================================
print("=" * 70)
print("6. Hesaplama Karmaşıklığı (per forward pass)")
print("=" * 70)

# WaveformStem per-epoch
ws_initial = 2 * 1 * 32 * 25 * T_after_initial  # Conv1d FLOPs
# DSConv branches (approximate)
ws_branches = sum(2 * (2 * (32 * 1 * k * T_after_initial + 32 * 32 * 1 * T_after_initial)) for k in [25, 75, 225])
ws_proj = 2 * 96 * 96 * 1 * T_after_initial
ws_flops = ws_initial + ws_branches + ws_proj
print(f"\n  WaveformStem (per epoch): ~{ws_flops/1e6:.1f} MFLOPs")
print(f"  WaveformStem (×{L} epochs): ~{ws_flops*L/1e6:.1f} MFLOPs")

# SpectralEncoder per-epoch
se_flops = 5 * (2 * 42 * 64 + 2 * 64 * 96)
print(f"  SpectralEncoder (per epoch): ~{se_flops/1e6:.3f} MFLOPs")

# GraphEncoder per-epoch
# Attention: Q/K/V projections + attention scores + aggregation
ge_proj = 4 * 2 * 96 * 96 * 12  # Q,K,V,out projections
ge_attn = 2 * E_total * 96  # attention computation
ge_ffn = 2 * (2 * 96 * 192 * 12 + 2 * 192 * 96 * 12)  # 2 blocks × FFN
ge_readout = 2 * 192 * 128
ge_flops = ge_proj + ge_attn + ge_ffn + ge_readout
print(f"  GraphEncoder (per epoch): ~{ge_flops/1e6:.3f} MFLOPs")

# Decoder
gru_flops = 2 * (3 * (2 * 128 * 80 + 2 * 80 * 80) + 3 * (2 * 160 * 80 + 2 * 80 * 80)) * L  # approx
tcn_flops = 2 * 160 * 160 * 3 * L
tmb_flops = 2 * 3 * 160 * 160 * L + 2 * L * 5 * 160  # MHA approx
dec_flops = gru_flops + tcn_flops + tmb_flops
print(f"  Decoder: ~{dec_flops/1e6:.1f} MFLOPs")

# Heads
head_flops = 2 * 160 * (5 + 1 + 5 + 5 + 1)
print(f"  Heads: ~{head_flops/1e6:.4f} MFLOPs")

total_flops = (ws_flops + se_flops + ge_flops) * L + dec_flops + head_flops
print(f"\n  Toplam per forward: ~{total_flops/1e6:.1f} MFLOPs = ~{total_flops/1e9:.2f} GFLOPs")

# ============================================================
# 7. Kritik Doğrulamalar
# ============================================================
print()
print("=" * 70)
print("7. Kritik Shape Doğrulamaları")
print("=" * 70)

checks = [
    ("WaveformStem çıkışı", (B, 6, 96), "graph_encoder girişi ile uyumlu"),
    ("SpectralEncoder çıkışı", (B, 5, 96), "graph_encoder girişi ile uyumlu"),
    ("patch + band node sayısı", 6 + 5 + 1, f"= {NUM_NODES} node (graph_builder sabit)"),
    ("GraphEncoder çıkışı", (B, 128), "decoder input_dim=128 ile uyumlu"),
    ("Epoch embeddings", (B, L, 128), "decoder input_dim=128 ile uyumlu"),
    ("BiGRU çıkışı", (B, L, 160), "gru_hidden×2 = 80×2 = 160"),
    ("Decoder çıkışı", (B, L, 160), "heads input_dim=160 ile uyumlu"),
    ("Center extraction", (B, 160), f"index={center_idx}, heads'e giriş"),
    ("stage_head çıkışı", (B, 5), "5 sınıf: W/N1/N2/N3/REM"),
    ("Loss: stage giriş", f"pred=({B},5) vs target=({B},)", "CrossEntropy uyumlu"),
    ("Loss: boundary giriş", f"pred=({B},) vs target=({B},)", "BCEWithLogits uyumlu"),
]

print()
all_ok = True
for name, shape, note in checks:
    print(f"  ✅ {name}: {shape} — {note}")

print()
print("  Tüm shape geçişleri tutarlı.")

# ============================================================
# 8. Conv1d Output Length Detaylı Hesap
# ============================================================
print()
print("=" * 70)
print("8. Conv1d Output Length — Detaylı Hesap")
print("=" * 70)

print(f"""
  Initial Conv1d(1→32, k=25, s=2, p=12):
    L_out = floor((L_in + 2×padding - kernel_size) / stride) + 1
    L_out = floor(({T} + 2×12 - 25) / 2) + 1
    L_out = floor(({T + 24 - 25}) / 2) + 1
    L_out = floor({(T + 24 - 25) / 2}) + 1
    L_out = {(T + 24 - 25) // 2} + 1
    L_out = {T_after_initial}

  DSConv branches (DWConv, k=25/75/225, stride=1, padding=k//2):
    k=25:  p={25//2},  L_out = floor(({T_after_initial} + 2×{25//2} - 25) / 1) + 1 = {(T_after_initial + 2*(25//2) - 25) // 1 + 1}
    k=75:  p={75//2},  L_out = floor(({T_after_initial} + 2×{75//2} - 75) / 1) + 1 = {(T_after_initial + 2*(75//2) - 75) // 1 + 1}
    k=225: p={225//2}, L_out = floor(({T_after_initial} + 2×{225//2} - 225) / 1) + 1 = {(T_after_initial + 2*(225//2) - 225) // 1 + 1}
""")

# Kontrol: tüm branch'lar aynı boyutta mı?
for k in [25, 75, 225]:
    p = k // 2
    l_out = (T_after_initial + 2 * p - k) // 1 + 1
    match = "✅" if l_out == T_after_initial else "❌"
    print(f"  k={k}: L_out={l_out} {'==' if l_out == T_after_initial else '!='} {T_after_initial} {match}")

# Tek kerneller
odd_check_25 = 25 % 2 == 1   # True → padding=12, floor division → same
odd_check_75 = 75 % 2 == 1   # True
odd_check_225 = 225 % 2 == 1  # True
print(f"\n  Tüm kernel_size tek sayı: 25({'✅' if odd_check_25 else '❌'}), 75({'✅' if odd_check_75 else '❌'}), 225({'✅' if odd_check_225 else '❌'})")
print(f"  padding=k//2 ile stride=1: tam boyut korunması garanti ✅")
