"""
PhysioGraphSleep — Loss ve Training Hesaplamaları
==================================================

Focal loss, class weights, adaptive loss, sampler weights,
curriculum stage parametreleri, scheduler hesaplamaları.
"""

import numpy as np

print("=" * 70)
print("PhysioGraphSleep — Loss & Training Hesaplamaları")
print("=" * 70)

# ============================================================
# 1. Class Distribution & Weights
# ============================================================
print("\n" + "=" * 70)
print("1. Sınıf Dağılımı ve Ağırlıkları")
print("=" * 70)

# Sleep-EDF-20 yaklaşık dağılım (14 subject, ~74K training epoch)
# Gerçek veriden alınacak, ama tahmini:
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
# Approximate counts from v1 training
approx_counts = np.array([50200, 1800, 12300, 4400, 5300], dtype=np.float32)
total = approx_counts.sum()
print(f"\n  Tahmini training epoch sayıları (14 subject):")
for i, name in enumerate(STAGE_NAMES):
    print(f"    {name}: {int(approx_counts[i]):>6d} ({100*approx_counts[i]/total:.1f}%)")
print(f"    Toplam: {int(total)}")
print()

# Inverse-frequency class weights (notebook'taki formül)
class_weights = 1.0 / (approx_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * 5  # normalize to sum=5

print("  Inverse-frequency class weights (notebook formülü):")
print(f"    w = 1.0 / (counts + 1e-6)")
print(f"    w = w / sum(w) × 5")
print(f"    Sonuç: {[f'{w:.4f}' for w in class_weights]}")
print()

# Ağırlık oranları
max_w = class_weights.max()
min_w = class_weights.min()
print(f"    Maks/Min ağırlık oranı: {max_w:.4f}/{min_w:.4f} = {max_w/min_w:.1f}×")
print(f"    N1 ağırlığı / W ağırlığı: {class_weights[1]:.4f}/{class_weights[0]:.4f} = {class_weights[1]/class_weights[0]:.1f}×")
print()

# ============================================================
# 2. Focal Loss Hesaplaması
# ============================================================
print("=" * 70)
print("2. Focal Loss Hesaplaması")
print("=" * 70)

gamma = 2.0
smoothing = 0.05
print(f"\n  FocalLoss(gamma={gamma}, label_smoothing={smoothing})")
print()

# Örnek: doğru sınıf olasılığı p_t değişkenine göre
print("  Focal modulation factor: (1 - p_t)^gamma × CE")
print(f"  gamma={gamma} ile:")
print(f"   p_t=0.1 → (1-0.1)^{gamma} = {(1-0.1)**gamma:.4f} → büyük ağırlık (yanlış tahmin)")
print(f"   p_t=0.3 → (1-0.3)^{gamma} = {(1-0.3)**gamma:.4f}")
print(f"   p_t=0.5 → (1-0.5)^{gamma} = {(1-0.5)**gamma:.4f}")
print(f"   p_t=0.7 → (1-0.7)^{gamma} = {(1-0.7)**gamma:.4f}")
print(f"   p_t=0.9 → (1-0.9)^{gamma} = {(1-0.9)**gamma:.4f} → düşük ağırlık (kolay örnek)")
print(f"   p_t=0.95→ (1-0.95)^{gamma} = {(1-0.95)**gamma:.6f}")
print(f"   p_t=0.99→ (1-0.99)^{gamma} = {(1-0.99)**gamma:.8f}")
print()

# Label smoothing etkisi
num_classes = 5
smooth_target = smoothing / (num_classes - 1)
smooth_correct = 1.0 - smoothing + smoothing / num_classes

# Wait, PyTorch label_smoothing in cross_entropy is:
# y_smooth[correct] = 1 - smoothing + smoothing/num_classes
# y_smooth[wrong]   = smoothing / num_classes
# Veya daha doğrusu:
# CE = (1-smoothing) × CE(one_hot) + smoothing × H(uniform)
# = (1-smoothing) × (-log p_correct) + smoothing × (-1/K × sum log p_i)

true_target = 1.0 - smoothing  # Wrong: should be more nuanced
# Actually PyTorch formula:
# loss = (1 - label_smoothing) * ce_original + label_smoothing * uniform_ce
print(f"  Label smoothing={smoothing}:")
print(f"    PyTorch formülü: loss = (1-ε)×CE_orijinal + ε×CE_uniform")
print(f"    Burada ε={smoothing}, sınıf sayısı K={num_classes}")
print(f"    Etkisi: modelin aşırı güvenli (p→1.0) olmasını engeller")
print(f"    Regularization etkisi: özellikle W sınıfı için (çoğunluk)")
print()

# ============================================================
# 3. Multi-Task Loss Ağırlıkları
# ============================================================
print("=" * 70)
print("3. Multi-Task Loss Ağırlıkları")
print("=" * 70)

loss_weights = {
    "stage": 1.0,
    "boundary": 0.35,
    "prev": 0.20,
    "next": 0.20,
    "n1": 0.30,
}

total_weight = sum(loss_weights.values())
print(f"\n  L_total = Σ w_i × L_i")
for name, w in loss_weights.items():
    print(f"    {name:12s}: w = {w:.2f} ({100*w/total_weight:.1f}%)")
print(f"  Toplam ağırlık: {total_weight:.2f}")
print()

print("  Her task'ın göreceli katkısı:")
for name, w in loss_weights.items():
    print(f"    {name:12s}: {100*w/total_weight:.1f}%")
print()

# Stage sınıflandırma baskınlık kontrolü
print(f"  Ana task (stage) oranı: {100*loss_weights['stage']/total_weight:.1f}%")
print(f"  Yardımcı taskler toplamı: {100*(total_weight-loss_weights['stage'])/total_weight:.1f}%")
print(f"  → Ana task baskın ama yardımcılar ihmal edilmemiş ✅")
print()

# ============================================================
# 4. Adaptive Loss (Stage C)
# ============================================================
print("=" * 70)
print("4. Adaptive Loss Hesaplamaları (Stage C)")
print("=" * 70)

K = 10.0
gamma_adaptive = 1.0
warmup = 5
eps = 1e-4

print(f"\n  Formül: W_i = K / (F1_i + ε)^γ, normalize → sum = num_classes")
print(f"  K={K}, γ={gamma_adaptive}, warmup={warmup} epoch, ε={eps}")
print()

# Senaryo 1: İlk epoch sonrası tipik F1
scenarios = [
    ("Erken eğitim (epoch ~5)", [0.80, 0.05, 0.60, 0.40, 0.50]),
    ("Orta eğitim (epoch ~15)", [0.95, 0.30, 0.85, 0.70, 0.80]),
    ("İyi eğitim (epoch ~25)", [0.98, 0.45, 0.88, 0.78, 0.85]),
    ("v1 son durum", [0.983, 0.406, 0.877, 0.780, 0.850]),
    ("Hedef", [0.98, 0.55, 0.90, 0.82, 0.88]),
    ("Edge case: N1 çok düşük", [0.95, 0.10, 0.85, 0.70, 0.80]),
    ("Edge case: N1 sıfıra yakın", [0.90, 0.01, 0.80, 0.60, 0.70]),
]

for scenario_name, f1_values in scenarios:
    f1 = np.array(f1_values)
    f1_clipped = np.clip(f1, eps, None)
    raw = K / np.power(f1_clipped, gamma_adaptive)
    normalized = raw / raw.sum() * len(raw)  # sum = 5
    
    print(f"  {scenario_name}:")
    print(f"    F1:      {[f'{v:.3f}' for v in f1]}")
    print(f"    Weights: {[f'{v:.3f}' for v in normalized]}")
    
    # N1/W oranı
    n1_w_ratio = normalized[1] / normalized[0]
    print(f"    N1/W oranı: {n1_w_ratio:.2f}×")
    
    # Dengelilik kontrolü
    min_w = normalized.min()
    max_w = normalized.max()
    print(f"    Min/Max ağırlık: {min_w:.3f}/{max_w:.3f} (oran: {max_w/min_w:.1f}×)")
    
    # Tehlike kontrolü
    if max_w / min_w > 10:
        print(f"    ⚠️ DENGESİZ! Maks/min oranı > 10")
    elif max_w / min_w > 5:
        print(f"    ⚠️ Dikkat: Maks/min oranı > 5")
    else:
        print(f"    ✅ Dengeli dağılım")
    print()

# gamma karşılaştırma
print("  " + "-" * 60)
print("  Gamma karşılaştırması (v1 son durum F1 ile):")
f1_v1 = np.array([0.983, 0.406, 0.877, 0.780, 0.850])
f1_edge = np.array([0.95, 0.10, 0.85, 0.70, 0.80])

for g in [1.0, 1.5, 2.0, 3.0]:
    raw_v1 = K / np.power(np.clip(f1_v1, eps, None), g)
    norm_v1 = raw_v1 / raw_v1.sum() * 5
    raw_e = K / np.power(np.clip(f1_edge, eps, None), g)
    norm_e = raw_e / raw_e.sum() * 5
    
    print(f"\n    gamma={g}:")
    print(f"      v1 final:  {[f'{v:.3f}' for v in norm_v1]} (max/min={norm_v1.max()/norm_v1.min():.1f}×)")
    print(f"      edge case: {[f'{v:.3f}' for v in norm_e]} (max/min={norm_e.max()/norm_e.min():.1f}×)")
    
    if g == gamma_adaptive:
        print(f"      ← KULLANILAN DEĞER ✅")

print()

# ============================================================
# 5. Sampler Weights (Stage-specific)
# ============================================================
print("=" * 70)
print("5. Sampler Ağırlıkları (Curriculum stage'lere göre)")
print("=" * 70)

def compute_sampler_weights(counts, n1_boost=None):
    """build_weighted_sampler mantığı."""
    cw = 1.0 / (counts + 1e-6)
    if n1_boost is not None and n1_boost > 1.0:
        cw[1] *= n1_boost
    cw = cw / cw.sum()
    return cw

stages = [
    ("Stage A", 2.0),
    ("Stage B", None),
    ("Stage C", 1.5),
]

for stage_name, boost in stages:
    if boost is None:
        print(f"\n  {stage_name}: natural distribution (shuffle=True, no sampler)")
        # Her sınıfın gerçek oranı
        probs = approx_counts / total
        print(f"    Gerçek oranlar: {[f'{p:.4f}' for p in probs]}")
        print(f"    N1 oranı: {probs[1]*100:.2f}% (çok düşük ama decoder eğitimi için teorik)")
    else:
        weights = compute_sampler_weights(approx_counts, boost)
        
        # Her sınıfın örneklenme olasılığı
        # sample_weights[i] = class_weight[label[i]]
        # Her epoch'un seçilme olasılığı ∝ sample_weight
        # Beklenen sınıf oranları: class_weight * class_count / sum(all sample weights)
        expected_ratio = weights * approx_counts
        expected_ratio = expected_ratio / expected_ratio.sum()
        
        print(f"\n  {stage_name}: n1_boost={boost}")
        print(f"    Class weights:       {[f'{w:.6f}' for w in weights]}")
        print(f"    Beklenen batch oranı: {[f'{r:.4f}' for r in expected_ratio]}")
        print(f"    N1 beklenen oranı: {expected_ratio[1]*100:.1f}%")
        
        # 1/5 = 20% olursa mükemmel balanced
        print(f"    Mükemmel denge: 20% per class")
        
        n1_actual = approx_counts[1] / total * 100
        print(f"    N1 gerçek: {n1_actual:.1f}% → N1 beklenen: {expected_ratio[1]*100:.1f}% ({expected_ratio[1]*100/n1_actual:.1f}× boost)")
print()

# ============================================================
# 6. Optimizer & Scheduler Hesaplamaları
# ============================================================
print("=" * 70)
print("6. Optimizer & Scheduler Hesaplamaları")
print("=" * 70)

# Curriculum stage yapılandırması
print(f"""
  Stage A (Encoder pretraining):
    Epochs: 30
    LR: 1e-3
    Frozen: decoder, boundary_head, prev_head, next_head
    Trainable: waveform_stem, spectral_encoder, graph_encoder, stage_head, n1_head

  Stage B (Decoder training):
    Epochs: 25
    LR: 1e-4
    Frozen: waveform_stem, spectral_encoder, graph_encoder
    Trainable: sequence_decoder, heads (all)

  Stage C (End-to-end fine-tuning):
    Epochs: 25
    LR: 1e-4
    Frozen: nothing
    Trainable: all modules
    Adaptive loss: warmup={warmup} epoch sonra aktif
""")

# Trainable parametre sayıları per stage
waveform = 39256
spectral = 45600
graph = 174753
decoder = 424000
heads = 4337

# Stage A trainable
# Frozen: decoder, boundary_head, prev_head, next_head
# boundary_head: LN(160) + Linear(160,1) = 320 + 161 = 481
# prev_head: LN(160) + Linear(160,5) = 320 + 805 = 1125
# next_head: same = 1125
frozen_a = decoder + 481 + 1125 + 1125  # = 426731
trainable_a = waveform + spectral + graph + (heads - 481 - 1125 - 1125)
total_params = waveform + spectral + graph + decoder + heads

print(f"  Stage A trainable parametreler:")
print(f"    waveform_stem:       {waveform:>10,}")
print(f"    spectral_encoder:    {spectral:>10,}")
print(f"    graph_encoder:       {graph:>10,}")
print(f"    stage_head:          {1125:>10,}")
print(f"    n1_head:             {481:>10,}")
print(f"    Trainable:           {trainable_a:>10,}")
print(f"    Frozen:              {total_params - trainable_a:>10,}")
print()

# Stage B trainable
trainable_b = decoder + heads  # all heads
frozen_b = waveform + spectral + graph
print(f"  Stage B trainable parametreler:")
print(f"    sequence_decoder:    {decoder:>10,}")
print(f"    heads (all):         {heads:>10,}")
print(f"    Trainable:           {trainable_b:>10,}")
print(f"    Frozen:              {frozen_b:>10,}")
print()

# Stage C trainable
trainable_c = total_params
print(f"  Stage C trainable parametreler:")
print(f"    All:                 {trainable_c:>10,}")
print(f"    Frozen:              {0:>10,}")
print()

# ============================================================
# 7. CosineAnnealingWarmRestarts Scheduler
# ============================================================
print("=" * 70)
print("7. CosineAnnealingWarmRestarts Scheduler")
print("=" * 70)

t_0 = 10
t_mult = 2
eta_min = 1e-6

# Her stage'de ayrı scheduler oluşturuluyor
for stage_name, base_lr, max_epochs in [("A", 1e-3, 30), ("B", 1e-4, 25), ("C", 1e-4, 25)]:
    print(f"\n  Stage {stage_name} (lr={base_lr}, max_epochs={max_epochs}):")
    print(f"    CosineAnnealingWarmRestarts(T_0={t_0}, T_mult={t_mult}, eta_min={eta_min})")
    
    # Restart noktaları
    restart_at = []
    t = 0
    period = t_0
    while t < max_epochs:
        restart_at.append(t)
        t += period
        period *= t_mult
    
    print(f"    Restart noktaları: {restart_at}")
    
    # LR trajectory (birkaç nokta)
    print(f"    LR trajectory:")
    current_period_start = 0
    current_period_len = t_0
    for epoch in range(max_epochs):
        # Hangi period'dayız?
        if epoch >= current_period_start + current_period_len:
            current_period_start += current_period_len
            current_period_len *= t_mult
        
        t_cur = epoch - current_period_start
        lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + np.cos(np.pi * t_cur / current_period_len))
        
        if epoch % 5 == 0 or epoch == max_epochs - 1:
            print(f"      Epoch {epoch:2d}: lr={lr:.6f}")

print()

# ============================================================
# 8. Early Stopping & Checkpointing
# ============================================================
print("=" * 70)
print("8. Early Stopping & Checkpointing")
print("=" * 70)

patience = 10

print(f"""
  Her stage'de ayrı ModelCheckpoint:
    metric: macro_f1 (maximize)
    patience: {patience} epoch
    
  Checkpoint dosyaları:
    Stage A → checkpoints/stage-a.pt
    Stage B → checkpoints/stage-b.pt
    Stage C → checkpoints/stage-c.pt
    
  Auto-resume mantığı:
    1. stage-c.pt varsa → eğitim tamamlanmış, atla
    2. stage-b.pt varsa → Stage C'den devam et
    3. stage-a.pt varsa → Stage B'den devam et
    4. Hiçbiri yoksa → Stage A'dan başla

  Checkpoint içeriği:
    model: model.state_dict()
    optimizer: optimizer.state_dict()
    epoch: int
    metrics: dict (accuracy, macro_f1, kappa, per-class metrics)
""")

# ============================================================
# 9. Gradient Clipping Hesabı
# ============================================================
print("=" * 70)
print("9. Gradient Clipping")
print("=" * 70)

grad_clip = 1.0
print(f"""
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm={grad_clip})
  
  Global gradient L2 norm hesaplanır:
    ||g|| = sqrt(Σ ||g_i||²)  tüm parametreler üzerinden
    
  Eğer ||g|| > {grad_clip}:
    g_i ← g_i × ({grad_clip} / ||g||)
    
  Bu, gradient explosion'ı önler.
  Özellikle BiGRU'da uzun sequence'larda (L=25) gradient
  patlamasını engeller.
  
  Total trainable params:
    Stage A: {trainable_a:,} → gradient boyutu = {trainable_a:,} float32
    Stage B: {trainable_b:,}
    Stage C: {trainable_c:,}
""")
