"""
PhysioGraphSleep — Data Pipeline ve Graph Construction Hesaplamaları
====================================================================

Dataset boyutları, spectral feature extraction, graph construction,
augmentation parametreleri.
"""

import numpy as np

print("=" * 70)
print("PhysioGraphSleep — Data Pipeline Hesaplamaları")
print("=" * 70)

# ============================================================
# 1. Dataset Boyutları
# ============================================================
print("\n" + "=" * 70)
print("1. Sleep-EDF-20 Dataset Boyutları")
print("=" * 70)

num_subjects = 20
train_subjects = 14
val_subjects = 3
test_subjects = 3

# Her subject yaklaşık ~8 saat uyku = ~960 epoch (30s aralıklar)
# Ama değişkenlik var, tipik: 800-1200 epoch/subject
epochs_per_subject_avg = 1000  # yaklaşık

# Subject split: 14-3-3
print(f"""
  Toplam subject: {num_subjects}
  Split: train={train_subjects}, val={val_subjects}, test={test_subjects}

  Tahmini epoch sayıları:
    Train: ~{train_subjects * epochs_per_subject_avg:,} epoch
    Val:   ~{val_subjects * epochs_per_subject_avg:,} epoch
    Test:  ~{test_subjects * epochs_per_subject_avg:,} epoch

  Not: Gerçek sayılar çalıştırınca gösterilir.
  Sleep-EDF-20'de her subject 2 gece kaydı olabilir.
""")

# ============================================================
# 2. Sinyal İşleme Parametreleri
# ============================================================
print("=" * 70)
print("2. Sinyal İşleme Parametreleri")
print("=" * 70)

fs = 100  # Hz
epoch_duration = 30  # seconds
epoch_samples = fs * epoch_duration  # 3000

bandpass_low = 0.3  # Hz
bandpass_high = 35.0  # Hz

print(f"""
  Örnekleme frekansı: {fs} Hz
  Epoch süresi: {epoch_duration} s
  Epoch başına örnek: {fs} × {epoch_duration} = {epoch_samples}

  Bandpass filtre: {bandpass_low} - {bandpass_high} Hz
    Nyquist frekansı: {fs/2} Hz
    Normalized low:  {bandpass_low / (fs/2):.4f}
    Normalized high: {bandpass_high / (fs/2):.4f}
    → Delta (0.5-4 Hz) korunur ✅
    → 50 Hz mains noise filtrelenir ✅

  Kanal: EEG Fpz-Cz (tek kanal)
    shape: (N_epochs, 1, 3000)
""")

# ============================================================
# 3. Sequence Construction
# ============================================================
print("=" * 70)
print("3. Sequence Construction")
print("=" * 70)

seq_len = 25
center_idx = seq_len // 2  # 12

print(f"""
  Sequence uzunluğu: {seq_len}
  Center epoch index: {seq_len} // 2 = {center_idx}

  Her sample = {seq_len} ardışık epoch penceresi
  Center epoch (index {center_idx}) sınıflandırılır

  Context:
    Geçmiş: {center_idx} epoch (= {center_idx * epoch_duration}s = {center_idx * epoch_duration / 60:.1f} dakika)
    Gelecek: {seq_len - center_idx - 1} epoch (= {(seq_len - center_idx - 1) * epoch_duration}s)

  Padding:
    İlk {center_idx} ve son {seq_len - center_idx - 1} epoch'ta zero-padding uygulanır
    Mask tensörü ile invalid epoch'lar işaretlenir

  Dataset uzunluğu = N_epochs (her epoch bir center olur)
    Train: ~{train_subjects * epochs_per_subject_avg:,} sample
    Val:   ~{val_subjects * epochs_per_subject_avg:,} sample
    Test:  ~{test_subjects * epochs_per_subject_avg:,} sample
""")

# ============================================================
# 4. Spectral Feature Extraction
# ============================================================
print("=" * 70)
print("4. Spectral Feature Extraction")
print("=" * 70)

num_patches = 6
patch_duration = 5  # seconds
patch_samples = fs * patch_duration  # 500

num_bands = 5
band_ranges = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 11.0),
    "sigma": (11.0, 16.0),
    "beta": (16.0, 30.0),
}

features_per_patch = 7
features_per_band = num_patches * features_per_patch  # 42

print(f"""
  Patch yapısı:
    Epoch uzunluğu: {epoch_samples} sample = {epoch_duration}s
    Patch sayısı: {num_patches}
    Patch uzunluğu: {patch_samples} sample = {patch_duration}s
    Doğrulama: {num_patches} × {patch_samples} = {num_patches * patch_samples} = {epoch_samples} ✅

  Frekans bantları ({num_bands} bant):""")

for band_name, (low, high) in band_ranges.items():
    print(f"    {band_name:8s}: {low:5.1f} - {high:5.1f} Hz")

print(f"""
  Welch PSD parametreleri:
    nperseg = min(256, {patch_samples}) = 256
    Frekans çözünürlüğü: {fs}/{256} = {fs/256:.4f} Hz

  Her patch × her band → 7 feature:
    1. abs_power    — Mutlak güç (bant integral)
    2. rel_power    — Göreceli güç (bant / toplam)
    3. log_power    — log(abs_power)
    4. entropy      — Shannon spektral entropi
    5. delta_theta  — log(delta / theta) oranı
    6. sigma_beta   — log(sigma / beta) oranı
    7. theta_alpha  — log(theta / alpha) oranı

  Boyut hesabı:
    features_per_band = num_patches × features_per_patch
                      = {num_patches} × {features_per_patch} = {features_per_band}
    
    Toplam: ({num_bands}, {features_per_band}) = ({num_bands} bands, {features_per_band} features)
""")

# Welch PSD detayları
nperseg = min(256, patch_samples)
# Welch default: noverlap = nperseg // 2
noverlap = nperseg // 2
# Frekans binleri
n_freq_bins = nperseg // 2 + 1  # positive frequencies only
freq_resolution = fs / nperseg

print(f"  Welch PSD detayları:")
print(f"    nperseg:   {nperseg}")
print(f"    noverlap:  {noverlap}")
print(f"    n_freq:    {n_freq_bins} bins")
print(f"    freq_res:  {freq_resolution:.4f} Hz")
print(f"    max_freq:  {fs/2} Hz")
print()

# Her bant kaç freq bin'i kapsıyor?
print(f"  Bant başına frekans bin sayısı:")
freqs = np.linspace(0, fs/2, n_freq_bins)
for band_name, (low, high) in band_ranges.items():
    mask = (freqs >= low) & (freqs <= high)
    n_bins = mask.sum()
    print(f"    {band_name:8s}: {low:5.1f}-{high:5.1f} Hz → {n_bins} bins")

# Cross-band features
print(f"""
  Çapraz-bant oranları (features 5-7):
    delta_theta = log(P_delta / P_theta)
      → Derin uyku (N3) göstergesi: yüksek delta, düşük theta
    sigma_beta  = log(P_sigma / P_beta)
      → Sleep spindle göstergesi (N2): yüksek sigma
    theta_alpha = log(P_theta / P_alpha)
      → Uyku başlangıcı (N1) göstergesi: theta artışı

  NOT: Bu 3 feature her bant altında AYNI değerler
  (bant-bağımsız)! Yani 5 bant × 3 = 15 eleman tekrar ediyor.
  Ama model bunu handle edebilir (redundancy zararsız).
""")

# ============================================================
# 5. Graph Construction
# ============================================================
print("=" * 70)
print("5. Graph Construction (HeteroGraphEncoder)")
print("=" * 70)

PATCH_OFFSET = 0
BAND_OFFSET = 6
SUMMARY_OFFSET = 11
NUM_PATCH = 6
NUM_BAND = 5
NUM_NODES = 12

# Edge hesaplamaları
print(f"""
  Node yapısı:
    Patch nodes:   {NUM_PATCH} (index {PATCH_OFFSET}-{PATCH_OFFSET + NUM_PATCH - 1})
    Band nodes:    {NUM_BAND}  (index {BAND_OFFSET}-{BAND_OFFSET + NUM_BAND - 1})
    Summary node:  1          (index {SUMMARY_OFFSET})
    Toplam:        {NUM_NODES} node per epoch
""")

# Edge type 0: Patch-Patch (temporal adjacency)
patch_patch_edges = []
for i in range(NUM_PATCH - 1):
    patch_patch_edges.append((PATCH_OFFSET + i, PATCH_OFFSET + i + 1))
    patch_patch_edges.append((PATCH_OFFSET + i + 1, PATCH_OFFSET + i))
E_pp = len(patch_patch_edges)

# Edge type 1: Band-Band (fully connected)
band_band_edges = []
for i in range(NUM_BAND):
    for j in range(NUM_BAND):
        if i != j:
            band_band_edges.append((BAND_OFFSET + i, BAND_OFFSET + j))
E_bb = len(band_band_edges)

# Edge type 2: Patch-Band (cross-modal)
patch_band_edges = []
for i in range(NUM_PATCH):
    for j in range(NUM_BAND):
        patch_band_edges.append((PATCH_OFFSET + i, BAND_OFFSET + j))
        patch_band_edges.append((BAND_OFFSET + j, PATCH_OFFSET + i))
E_pb = len(patch_band_edges)

# Edge type 3: Summary (connected to all others)
summary_edges = []
for i in range(SUMMARY_OFFSET):  # 0-10
    summary_edges.append((SUMMARY_OFFSET, i))
    summary_edges.append((i, SUMMARY_OFFSET))
E_s = len(summary_edges)

E_total = E_pp + E_bb + E_pb + E_s

print(f"  Edge hesaplamaları:")
print(f"    Type 0 — Patch-Patch (temporal adjacency, bidirectional):")
print(f"      (NUM_PATCH-1) × 2 = {NUM_PATCH-1} × 2 = {E_pp}")
print(f"      Edges: {patch_patch_edges}")
print()

print(f"    Type 1 — Band-Band (fully connected, bidirectional):")
print(f"      NUM_BAND × (NUM_BAND-1) = {NUM_BAND} × {NUM_BAND-1} = {E_bb}")
print(f"      (Her bant diğer 4 banda bağlı)")
print()

print(f"    Type 2 — Patch-Band (cross-modal, bidirectional):")
print(f"      NUM_PATCH × NUM_BAND × 2 = {NUM_PATCH} × {NUM_BAND} × 2 = {E_pb}")
print()

print(f"    Type 3 — Summary (hub node, bidirectional):")
print(f"      (NUM_NODES-1) × 2 = {NUM_NODES-1} × 2 = {E_s}")
print()

print(f"    Toplam edges per epoch: {E_pp} + {E_bb} + {E_pb} + {E_s} = {E_total}")
print()

# Batch genişletme
B = 64
print(f"  Batch genişletme (B={B}):")
print(f"    Batched nodes:  {B} × {NUM_NODES} = {B * NUM_NODES}")
print(f"    Batched edges:  {B} × {E_total} = {B * E_total}")
print(f"    edge_index: (2, {B * E_total})")
print()

# ============================================================
# 6. Augmentation Hesaplamaları
# ============================================================
print("=" * 70)
print("6. Augmentation Parametreleri")
print("=" * 70)

noise_std = 0.01
shift_max = 50
scale_range = (0.9, 1.1)

T = 3000  # epoch length in samples

# Time mask
mask_min = T // 20  # 150
mask_max = T // 7 + 1  # 429

print(f"""
  Augmentasyon (sadece train, her epoch'a bağımsız uygulanır):
  
  1. Gaussian Noise (p=0.50):
     x += N(0, σ={noise_std})
     SNR etkisi: peak amplitude ~1.0 ise, SNR = 20×log10(1/{noise_std}) = {20*np.log10(1/noise_std):.0f} dB
     → Çok hafif noise, sinyali bozmaz
  
  2. Time Shift (p=0.30):
     Circular roll ±{shift_max} sample
     Zaman kayması: ±{shift_max/fs:.1f} saniye
     → {shift_max/T*100:.1f}% epoch kayması, fizyolojik olarak makul
  
  3. Amplitude Scale (p=0.50):
     x *= Uniform({scale_range[0]}, {scale_range[1]})
     → ±10% amplitüd değişimi
  
  4. Time Mask (p=0.30):
     Mask length: Uniform({mask_min}, {mask_max}) sample
     Mask length:  {mask_min/fs:.1f}s - {mask_max/fs:.1f}s
     Maskelenen oran: {mask_min/T*100:.1f}% - {mask_max/T*100:.1f}%
     → SpecAugment tarzı regularization

  Beklenen augmentasyon kombinasyonları per epoch:
    Hiç augmentasyon yok: 0.5 × 0.7 × 0.5 × 0.7 = {0.5*0.7*0.5*0.7:.4f} ({0.5*0.7*0.5*0.7*100:.1f}%)
    En az bir: {1 - 0.5*0.7*0.5*0.7:.4f} ({(1-0.5*0.7*0.5*0.7)*100:.1f}%)
""")

# ============================================================
# 7. DataLoader Hesaplamaları
# ============================================================
print("=" * 70)
print("7. DataLoader Hesaplamaları")
print("=" * 70)

batch_size = 64
num_workers = 2
train_epochs_est = 14000  # yaklaşık train epoch sayısı

batches_per_epoch = train_epochs_est // batch_size
last_batch_size = train_epochs_est % batch_size

print(f"""
  Batch size: {batch_size}
  Num workers: {num_workers} (Colab'da ayarlanan)
  Pin memory: True

  Train DataLoader:
    Samples: ~{train_epochs_est:,}
    Batches: {train_epochs_est} // {batch_size} = {batches_per_epoch}
    Son batch: {last_batch_size} sample (WeightedRandomSampler replacement=True → always {batch_size})

  Not: WeightedRandomSampler(replacement=True) → her epoch
       {train_epochs_est} sample seçer, {batches_per_epoch} batch oluşur.
       Bazı samplelar birden fazla kez seçilebilir (özellikle N1).

  Val DataLoader:
    Samples: ~{val_subjects * epochs_per_subject_avg:,}
    Batches: ~{val_subjects * epochs_per_subject_avg // batch_size}
    Shuffle: False (deterministik evaluation)

  Bellek per batch:
    Signal:   {batch_size}×{seq_len}×1×{T} × 4B = {batch_size*seq_len*1*T*4/1e6:.1f} MB
    Spectral: {batch_size}×{seq_len}×5×42 × 4B = {batch_size*seq_len*5*42*4/1e6:.2f} MB
    Labels:   ihmal edilebilir
    Total:    ~{(batch_size*seq_len*1*T*4 + batch_size*seq_len*5*42*4)/1e6:.1f} MB per batch
""")

# ============================================================
# 8. Boundary & Auxiliary Labels
# ============================================================
print("=" * 70)
print("8. Boundary & Yardımcı Label Hesaplamaları")
print("=" * 70)

print(f"""
  Her sample (center epoch) için hesaplanan auxiliary labels:
  
  1. boundary (float):
     center_label != seq_labels[center_idx-1] veya
     center_label != seq_labels[center_idx+1]
     → Uyku evresi geçiş noktası mı?
     Beklenen boundary oranı: ~15-25% (tipik geçiş oranı)
  
  2. prev_label (int64):
     seq_labels[center_idx - 1]
     → Önceki epoch'un evresi
  
  3. next_label (int64):
     seq_labels[center_idx + 1]
     → Sonraki epoch'un evresi
  
  4. n1_label (float):
     1.0 if center_label == 1 else 0.0
     → Binary N1 detector
     Beklenen pozitif oranı: ~2.4% (N1 prevalansı)
  
  İmbalance durumu:
    boundary: ~20% pozitif → hafif imbalance, BCEWithLogitsLoss yeterli
    n1_label: ~2.4% pozitif → ciddi imbalance, ama yardımcı task
    → n1_aux_weight=0.30 ile kontrol altında
""")

# ============================================================
# 9. Hash/Cache Sistemi
# ============================================================
print("=" * 70)
print("9. Veri Boyutu Tahminleri")
print("=" * 70)

# Raw veri boyutu
raw_epochs = train_subjects * epochs_per_subject_avg
raw_signal_mb = raw_epochs * 1 * T * 4 / 1e6  # float32
raw_spectral_mb = raw_epochs * 5 * 42 * 4 / 1e6
raw_labels_mb = raw_epochs * 8 / 1e6  # int64

print(f"""
  Raw veri boyutu (train, float32):
    Signal:   {raw_epochs:,} × 1 × {T} × 4B = {raw_signal_mb:.0f} MB
    Spectral: {raw_epochs:,} × 5 × 42 × 4B = {raw_spectral_mb:.1f} MB
    Labels:   {raw_epochs:,} × 8B = {raw_labels_mb:.2f} MB
    Toplam:   ~{raw_signal_mb + raw_spectral_mb + raw_labels_mb:.0f} MB
    
  Cache sistemi:
    Ön-işlenmiş veriler cache_dir'e kaydedilir
    İlk çalıştırmada: download + preprocess (~30-60 dk)
    Sonraki çalıştırmalarda: cache'den yükle (~1-2 dk)
""")
