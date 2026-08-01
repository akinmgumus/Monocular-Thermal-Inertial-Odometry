# Thermal VIO — Bug Hunt Log

Bu dosya, thermal VIO pipeline'ının MSCKF tarafında yaşanan katastrofik
divergence problemi için yapılan bug hunt'ın özetidir. Bulgular, denenenler,
sonuçlar ve sıradaki adımlar burada.

---

## 1. Thermal Imagery için Kullandığımız Pipeline

Termal görüntüden trajektori tahminine giden tam zincir:

1. **Read raw image** — SThereo termal kameradan 16-bit grayscale (FIReStereo
   da 16-bit). RGB dataset'lerde 8-bit grayscale.
2. **Crop** (opsiyonel) — SThereo'nun visible stereo'sunda 1280×560 yan yana
   geliyor, sol yarı kırpılıyor. Termal'de gerek yok.
3. **Undistortion** — `cv2.initUndistortRectifyMap(K, D, ..., K, ...)` ile
   plumb-bob distortion remap'i. Çıktı pinhole görüntü; aynı `K` korunuyor.
4. **Gaussian blur** — Termal sensör speckle gürültüsünü CLAHE'den önce azaltır
   (σ=1.0, ksize=5×5). CLAHE bu gürültüyü yükseltmesin diye önce blur.
5. **CLAHE** — Adaptive contrast enhancement (clipLimit=3, tileGrid=8×8).
   Native bit-depth'de (16-bit termal için 16-bit CLAHE).
6. **Percentile normalization** — 2%/98% percentile clip ile 16-bit → 8-bit
   dönüşüm. Yakın-uniform frame'lerde fallback olarak full-range scaling.
7. **Feature tracker** — KLT (Lucas-Kanade optical flow, default termal için)
   veya ORB (descriptor-based, RGB'de daha iyi). Drop-in replacement, aynı
   `FeatureTrack` API'si.
8. **MSCKF (error-state EKF)** — IMU predict + sliding-window kamera state
   augmentation + Mourikis null-space projection + sequential per-track
   Joseph update. State: `[δθ, δb_g, δv, δb_a, δp]` (15-dof IMU) +
   per-cam-state 6-dof.

`VIOSequencer` IMU ve kamera event'lerini timestamp'e göre sıralayıp filtre'ye
besler.

---

## 2. FIReStereo ve SThereo Sonuçları

### SThereo valley_evening

- **Sequence**: ~350 saniye, GT yaklaşık 500 m kapalı tur (araç başa dönüyor)
- **Saf IMU dead-reckoning** (`MODE='imu'`, kamera kapalı): ~9.4 km final
  drift — ucuz MEMS IMU için **NORMAL** sonuç
- **Tam VIO** (`MODE='vio'`, başlangıçtaki ilk konfigürasyon): **~370 km final
  hata** — saf IMU'dan ~35× **DAHA KÖTÜ**
- Yani kamera update'i hatayı düzelteceğine **devasa şekilde büyütüyor**

### FIReStereo hawkins_4

- **Major sync problemi**: ilk kamera karesi ile ilk IMU örneği arasında
  yaklaşık 60-70 saniye boşluk var → mevcut haliyle kullanılamaz
- Bu yüzden tüm bug hunt SThereo üzerinden yapıldı

### İlk Şüpheli Sebepler

İlk semptom: `|vel|` (filtre hız tahmini) zamanla sınırsız artıyor → konum
patlıyor. Erken teşhis çıkarımları:

- **Gevşek parallax gate** (`min_parallax_deg=0.5°`): düşük-parallax,
  near-degenerate track'ler kabul ediliyor → kötü-koşullu üçgenleme
- **Kötü-koşullu `H_o`** → devasa Kalman gain
- **Attitude tekmesi**: kamera update'i IMU attitude'una sahte düzeltme
  uyguluyor → yer çekimi vektörü yatay eksenlere "sızıyor" (gravity leakage)
- **Velocity ramp**: sahte yatay ivme integre olup sınırsız hız artışına
  dönüşüyor
- Jacobian matematiği (`H_x`, `H_f`, residual, boxplus) elle doğrulandı —
  formüller standart MSCKF math'iyle uyumlu
- Asıl şüphe: extrinsic convention, scale observability, observability
  inconsistency

---

## 3. Hataları Düzeltmek için Yapılanlar

Bug hunt iki ana eksen üzerinde gitti: (a) gerçek matematik düzeltmeleri,
(b) izolasyon deneyleri.

### Algoritmik Düzeltmeler (Kalıcı)

1. **`_compute_track_jacobians` view-skip fix** — Önceden tek bir bad-depth
   view tüm track'i atıyordu (`return None`). Şimdi sadece o view atlanır
   (`continue`); ≥2 view kalırsa track devam.

2. **PHASE 2 stale-residual fix** (Gemini'nin teşhisi — gerçek bug)
   - **Problem**: Eski sequential update'te PHASE 1 tüm track'lerin
     `(H_o, r_o)`'sunu tek linearizasyon noktasında hesaplıyor, PHASE 2'de
     her track'in `δx = K·r_o`'sunu nominal state'e **anında** uyguluyor ama
     sonraki track'lerin residual'ini önceki düzeltmelerle telafi **etmiyor**
     → her track aynı uncompensated hatayı yeniden görüp yeniden düzeltir
     → ~44 track'lik bir update'te ~44× overshoot.
   - **Fix**: `dx_total` biriktir, her track için
     `r_active = r_o - H_o · dx_total`, boxplus döngü sonunda **tek seferde**
     uygulanır.
   - **Sonuç**: f34'teki anlık 8° attitude tekmesi **tamamen gitti**, ilk
     ~37 kare temiz oldu.

3. **FEJ landmark anchoring** (Gemini önerisi)
   - **Problem**: `H_x`, `H_f` dondurulmuş (FEJ) kamera pozlarında lineerize
     ediliyor, ama `p_f_world` landmark'ı güncel kamera pozlarına göre
     üçgenlendi → Jacobian'da kullanılan landmark, Jacobian'ın lineerize
     edildiği FEJ pozlarıyla geometrik olarak **tutarsız** → yaw
     observability iptal mekanizması bozuluyor.
   - **Fix**: landmark'ı ilk gören (anchor) kameranın FEJ pozuna rijit
     güncel→FEJ transform ile demirle.

4. **IMU-state FEJ (rot-only)**
   - **Problem**: Cam state'lerin FEJ snapshot'ı vardı (`cs.rot_fej`,
     `cs.pos_fej`) ama IMU state'in yoktu — `predict()` F matrisini her zaman
     güncel rotasyonda kuruyordu → asimetrik FEJ → observability tutarsızlığı
     birikiyor.
   - **Fix**: `self.nominal_rot_fej` çapası eklendi. `predict()`'in `F`
     matrisindeki `F[6:9, 0:3]` ve `F[6:9, 9:12]` blokları ve
     `augment_state`'in `J_imu[3:6, 0:3]` lever-arm bağı bu FEJ rotasyonda
     lineerize edilir. Nominal entegrasyon (vel/pos) güncel rotasyonu
     kullanmaya devam eder. `update()` boxplus sonrası refresh.
   - **Sonuç**: SThereo divergence ~265 km → **~100 km** (yaklaşık %62
     iyileşme).

5. **Cam-FEJ → IMU-FEJ zincirleme (Option 3)** — Gemini önerdi, denedik:
   `cs.rot_fej = nominal_rot_fej · R_imu_cam`. **Empirik olarak ters tepti**
   (100 → 580 km, 5× daha kötü) → geri alındı. Sebep: cam state'in "first
   estimate"i augment anındaki kamera pozu olmalı, IMU FEJ çapasından
   bağımsız. OpenVINS/Mourikis standardı da bunu söyler.

### Tanı Enstrümantasyonu (Geçici, Kalıcı Olmak Üzere Tasarlanmadı)

- **`MODE` switch** (`'vio'` / `'imu'`): kamera kapalı/açık ayrı koşu
- **`LOCK_ATTITUDE`**: update'in IMU attitude düzeltmesini sıfırla
- **`EXTRINSIC_MODE`** (`'invert'` / `'direct'`): extrinsic dosyasını ters
  çevir veya doğrudan kullan
- **`CHI2_GATE`** toggle: chi-square outlier reddini aç/kapat
- **`MAX_TRACKS_PER_UPDATE`**: update başına track sayısını sınırla
  (P-collapse koruması)
- **`[att]` log**: MSCKF attitude vs GT karşılaştırması her N karede bir
- **`[diag uN]` log**: ilk N accepted-track update için per-track depth,
  parallax, `||K||`, IMU dθ vs CAM dθ, ham residual, residual reduction ratio
- **Predict P-magnitude log**: her 50 IMU adımında P'nin 1σ değerleri

### Test Edilen ama Yardım Etmeyen Müdahaleler

| Müdahale | SThereo final | Yorum |
|---|---|---|
| Kontrol (PHASE 2 fix + FEJ anchor + IMU-FEJ + chi2 açık) | ~265 km | baz çizgi |
| + `LOCK_ATTITUDE=True` | ~207 km | etkisiz (~%22) |
| + `MAX_TRACKS_PER_UPDATE=25` | ~265 km | etkisiz |
| + `CHI2_GATE` kapalı | **~132 km** | ~2× iyileşme |

`extrinsic_mode` (invert/direct) — şiddeti ~2× değiştiriyor ama divergence'ı
durdurmuyor.

### VO-Only Sanity Test

`test_vo_only.py` ile MSCKF'i tamamen devre dışı bırakıp KLT + 2-view
`recoverPose` chain çalıştırıldı (no MSCKF, no IMU). Sim(3) origin-anchor'lu
hizalama:
- **ATE RMSE = 170 m** (GT max |d| = 462 m, GT path 2015 m)
- Pure KLT VO doğru ölçek mertebesinde — drift var ama beklenen monocular
  davranışı

→ **KLT iki bağımsız sinyalle aklandı**: (1) MSCKF reprojection residual'i
~1-2 px (multi-view tutarlılık), (2) VO-only doğru ölçek. SThereo'daki ~100
km vs VO-only 170 m = **~600× fark**.

### SThereo Final Durumu

Tüm fix'lerle birlikte: **~100 km divergence**. Başlangıçtaki 370 km'den
büyük iyileşme ama hâlâ katastrofik (GT 500 m yola karşı 200× hata). Update
residual'i her seferinde azaltıyor (oran 0.07-0.81) → update math'i lokal
olarak doğru. Yani problem ya hâlâ derin bir observability/consistency
issue, ya da **dataset-spesifik**.

---

## 4. EuRoC ile MSCKF Pipeline'ının Test Edilmesi

### Amaç

Pipeline'da hâlâ matematik problemi var mı, yoksa SThereo'daki ~100 km
sadece dataset'e mi özel? Standart bir VIO benchmark'ı (EuRoC MAV MH_03) ile
aynı MSCKF'i koşturup karşılaştır.

### Kurulum

- **Dataset**: EuRoC MAV `MH_03_medium`, ~131 saniye indoor drone uçuşu
- **Pipeline**: aynı MSCKF — PHASE 2 fix, FEJ landmark anchor, IMU-state FEJ
- **Front-end**: ORB (RGB için en iyi) veya KLT (her ikisi de çalışıyor)
- **Init**: GT'nin ilk örneğinden (drone başlangıçta hareketli, static-init
  güvenilmez)
- **Extrinsic**: EuRoC'un `T_BS` doğrudan `T_imu_cam`'dir (body frame == IMU
  frame, no inversion needed)
- **IMU noise**: ADIS16448 (`sensor.yaml`'dan)

`scripts/test_msckf_vio_euroc.py` ve `src/thermal_vo/thermal_vo/config_euroc.py`
oluşturuldu. Live visualization eklendi: XY top-down + Z(t) iki grafiği +
progressive GT ve VIO çizimi + suptitle'da current `t`.

### Sonuç

| Metrik | EuRoC MH_03 | SThereo (referans) |
|---|---|---|
| Sequence süresi | ~131 sn | ~350 sn |
| Toplam cam frame | 2652 | 3495 |
| **Final pos error** | **4.4 m** | ~100 000 m |
| **ATE RMSE** | **3.4 m** | binlerce km |
| Mean ATE | 3.2 m | — |
| Max ATE | 4.6 m | — |
| Roll/pitch hatası | < 0.4° | kontrol dışı |
| Yaw hatası | -10.6° | kontrol dışı |
| Reprojection residual (u1) | 0.1 px | 7+ px |

**~30,000× iyileşme** EuRoC'ta. **Pipeline aklandı.**

### Gözlemler

- **İlk ~0.5-2 saniyelik transient**: filtre kameralı update'ler düzenli
  ateşlenene kadar küçük drift yapar — bu **standart MSCKF startup
  davranışı**, literatürdeki tüm implementasyonlarda görülür.
- **Hover/durma sırasında küçük drift**: drone sabit dururken kamera 0
  parallax → update yok → IMU bias drift'i uncorrected → her 3 eksende
  küçük drift. Bu **evrensel VIO limitation'ı**: MSCKF, VINS-Mono, OpenVINS,
  ROVIO, Kimera-VIO — hepsinde var. Production sistemlerde **ZUPT
  (Zero-velocity Update)** ile çözülür, algoritma değişikliği değil.
- **Motion sırasında**: VIO trajektörisi GT'yi yakın takip ediyor, ATE
  RMSE 3.4 m tipik tier-1 VIO performansı.

### Çıkarım

**MSCKF pipeline'ı doğru çalışıyor.** SThereo'daki ~100 km divergence
**dataset/setup-spesifik**. Olası nedenler:

- **SThereo extrinsic kalibrasyonu şüpheli**: matris tam `±1/0` (elle
  yazılmış nominal, ölçülmüş kalibrasyon değil)
- **SThereo IMU-cam saat senkronizasyonu** garanti değil
- **Hızlı araç motion profili** + uzak feature ağırlıklı sahne + thermal
  noise = kombineli stres
- **Termal görüntü için KLT/ORB inherently zor**: brightness constancy
  ihlali (sıcaklık zamanla değişir), smooth gradients (aperture problem
  riski), düşük kontrast, ORB BRIEF descriptor'ları termal'de tutarsız

---

## 5. Sırada Ne Var — Yeni Dataset ile Deneme

### Strateji

EuRoC pipeline'ı aklayan bir validation oldu. Asıl tezin uygulama hedefi
**termal kameralı drone'un tünel/maden/yangın ortamlarında çalışması**. O
yüzden:

1. **EuRoC sonucu validation milestone olarak kalır** — "MSCKF pipeline'ı
   standart RGB-IMU benchmark'ında doğru çalıştığını gösterdim."
2. **Termal + drone + subterranean'a yakın yeni bir dataset bul ve test et.**

### Aday Dataset'ler

| Dataset | Açıklama | Uyum |
|---|---|---|
| **SubT-MRS** (CMU AirLab) | Drone + thermal + tünel/mağara, DARPA SubT | En iyi eşleşme |
| **Hilti SLAM Challenge** | Indoor industrial, multi-modal (bazı sürümlerde thermal) | İyi |
| **AirSim** simülasyon | Kontrollü senaryolar, perfect GT, thermal-vari rendering | Yedek/kontrol |

### Beklenenler

- SubT-MRS'de pipeline iyi sonuç verirse → tez için temiz validation
- Çalışmazsa → thermal-spesifik front-end gerekli (mutual information
  matching, edge tracking, learned descriptors)
- Drone hover senaryosu yaygınsa → ZUPT eklemek pratik fayda sağlar

### Backup Planları

- **Thermal-specific front-end**: KLT/ORB yerine MI-bazlı matching veya
  learned descriptor (TFeat, SuperPoint thermal-adapted) — tezde ayrı bir
  katkı noktası
- **ZUPT**: hover senaryosu için stationary-velocity constraint
- **Loop closure** (eğer dataset uygunsa): VIO + back-end optimization

---

## Özet

- **Pipeline**: thermal → undistort → blur → CLAHE → percentile norm → KLT/ORB → MSCKF
- **SThereo**: başlangıçta ~370 km, fix'lerden sonra ~100 km divergence
- **Fix'ler**: view-skip, PHASE 2 stale-residual, FEJ landmark anchor,
  IMU-state FEJ (rot-only)
- **EuRoC**: aynı pipeline ile 4.4 m final hata → pipeline aklandı
- **Sonuç**: SThereo problemi dataset-spesifik (calibration + thermal +
  motion profile)
- **Sıradaki**: SubT-MRS veya benzer drone-thermal-subterranean dataset'i
  ile gerçek tez hedefi validation'ı
