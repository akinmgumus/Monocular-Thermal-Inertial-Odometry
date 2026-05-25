"""
Live-visualization smoke test for the thermal MSCKF VIO pipeline.

Two panels updated each frame:
  • Top  — previous + current thermal frame side-by-side, with green lines
           between matched ORB features that survived ratio + RANSAC and were
           assigned to a still-tracked feature.
  • Bot. — top-down trajectory (XY): VIO estimate vs SThereo local-pose GT,
           anchored to a common origin at the static-init boundary.

Set VIZ_EVERY > 1 to skip frames in the viz; the filter still runs every
frame, only the redraw is throttled.

MODE isolates the sensors for diagnosis:
  'vio' — IMU predict + camera measurement update   (full fusion)
  'imu' — IMU predict only, camera disabled          (inertial dead-reckoning)
Run 'imu' and 'vio' back to back and compare the saved trajectories: if 'vio'
is worse than 'imu', the camera path is what diverges — the gap localises it.

Run:
    python3 scripts/test_msckf_vio_live.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/thermal_vo'))

from thermal_vo.dataloader      import ThermalDataLoader, IMULoader, VIOSequencer
from thermal_vo.orb             import ORBTracker
from thermal_vo.klt             import KLTTracker
from thermal_vo.msckf           import MSCKF
from thermal_vo                 import config_sthereo as config


# ── FRONT-END SEÇİMİ ──────────────────────────────────────────────────────
# 'orb' → ORBTracker (descriptor-tabanlı, termal'de track avg_len ≈ 1.4)
# 'klt' → KLTTracker (pixel-level optical flow, termal'de track avg_len ≈ 4.5+)
# Bu sabitte hata yaparsan make_tracker() açık bir hata ile durdurur.
METHOD     = 'klt'

# ── ODOMETRY MODU ─────────────────────────────────────────────────────────
# 'vio' → IMU predict + kamera measurement update (tam füzyon, mevcut pipeline)
# 'imu' → yalnızca IMU predict; kamera tamamen devre dışı (saf dead-reckoning)
# Tanı için: 'imu' ile 'vio'yu arka arkaya koş, kaydedilen trajektorileri
# karşılaştır. Çıktılar moda göre ayrı dosyalara yazılır.
MODE       = 'vio'
if MODE not in ('vio', 'imu'):
    raise ValueError(f"Bilinmeyen MODE={MODE!r}; beklenen 'vio' veya 'imu'.")

# ── TANI: ATTITUDE-LOCK ───────────────────────────────────────────────────
# True iken kamera update'i IMU attitude'una (δθ) DOKUNMAZ — delta_x[0:3]
# sıfırlanır. ÇÖZÜM DEĞİL, izolasyon deneyi: diverge durursa spurious
# attitude tekmesi suçludur. Bug bulunup düzeltilince False'a alınacak.
LOCK_ATTITUDE = False

# ── TANI: EXTRINSIC KONVANSIYONU ──────────────────────────────────────────
# 'invert' → eski davranış: dosya T_cam_imu kabul edilip ters çevrilir.
# 'direct' → hipotez: dosya zaten kamera→IMU dönüşümü; ters çevirmeden kullan.
# Eksen analizi 'direct'in standart IMU çerçevesi (x-ileri, y-sol, z-yukarı)
# verdiğini gösteriyor; 'invert' alışılmadık (x-yukarı) çerçeve veriyor.
EXTRINSIC_MODE = 'direct'
if EXTRINSIC_MODE not in ('invert', 'direct'):
    raise ValueError(f"Bilinmeyen EXTRINSIC_MODE={EXTRINSIC_MODE!r}")

# ── TANI: CHI2 GATE ───────────────────────────────────────────────────────
# False iken MSCKF'in per-track chi-square (Mahalanobis) outlier kapısı
# atlanır. Adım 3 turnusol testi: chi2 reddi divergence'ın parçası mı?
CHI2_GATE = False

MAX_WINDOW = 15    # max number of cam states in MSCKF sliding window (older are marginalized out)
VIZ_EVERY  = 20    # redraw every Nth cam frame

# ── ADIM 1: UPDATE BAŞINA TRACK TAVANI ────────────────────────────────────
# marginalize_at_prune pencere dolunca yüzlerce track'i tek update'e
# basabiliyor → P aşırı küçülür (P-collapse) → chi2 gate sonraki ölçümleri
# reddeder. Tavan: her update'te en uzun (en sağlam) MAX_TRACKS_PER_UPDATE
# track işlenir. MSCKF'te fazla track doğruluğu artırmaz, P'yi çökertir.
MAX_TRACKS_PER_UPDATE = 25

USE_CLAHE = False  # denoise thermal + improve contrast with CLAHE (termal VO'da genellikle kapalı tutmak daha iyi sonuç veriyor)
if USE_CLAHE:
    clahe_mode = 'with_CLAHE'
else:
    clahe_mode = 'no_CLAHE'


def make_tracker(method):
    """Front-end factory. ORB ile KLT'nin parametre setleri farklı; her birinin
    kendi tuned-for-thermal değerleri burada toplandı. Bilinmeyen method →
    açık ValueError (yanlışlıkla başka bir tracker'a düşmesin)."""
    if method == 'orb':
        return ORBTracker(
            n_features=500,
            grid_rows=4,
            grid_cols=4,
            ratio_thresh=0.8,           # Lowe's ratio (lower = more strict)
            ransac_thresh=2.5,          # RANSAC inlier eşiği (px)
            min_track_length=3,
            fast_threshold=7,           # FAST corner threshold
            edge_threshold=15,          # ORB edge guard
            max_pixel_displacement=100,  # motion-prior gate (px)
        )
    
    if method == 'klt':
        return KLTTracker(
            n_features=500,
            fb_eps=1.5,                 # forward-backward round-trip toleransı (px)
            ransac_thresh=2.0,          # F-matrix RANSAC eşiği (px)
            min_track_length=3,
            lk_win_size=(30, 30),       # Lucas-Kanade pencere (termal smooth gradient için biraz büyük)
            lk_max_level=3,             # pyramidal LK (4 level toplam, ~30-40 px hareket destekler)
            quality_level=0.02,         # Shi-Tomasi göreli kalite eşiği
            min_distance=8.0,           # iki köşe arası minimum px mesafe
            max_pixel_displacement=100,  # motion-prior gate (px)
        )
    raise ValueError(f"Bilinmeyen METHOD={method!r}; beklenen 'orb' veya 'klt'.")


def wrap_yaw_deg(a):
    """Wrap angle difference to (-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


def run():
    # ---- CALIBRATION ----
    cal = config.load_camera_intrinsics()
    K, D = cal['K'], cal['D'] # K is 3x3, D is 1D array of distortion coeffs; undistort in loader, so D=None for MSCKF

    T_ext = config.load_extrinsic()
    if EXTRINSIC_MODE == 'invert':
        # Eski: dosya = T_cam_imu (IMU→kamera) kabul edilip ters çevrilir.
        R_imu_cam_mat = T_ext[:3, :3].T
        t_imu_cam     = -R_imu_cam_mat @ T_ext[:3, 3]
    else:  # 'direct'
        # Hipotez: dosya zaten kamera→IMU dönüşümü; doğrudan kullan.
        R_imu_cam_mat = T_ext[:3, :3]
        t_imu_cam     = T_ext[:3, 3]
    R_imu_cam = R.from_matrix(R_imu_cam_mat)
    print(f"extrinsic mode: {EXTRINSIC_MODE}   t_imu_cam={t_imu_cam}")

    # ---- DATA ----
    loader = ThermalDataLoader(
        config.THERMAL_LEFT_DIR,
        bit_depth=16,
        undistort=True, K=K, D=D,
        use_clahe=USE_CLAHE,              # CLAHE TANI DENEYİ: kapalı. Adaptif (tile-bazlı)
                                      # kontrast, kareler arası aynı noktanın parlaklığını
                                      # kaydırıp KLT'nin brightness-constancy varsayımını
                                      # bozuyordu. True = eski davranış.
        gaussian_sigma=1.0,           # denoise thermal speckle before CLAHE and ORB; higher = smoother but more detail loss
        gaussian_ksize=(3, 3),
    )
    imu = IMULoader.sthereo(config.IMU_CSV)
    print(f"images: {len(loader)}    IMU: {len(imu)}")

    # ---- STATIC INIT ----
    n_static = int(config.STATIC_INIT_SECONDS / np.median(np.diff(imu.timestamps)))
    msckf = MSCKF(
        K=K, D=None,            # görüntü seviyesinde undistort zaten uygulandı

        # piksel cinsinden ölçüm gürültüsü std'si.
        # ARTARSA: gürültülü tespitlere tolerans artar, outlier'lar daha az reddedilir ama güncellemeler zayıflar -> daha fazla drift.
        # AZALIRSA: filtre ölçümlere fazla güvenir, küçük tespit hatalarına aşırı tepki verir, tutarsızlık ve sıçramalar olur.
        pixel_noise_std=3.0,    # piksel

        # üçgenleme için minimum paralaks açısı (derece).
        # ARTARSA: yalnızca iyi gözlemlenen yakın feature'lar kullanılır, derinlik daha güvenilir ama kullanılabilir feature sayısı azalır.
        # AZALIRSA: uzak/zayıf paralakslı feature'lar da kabul edilir, derinlik tahmini kötüleşir -> daha fazla drift.
        min_parallax_deg=4.0,   # derece

        # kabul edilen minimum feature derinliği (metre).
        # ARTARSA: çok yakın feature'lar elenir, yakın yüzeyden gelen hatalı üçgenlemeler azalır ama feature kaybı olur.
        # AZALIRSA: çok yakın feature'lar da girer, kötü koşullu derinlik tahminleri artar -> ölçek drifti.
        min_depth=0.5,          # metre

        # kabul edilen maksimum feature derinliği (metre).
        # ARTARSA: uzak feature'lar kabul edilir, ölçek bilgisi zayıf gözlemlerden gelir -> ölçek drifti.
        # AZALIRSA: yalnızca yakın feature'lar kullanılır, derinlik daha sağlam ama az feature -> yetersiz kısıt.
        max_depth=400.0,         # metre

        # her güncellemede Gauss-Newton maksimum iterasyon sayısı.
        # ARTARSA: üçgenleme daha iyi yakınsar, doğruluk artar ama hesap maliyeti yükselir.
        # AZALIRSA: üçgenleme erken kesilir, derinlik tahmini yarım kalır ve hesap hızlanır.
        gn_max_iter=5,          # Gauss-Newton iterasyon sınırı

        # 3D outlier reddi için chi-kare eşik güven seviyesi.
        # ARTARSA: eşik gevşer, daha fazla ölçüm (outlier dahil) kabul edilir -> filtre bozulabilir.
        # AZALIRSA: eşik sıkılaşır, daha çok ölçüm reddedilir, gürültüye karşı sağlam ama feature kaybı riski.
        chi2_alpha=0.99,        # 3D outlier reddi eşiği (%99 güven)

        # başlangıç yönelim (attitude) belirsizliği (radyan); statik init sırasında gözlemlenemez.
        # ARTARSA: filtre başlangıç yönelimini düzeltmeye daha açık olur ama yakınsama yavaşlar.
        # AZALIRSA: başlangıç yönelimine aşırı güvenilir, hatalıysa düzeltilemez -> kalıcı bias.
        init_att_std=0.10,      # radyan

        # başlangıç jiroskop bias belirsizliği (rad/s); statik init sırasında gözlemlenemez.
        # ARTARSA: filtre gyro bias'ını daha hızlı/serbest tahmin eder ama erken gürültüye duyarlı olur.
        # AZALIRSA: gyro bias'a fazla güvenilir, gerçek bias farklıysa yavaş düzeltilir -> yönelim drifti.
        init_bg_std=0.01,      # rad/s

        # başlangıç hız belirsizliği (m/s); statik init sırasında gözlemlenemez.
        # ARTARSA: ilk hareketteki gerçek hız tahminine daha açık olur, esneklik artar.
        # AZALIRSA: sıfır başlangıç hızına aşırı güvenilir, statik değilse hata kalıcı olur.
        init_vel_std=0.5,       # m/s

        # başlangıç ivmeölçer bias belirsizliği (m/s²); statik init sırasında gözlemlenemez.
        # ARTARSA: accel bias daha hızlı tahmin edilir ama yer çekimi ile karışabilir, yakınsama zorlaşır.
        # AZALIRSA: accel bias'a fazla güvenilir, gerçek bias farklıysa konum/hız drifti oluşur.
        init_ba_std=0.15,       # m/s²

        # başlangıç konum belirsizliği (m); statik init sırasında yer çekimi üzerinden gözlemlenebilir.
        # ARTARSA: başlangıç konumu daha serbest bırakılır, gereksiz belirsizlik eklenir.
        # AZALIRSA: başlangıç konumuna güçlü güven verilir (origin gözlemlenebilir olduğu için uygundur).
        init_pos_std=0.05,      # metre
    )

    msckf.initialize_from_static(imu.gyro[:n_static], imu.accel[:n_static])
    msckf.lock_imu_attitude = LOCK_ATTITUDE
    if LOCK_ATTITUDE:
        print("TANI: LOCK_ATTITUDE açık — update IMU attitude'una (δθ) dokunmuyor")
    msckf.chi2_enabled = CHI2_GATE
    if not CHI2_GATE:
        print("TANI: CHI2_GATE kapalı — chi-square outlier kapısı atlanıyor")
    t_init_end = imu.timestamps[n_static - 1]
    msckf_yaw_init_deg = msckf.nominal_rot.as_euler('xyz', degrees=True)[2]
    print(f"static init: {n_static} samples,  |g|={np.linalg.norm(msckf.gravity):.4f}")
    print(f"MSCKF yaw at init: {msckf_yaw_init_deg:.2f}°")

    # Front-end seçimi: METHOD sabitine bağlı olarak ORB veya KLT.
    # Parametre setleri make_tracker() içinde her tracker için ayrı tutuluyor.
    tracker = make_tracker(METHOD)
    print(f"front-end: {METHOD.upper()}  ({type(tracker).__name__})")
    print(f"odometry mode: {MODE.upper()}"
          + ("  — kamera devre dışı, saf IMU dead-reckoning" if MODE == 'imu' else ""))

    # ---- GT ALIGNMENT ----
    gt   = np.loadtxt(config.GT_LOCAL_POSE, delimiter=',')
    gt_t = gt[:, 0]
    gt_xyz = gt[:, 1:4]
    gt_rpy = gt[:, 4:7]   # roll, pitch, yaw in degrees

    # t_init_end'de GT yaw'ı al, MSCKF dünyasıyla hizala
    gt_yaw_init_deg = np.interp(t_init_end, gt_t, gt_rpy[:, 2])
    print(f"GT yaw at t_init_end: {gt_yaw_init_deg:.2f}°")

    R_gt_to_msckf = R.from_euler('z', -gt_yaw_init_deg, degrees=True).as_matrix()
    gt_xyz_aligned = gt_xyz @ R_gt_to_msckf.T   # rotate

    # GT hızı: pozisyondan sayısal türev. np.gradient kenarlarda forward/backward
    # difference, içeride merkezi fark kullanır → smooth ve aynı uzunlukta. Hız
    # frame-invariant olduğu için rotation hizalamasına gerek yok (magnitude
    # alıyoruz). Hesabı bir kere yap, event loop'ta interpolate et.
    gt_vel = np.gradient(gt_xyz, gt_t, axis=0)            # (N, 3) m/s
    gt_speed = np.linalg.norm(gt_vel, axis=1)             # (N,)   m/s magnitude

    # ---- STATIC INIT VALIDATION ----
    # initialize_from_static "araç gerçekten durağan" varsayıyor; aksi halde
    # gyro/accel bias kestirimi hareket sinyaline yapışır ve filtrenin
    # başlangıcı kalıcı olarak bozulur. GT'ye bakarak bu varsayımı sağla.
    t_imu_start = imu.timestamps[0]
    gt_pos_start = np.array([
        np.interp(t_imu_start, gt_t, gt_xyz[:, i]) for i in range(3)
    ])
    gt_pos_end = np.array([
        np.interp(t_init_end,  gt_t, gt_xyz[:, i]) for i in range(3)
    ])
    init_elapsed = t_init_end - t_imu_start
    init_drift_m = float(np.linalg.norm(gt_pos_end - gt_pos_start))
    init_avg_speed = init_drift_m / init_elapsed if init_elapsed > 0 else 0.0
    print(f"static-init validation:  GT drift = {init_drift_m*100:.2f} cm "
          f"over {init_elapsed:.2f} s   (avg speed {init_avg_speed*100:.2f} cm/s)")
    if init_avg_speed > 0.05:
        print(f"  WARN: avg_speed > 5 cm/s -- araç sabit değil, bias kestirimi bozuk olabilir")
    print(f"  initial bg  estimate = {msckf.bg}")
    print(f"  initial |g| estimate = {np.linalg.norm(msckf.gravity):.4f} m/s²")

    # ---- FIGURE SETUP (interactive) ----
    plt.ion()
    fig, (ax_img, ax_traj, ax_z) = plt.subplots(
        3, 1, figsize=(13, 13),
        gridspec_kw={'height_ratios': [1, 2.0, 1.0]},
    )
    fig.suptitle("MSCKF VIO live test — thermal + matches + trajectory", fontsize=12)

    # Top panel: side-by-side image
    ax_img.set_xticks([]); ax_img.set_yticks([])
    img_artist = ax_img.imshow(
        np.zeros((cal['height'], 2 * cal['width']), dtype=np.uint8),
        cmap='gray', vmin=0, vmax=255,
    )
    match_lines  = []
    text_artist  = ax_img.text(
        5, 18, '', color='yellow', fontsize=10,
        bbox=dict(facecolor='black', alpha=0.5, pad=2),
    )

    # Middle panel: top-down (XY) trajectory
    gt_line,   = ax_traj.plot([], [], 'r--', lw=1.4, label='GT (local pose)')
    vio_line,  = ax_traj.plot([], [], 'b-',  lw=1.4, label='MSCKF VIO')
    vio_pt     = ax_traj.scatter([], [], c='b', s=70, zorder=5, edgecolor='k')
    ax_traj.set_xlabel('x [m]'); ax_traj.set_ylabel('y [m]')
    ax_traj.set_aspect('equal')
    ax_traj.grid(alpha=0.3); ax_traj.legend(loc='best')

    # Bottom panel: vertical channel (z) vs time. The top-down XY view hides
    # the vertical runaway entirely — for a ground vehicle GT z is ~flat, so
    # any z drift is pure filter error and stands out cleanly against it.
    gt_z_line,  = ax_z.plot([], [], 'r--', lw=1.4, label='GT z')
    vio_z_line, = ax_z.plot([], [], 'b-',  lw=1.4, label='MSCKF z')
    ax_z.set_xlabel('t [s]'); ax_z.set_ylabel('z [m]')
    ax_z.grid(alpha=0.3); ax_z.legend(loc='best')

    # ---- EVENT LOOP ----
    seq = VIOSequencer(loader, imu)
    traj       = []
    prev_imu_t = t_init_end
    prev_img   = None
    frame_id   = 0
    n_predict  = 0
    n_frame    = 0
    gt_origin  = None      # set on first cam frame after init

    try:
        for kind, t, payload in seq.events():
            if t <= t_init_end:
                continue

            if kind == 'imu':
                dt = t - prev_imu_t
                if dt > 0:
                    msckf.predict(dt, payload[1:4], payload[4:7])
                prev_imu_t = t
                n_predict += 1
                continue

            # kind == 'cam'
            img = payload

            if MODE == 'vio':
                msckf.augment_state(frame_id, R_imu_cam, t_imu_cam)
                tracker.process_frame(img, frame_id)

                # Pencereden düşmek üzere olan cam_state'leri gözlemleyen aktif
                # track'leri MSCKF'e son kez göster, sonra düşür. Bu olmadan uzun
                # kesintisiz track'ler filtreye hiç girmeden kayboluyor.
                n_drop = len(msckf.cam_states) - MAX_WINDOW
                if n_drop > 0:
                    for cs in msckf.cam_states[:n_drop]:
                        tracker.marginalize_at_prune(cs.frame_id)

                n_retired = len(tracker.dead_tracks)

                # ADIM 1: update başına track sayısını sınırla. dead_tracks
                # zaten min_track_length=3 filtresinden geçti; burada en uzun
                # (en çok gözlemli, en sağlam geometrili) olanları seçiyoruz.
                upd_tracks = list(tracker.dead_tracks)
                if len(upd_tracks) > MAX_TRACKS_PER_UPDATE:
                    upd_tracks.sort(key=lambda t: len(t.frame_ids), reverse=True)
                    upd_tracks = upd_tracks[:MAX_TRACKS_PER_UPDATE]

                msckf.update(upd_tracks)
                msckf.prune_cam_states(MAX_WINDOW)
            else:
                # MODE == 'imu': kamera devre dışı. State'i yalnızca IMU
                # predict ilerletir — augment / track / update atlanır.
                n_retired = 0

            # Anchor GT to the world origin VIO believes it started at
            # (msckf.nominal_pos was [0,0,0] at end of static init).
            if gt_origin is None:
                gt_origin = R_gt_to_msckf @ np.array([
                    np.interp(t, gt_t, gt_xyz[:, 0]),
                    np.interp(t, gt_t, gt_xyz[:, 1]),
                    np.interp(t, gt_t, gt_xyz[:, 2]),
                ])

            traj.append((t, *msckf.nominal_pos))
            frame_id += 1
            n_frame  += 1

            # ── TANI: attitude'un GT'den sapması ────────────────────────
            # Update attitude'u bozuyor mu? roll/pitch yer çekimine göre
            # mutlak, yaw init-offset ile. İlk 40 karede her kare bas; hız
            # kaçışıyla attitude sapması aynı anda mı başlıyor görmek için.
            if n_frame <= 40 or n_frame % 25 == 0:
                eu = msckf.nominal_rot.as_euler('xyz', degrees=True)
                gr = float(np.interp(t, gt_t, gt_rpy[:, 0]))
                gp = float(np.interp(t, gt_t, gt_rpy[:, 1]))
                gy = float(np.interp(t, gt_t, gt_rpy[:, 2]))
                spur = wrap_yaw_deg(wrap_yaw_deg(eu[2] - msckf_yaw_init_deg)
                                    - wrap_yaw_deg(gy - gt_yaw_init_deg))
                print(f"  [att] f{n_frame:3d} t={t-t_init_end:6.1f}s  "
                      f"MSCKF rpy=({eu[0]:+6.1f},{eu[1]:+6.1f},{eu[2]:+6.1f})  "
                      f"roll_err={wrap_yaw_deg(eu[0]-gr):+6.1f} "
                      f"pitch_err={wrap_yaw_deg(eu[1]-gp):+6.1f} "
                      f"yaw_spur={spur:+6.1f}  |vel|={np.linalg.norm(msckf.nominal_vel):8.1f}",
                      flush=True)

            if n_frame % 50 == 0:
                active = tracker.active_tracks
                if active:
                    track_lens = [len(t.frame_ids) for t in active] # Bu uzunluklar, her bir aktif izginin kaç kare boyunca takip edildiğini gösterir.
                    avg_len =  np.mean(track_lens) # Ortalama uzunluk, tüm aktif izlerin takip edildiği kare sayısının ortalamasıdır. Yüksek bir ortalama uzunluk, izlerin genellikle uzun süre takip edildiğini gösterir.
                    p90_len = np.percentile(track_lens, 90) # 90. yüzdelik uzunluk, izlerin %90'ının takip edildiği kare sayısını gösterir. Yüksek bir p90 değeri, çoğu izginin uzun süre takip edildiğini gösterir.
                    n_long = sum(1 for L in track_lens if L >= 5) # 5 veya daha fazla kare boyunca takip edilen izlerin sayısı. Bu, uzun süre takip edilen izlerin sayısını gösterir.
                    ms = getattr(tracker, 'last_match_stats', None)
                    base = f"  [tracker] active={len(active)} avg_len={avg_len:.1f} p90={p90_len:.1f} long(>=5)={n_long}"
                    if ms is None:
                        print(base)
                    elif 'after_desc' in ms:
                        # ORB pipeline: desc → disp → ransac
                        print(f"{base}  match: desc={ms['after_desc']} → disp={ms['after_disp']} → ransac={ms['after_ransac']}  "
                              f"(disp_drop={ms.get('dropped_disp', '?')}, ransac_drop={ms.get('dropped_ransac', '?')})")
                    elif 'tracked' in ms:
                        # KLT pipeline: tracked → fb → disp → ransac
                        print(f"{base}  match: LK={ms['tracked']} → fb={ms['after_fb']} → disp={ms['after_disp']} → ransac={ms['after_ransac']}")
                    else:
                        print(base)
                euler = msckf.nominal_rot.as_euler('xyz', degrees=True)
                # GT yaw at current time vs init → ground-truth yaw change
                gt_yaw_now    = float(np.interp(t, gt_t, gt_rpy[:, 2]))
                gt_dyaw       = wrap_yaw_deg(gt_yaw_now - gt_yaw_init_deg)
                msckf_dyaw    = wrap_yaw_deg(euler[2]  - msckf_yaw_init_deg)
                spurious_yaw  = wrap_yaw_deg(msckf_dyaw - gt_dyaw)
                print(f"  f{n_frame:4d}  pos={msckf.nominal_pos}  vel={msckf.nominal_vel}  "
                    f"bg={msckf.bg}  ba={msckf.ba}  rpy={euler}")
                print(f"        yaw  GT_Δ={gt_dyaw:+7.2f}°  MSCKF_Δ={msckf_dyaw:+7.2f}°  "
                    f"spurious={spurious_yaw:+7.2f}°")


            # ---- LIVE REDRAW ----
            if n_frame % VIZ_EVERY == 0 and prev_img is not None:
                # Side-by-side image with match lines
                combined = np.hstack([prev_img, img])
                img_artist.set_data(combined)
                w = prev_img.shape[1]

                for ml in match_lines:
                    ml.remove()
                match_lines.clear()

                n_drawn = 0
                for tr in tracker.active_tracks:
                    if len(tr.keypoints) >= 2 and tr.frame_ids[-1] == frame_id - 1:
                        p1, p2 = tr.keypoints[-2], tr.keypoints[-1]
                        line, = ax_img.plot(
                            [p1[0], p2[0] + w], [p1[1], p2[1]],
                            'g-', lw=0.4, alpha=0.55,
                        )
                        match_lines.append(line)
                        n_drawn += 1

                text_artist.set_text(
                    f"frame {n_frame:4d}   matches: {n_drawn:3d}   "
                    f"retired: {n_retired:3d}   win: {len(msckf.cam_states)}"
                )

                # Trajectory: VIO is already anchored at origin; shift GT.
                traj_arr = np.array(traj)
                vio_line.set_data(traj_arr[:, 1], traj_arr[:, 2])
                vio_pt.set_offsets([[traj_arr[-1, 1], traj_arr[-1, 2]]])

                gt_mask = gt_t <= t
                gt_xy   = gt_xyz_aligned[gt_mask, :2] - gt_origin[:2]
                gt_line.set_data(gt_xy[:, 0], gt_xy[:, 1])

                # GT velocity şu anki t'de — sayısal türev tablosundan interpolate.
                gt_speed_now = float(np.interp(t, gt_t, gt_speed))
                vio_speed = float(np.linalg.norm(msckf.nominal_vel))

                ax_traj.relim(); ax_traj.autoscale_view()
                ax_traj.set_title(
                    f"VIO pos = ({msckf.nominal_pos[0]:+.2f}, "
                    f"{msckf.nominal_pos[1]:+.2f}, {msckf.nominal_pos[2]:+.2f}) m   "
                    f"VIO vel = {vio_speed:.2f} m/s ({vio_speed * 3.6:.2f} km/h)   "
                    f"GT vel = {gt_speed_now:.2f} m/s ({gt_speed_now * 3.6:.2f} km/h)"
                )

                # z vs time — VIO ile GT, ilk kamera karesine göre relatif.
                # z, yaw hizalamasından (z ekseni rotasyonu) etkilenmez.
                t0 = traj_arr[0, 0]
                vio_z_line.set_data(traj_arr[:, 0] - t0, traj_arr[:, 3])
                gt_z_line.set_data(gt_t[gt_mask] - t0,
                                   gt_xyz_aligned[gt_mask, 2] - gt_origin[2])
                ax_z.relim(); ax_z.autoscale_view()

                plt.pause(0.001)

            prev_img = img

    except KeyboardInterrupt:
        print("\n[interrupted]")

    # ---- PERSIST ----
    traj = np.array(traj)
    print(f"\nmode={MODE}   processed: {n_predict} IMU predicts, {n_frame} cam frames")
    if len(traj):
        print(f"final pos: ({traj[-1, 1]:+.3f}, {traj[-1, 2]:+.3f}, {traj[-1, 3]:+.3f})")
        os.makedirs('results', exist_ok=True)
        traj_path = f'results/msckf_{MODE}_trajectory_{clahe_mode}.txt'
        png_path  = f'results/msckf_{MODE}_live_final_{clahe_mode}.png'
        np.savetxt(traj_path, traj, header='t x y z')
        plt.savefig(png_path, dpi=120)
        print(f"saved → {traj_path}")
        print(f"        {png_path}")

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run()
