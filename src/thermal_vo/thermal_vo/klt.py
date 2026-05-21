# KLT (Kanade-Lucas-Tomasi) tabanlı özellik takipçisi.
#
# ORB ile karşılaştırma (termal görüntüler için):
#   + Descriptor yok — doğrudan intensity gradient kullanır. BRIEF binary
#     testlerinin termal'de tutarsız kalması problemi yok.
#   + Sub-pixel doğruluk doğal olarak gelir.
#   − Brightness constancy varsayımı termal'de tam tutmaz (sıcaklık zamanla
#     değişir); FB-consistency check bunu yakalamaya çalışır.
#   − Bir track kaybolunca tekrar yakalanamaz (descriptor re-match yok).
#     Refill ile sürekli yeni track açmak gerekir.
#
# Public API ORBTracker ile birebir aynı tutuldu:
#   .process_frame(image, frame_id)
#   .active_tracks      : list[FeatureTrack]
#   .dead_tracks        : list[FeatureTrack]
#   .marginalize_at_prune(frame_id)
#   .last_match_stats   : dict (tanı için, her frame güncellenir)
# Bu sayede MSCKF test scriptinde tek satırlık import değişikliği ile geçilir.


import cv2
import numpy as np


class FeatureTrack:
    """
    Tek bir feature noktasının yaşam döngüsünü temsil eder.

    MSCKF.update() bu objeden iki şey okur:
      - frame_ids: hangi cam_state'lerde gözlemlendi
      - keypoints: her cam_state'te hangi pixel koordinatında

    ORB versiyonundan farklı olarak burada `descriptor` field'ı YOK —
    KLT pixel-level optical flow yapar, descriptor matching kullanmaz.
    """
    def __init__(self, track_id, frame_id, kp):
        self.track_id   = track_id
        self.frame_ids  = [frame_id]
        self.keypoints  = [kp]      # (x, y) tuple, pixel cinsinden


class KLTTracker:
    def __init__(self,
                 n_features=1000,
                 fb_eps=1.5,
                 ransac_thresh=2.0,
                 min_track_length=3,
                 lk_win_size=(25, 25),
                 lk_max_level=3,
                 quality_level=0.01,
                 min_distance=8.0,
                 max_pixel_displacement=25.0,
                 refill_threshold=None):
        """
        Args:
            n_features: Aynı anda tutulmaya çalışılan maksimum track sayısı.
                        Bu sayı kadar Shi-Tomasi köşesi bootstrap ve refill'de
                        hedeflenir.

            fb_eps:     Forward-backward round-trip toleransı (piksel).
                        Bir track frame1 → frame2 → frame1 yolculuğu yaptığında
                        başladığı yere bu mesafeden fazla uzakta dönerse silinir.
                        1.5 px sıkı; termal'de gerekirse 2.0'a gevşetilir.

            ransac_thresh: F-matrix RANSAC inlier eşiği (piksel). Lowe ratio +
                        FB-consistency'yi geçmiş ama epipolar geometri ihlal
                        eden geometrik outlier'ları yakalar.

            min_track_length: Bir track'in dead_tracks'e (MSCKF update'e) konması
                        için gereken minimum gözlem sayısı. ORB ile aynı.

            lk_win_size: Lucas-Kanade penceresi (pixel). Bu pencere içindeki
                        tüm piksellerin "aynı (u,v) flow vektörüyle hareket
                        ettiği" varsayımıyla least-squares çözüm yapılır.
                        Büyük pencere → structure matrix daha kuvvetli ama
                        kenar artefaktları artar. (25, 25) termal'de
                        smooth gradient'ler için (21, 21)'den biraz daha
                        kararlı.

            lk_max_level: Image pyramid level sayısı (0 dahil). max_level=3
                        toplam 4 seviye, en kabası 1/8 çözünürlük. ~30-40 px
                        frame-içi hareket bu sayede ele alınabilir. Hızlı
                        rotasyonlarda kritik.

            quality_level: Shi-Tomasi göreli kalite eşiği. Bulunan en güçlü
                        köşenin min(λ₁, λ₂)'sinin bu oranı kadar güçlü olan
                        köşeler kabul edilir. Termal düşük-kontrast olduğu
                        için 0.005-0.01 arası tipik; daha yüksek (0.05+)
                        çoğu termal frame'de yeterli köşe bulamaz.

            min_distance: İki Shi-Tomasi köşesi arasındaki minimum piksel
                        mesafesi (non-maximum suppression yarıçapı). Spatial
                        diversity sağlar; cluster oluşumunu önler. ORB'taki
                        grid mantığının KLT karşılığı.

            max_pixel_displacement: Bir track'in frame-arası izin verilen
                        maksimum öklid kayması (piksel). FB-consistency'yi
                        geçen ama fwd/bwd flow'un ikisi de yanlış ama tutarlı
                        bir yere yakınsadığı durumları yakalar — ORB'ta
                        eklediğimiz motion-prior gate ile aynı mantık.

            refill_threshold: Aktif track sayısı bu eşiğin altına düştüğünde
                        Shi-Tomasi ile yeni köşe ekleyip n_features'a kadar
                        doldurur. None → n_features // 2 (varsayılan).
        """
        self.n_features             = n_features
        self.fb_eps                 = fb_eps
        self.ransac_thresh          = ransac_thresh
        self.min_track_length       = min_track_length
        self.lk_win_size            = lk_win_size
        self.lk_max_level           = lk_max_level
        self.quality_level          = quality_level
        self.min_distance           = min_distance
        self.max_pixel_displacement = max_pixel_displacement
        self.refill_threshold       = (refill_threshold if refill_threshold is not None
                                       else n_features // 2)

        # cv2.calcOpticalFlowPyrLK için sözlük formatlı parametreler. Iterasyon
        # kriteri: ya 30 iterasyon ya da bir adımda Δ < 0.01 piksel.
        self.lk_params = dict(
            winSize=lk_win_size,
            maxLevel=lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self.next_track_id = 0    # Her yeni track'e benzersiz ID veren counter
        self.active_tracks = []   # Şu anda takip edilen track'ler
        self.dead_tracks   = []   # Bu frame'de retire edilenler (MSCKF'e gider)
        self.prev_image    = None # Bir önceki frame, LK için referans

        # Tanı amaçlı — her process_frame sonunda doldurulur. Test scriptinde
        # her ~50 frame'de bir yazdırılıp hangi filtreden ne kadar düştüğüne
        # bakarız.
        self.last_match_stats = {
            'tracked':      0,   # LK'nın takibe alabildiği track sayısı
            'after_fb':     0,   # FB-consistency'den geçen
            'after_disp':   0,   # max_pixel_displacement'tan geçen
            'after_ransac': 0,   # F-matrix RANSAC'tan geçen
        }


    # ------------------------------------------------------------------
    def _detect_corners(self, image, mask=None):
        """
        Shi-Tomasi köşe detektörü. "İyi takip edilebilecek" pikselleri bulur.

        İçeride yapılan:
          1. Image gradient (Sobel ile Iₓ, Iᵧ).
          2. Her piksel için 2×2 structure matrix M = [[ΣIₓ², ΣIₓIᵧ],
                                                       [ΣIₓIᵧ, ΣIᵧ²]].
             Σ küçük bir komşulukta (cv2 default 3×3).
          3. Her piksel için min(λ₁, λ₂) hesapla; yüksek olanları köşe say.
             min(λ₁, λ₂) >= quality_level × max(min(λ_i)_over_image) olanlar.
          4. min_distance yarıçaplı non-maximum suppression uygula.
          5. En kuvvetli en fazla n_features adet köşeyi dön.

        Args:
            image: uint8 grayscale (preprocessed thermal frame).
            mask:  opsiyonel uint8 maske; 0 olan bölgelerde köşe ARANMAZ.
                   Refill'de aktif track'lerin etrafını maskelemek için.

        Returns:
            (N, 2) float32 numpy array — her satır bir köşenin (x, y) pixel
            koordinatı. N ≤ n_features. Hiç köşe bulunamazsa (0, 2) boş array.
        """
        corners = cv2.goodFeaturesToTrack(
            image,
            maxCorners=self.n_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
        )
        # cv2 None döndürebilir (görüntüde hiç köşe bulamazsa): tek tip
        # davranış için her zaman 2D float32 array dön.
        if corners is None:
            return np.empty((0, 2), dtype=np.float32)
        # cv2 (N, 1, 2) shape verir — (N, 2)'ye düzleştir.
        return corners.reshape(-1, 2).astype(np.float32)


    # ------------------------------------------------------------------
    def _refill_mask(self, image_shape):
        """
        Refill sırasında Shi-Tomasi'nin "buralarda köşe arama" diyebilmesi için
        aktif track'lerin etrafını maskeleyen uint8 array üretir.

        Maske mantığı:
          - Her yer 255'le başlar (= "ara").
          - Her aktif track'in SON keypoint'i etrafında `min_distance` yarıçaplı
            dolu disk çizilir, değer 0 (= "arama").

        radius = min_distance seçimi: zaten "iki Shi-Tomasi köşesi birbirinden
        min_distance uzakta olmalı" kuralı işliyor; aynı kuralı mevcut track'le
        yeni köşe arasında uygulamak tutarlılık sağlar.

        Args:
            image_shape: (H, W) tuple — image.shape[:2]
        Returns:
            (H, W) uint8 mask
        """
        mask = np.full(image_shape, 255, dtype=np.uint8)
        radius = int(round(self.min_distance))
        for tr in self.active_tracks:
            x, y = tr.keypoints[-1]
            cv2.circle(mask, (int(round(x)), int(round(y))), radius, 0, thickness=-1)
        return mask


    # ------------------------------------------------------------------
    def _retire_track(self, track):
        """
        Bir track'in takipten çıkarılması (ölmesi) durumunda yapılacak işlemler.
        MSCKF.update() yalnızca min_track_length koşulunu sağlayan ölü track'lerle
        ilgilenir — daha kısa track'lerden iyi triangulation çıkmadığı için
        baştan filtreliyoruz.

        Args:
            track: retire edilecek FeatureTrack objesi
        """
        if len(track.frame_ids) >= self.min_track_length:
            self.dead_tracks.append(track)

    def marginalize_at_prune(self, frame_id_to_drop):
        """
        Sliding-window pencereden bir cam_state düşmek üzereyken çağrılır.
        O cam_state'i gözlemleyen tüm AKTİF track'ler, sonraki frame'lerde
        artık bağlamlarını kaybedecek (eski gözlemleri MSCKF'e bağlanamaz)
        — bu yüzden onları şimdi dead_tracks'e taşıyıp MSCKF.update()'in
        son kez görmesini sağlıyoruz. Aksi halde uzun, kesintisiz takip
        edilen feature'lar (en bilgili olanları) filtreye hiç girmeden
        kayboluyor.

        Klasik MSCKF "feature marginalization at pruning" davranışı.
        Tracker tipinden (ORB/KLT) bağımsız çalışır.
        """
        keep_active = []
        for tr in self.active_tracks:
            if frame_id_to_drop in tr.frame_ids:
                if len(tr.frame_ids) >= self.min_track_length:
                    self.dead_tracks.append(tr)
                # min_track_length altındaki track'ler sessizce düşer.
            else:
                keep_active.append(tr)
        self.active_tracks = keep_active


    # ------------------------------------------------------------------
    def process_frame(self, image, frame_id):
        """
        Her yeni frame'de çağrılır. Active track listesini günceller,
        FB-consistency + max_pixel_displacement + F-matrix RANSAC zincirinden
        geçemeyenleri retire eder, gerekirse yeni Shi-Tomasi köşeleriyle
        doldurur.

        Args:
            image:    yeni frame (uint8 grayscale, preprocessed)
            frame_id: bu frame'in benzersiz ID'si (MSCKF cam_state ile aynı)
        """
        # Her frame başında dead_tracks'i sıfırla. MSCKF.update() yalnız bu
        # frame'de retire olanlarla ilgilenir — geçmiş frame'lerinkiler zaten
        # işlenmiştir.
        self.dead_tracks = []

        # Tanı counter'larını sıfırla.
        self.last_match_stats = {
            'tracked':      0,
            'after_fb':     0,
            'after_disp':   0,
            'after_ransac': 0,
        }

        # ---- PHASE A: bootstrap ----
        # İki giriş koşulu:
        #   1) Hiç önceki frame yok (ilk çağrı, prev_image is None).
        #   2) Önceki frame vardı ama tüm track'ler kaybolmuş (active boş).
        # İkinci durumda da sıfırdan kuruyoruz — refill mantığı işlemediği
        # için pencere boş kalmasın. Bu frame'de takip edilecek bir önceki
        # feature olmadığından LK fazlarına girmiyoruz, doğrudan return.
        if self.prev_image is None or len(self.active_tracks) == 0:
            corners = self._detect_corners(image)
            for c in corners:
                self.active_tracks.append(
                    FeatureTrack(self.next_track_id, frame_id,
                                 (float(c[0]), float(c[1])))
                )
                self.next_track_id += 1
            self.prev_image = image
            return

        # ---- PHASE B: Pyramidal LK forward ----
        # Active track'lerin önceki frame'deki son keypoint'leri = prev_pts.
        # Bu noktaların yeni frame'deki konumlarını cv2.calcOpticalFlowPyrLK
        # bulur. (N, 1, 2) shape — cv2'nin beklediği layout.
        prev_pts = np.array(
            [t.keypoints[-1] for t in self.active_tracks],
            dtype=np.float32
        ).reshape(-1, 1, 2)

        # next_pts: (N, 1, 2) — kestirilen yeni konumlar
        # status_fwd: (N, 1) uint8 — 1 = LK takip etti, 0 = başarısız
        # err: (N, 1) float — residual (kullanmıyoruz, FB-check'imiz var)
        next_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_image, image, prev_pts, None, **self.lk_params
        )

        # status_fwd'i 1D bool array'e çevir; sonraki fazlar bool maskeleriyle
        # çalışacak.
        status_fwd = status_fwd.ravel().astype(bool)
        self.last_match_stats['tracked'] = int(status_fwd.sum())

        # Intermediate state — Phase C-F için self üzerinde tutuluyor.
        # Tüm fazlar yazıldıktan sonra (Blok 10) tek process_frame çağrısında
        # birleştiklerinde bu üç attribute local değişkene çekilebilir; şu an
        # inkremental smoke test için açık.
        self._lk_prev_pts = prev_pts.reshape(-1, 2)
        self._lk_next_pts = next_pts.reshape(-1, 2)
        self._lk_status   = status_fwd

        # Not: self.prev_image GÜNCELLENMİYOR. Phase F bunu yapacak.
        # Tracks da hala eski keypoint'lerinde — Phase F güncelleyecek.

        # ---- PHASE C: Backward LK + Forward-Backward Consistency ----
        # Asimetrik LK fail'lerini yakalama mekanizması: forward LK'nın
        # bulduğu next_pts'i ters yönde takip et (image → prev_image), elde
        # edilen back_pts orijinal prev_pts'e yakın ÇIKMALI. Aperture problem,
        # occlusion, brightness değişimi → bu round-trip kapanmaz, track elenir.
        back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            image, self.prev_image,
            self._lk_next_pts.reshape(-1, 1, 2),
            None,
            **self.lk_params
        )
        status_bwd = status_bwd.ravel().astype(bool)
        back_pts = back_pts.reshape(-1, 2)

        # Round-trip hatası: ||prev_pts - back_pts||. Güvenilir track'lerde
        # alt-piksel mesafede olmalı.
        fb_err = np.linalg.norm(self._lk_prev_pts - back_pts, axis=1)

        # Üçlü AND: fwd OK + bwd OK + round-trip toleransı içinde
        keep_fb = self._lk_status & status_bwd & (fb_err < self.fb_eps)

        self.last_match_stats['after_fb'] = int(keep_fb.sum())
        self._lk_keep_fb = keep_fb   # Phase D-F için stash

        # ---- PHASE D: in-bounds + max_pixel_displacement ----
        # İki basit gate:
        #   (i)  LK next_pt görüntü içinde mi? Sınırı aşan noktalar bir sonraki
        #        cv2 çağrısında bozulur ve F-matrix RANSAC'ı kirletirler.
        #   (ii) FB-pass etmiş ama "fiziksel olarak makul olmayan" büyük flow'ları
        #        ele (simetrik LK fail — fwd ve bwd ikisi de yanlış ama tutarlı
        #        bir yere yakınsamış). ORB'taki max_pixel_displacement gate ile
        #        aynı mantık.
        h, w = image.shape[:2]
        np_pts = self._lk_next_pts
        in_bounds = (np_pts[:, 0] >= 0) & (np_pts[:, 0] < w) & \
                    (np_pts[:, 1] >= 0) & (np_pts[:, 1] < h)

        flow = np.linalg.norm(np_pts - self._lk_prev_pts, axis=1)
        within_disp = flow < self.max_pixel_displacement

        keep_disp = self._lk_keep_fb & in_bounds & within_disp

        self.last_match_stats['after_disp'] = int(keep_disp.sum())
        self._lk_keep_disp = keep_disp

        # ---- PHASE E: F-matrix RANSAC ----
        # Son savunma hattı: pixel-space'te tutarlı görünen ama 3D'de hareketli
        # objelerden (yaya, geçen araç, sallanan dal) veya subtle LK drift'inden
        # gelen track'leri ele. Two-view geometry varsayımı: tüm match'ler aynı
        # kameranın ego-motion'ından gelir; ihlal edenler epipolar constraint'i
        # bozar ve RANSAC bunları outlier sayar.
        pts1 = self._lk_prev_pts[keep_disp]
        pts2 = self._lk_next_pts[keep_disp]

        if len(pts1) >= 8:   # F-matrix tahmini için 8 minimum nokta
            F, mask = cv2.findFundamentalMat(
                pts1, pts2, cv2.FM_RANSAC,
                self.ransac_thresh, 0.99,
            )
            if F is not None and mask is not None:
                ransac_inliers = mask.ravel().astype(bool)
            else:
                # F estimation çöktü — hepsini outlier say (güvenli taraf)
                ransac_inliers = np.zeros(len(pts1), dtype=bool)
        else:
            # 8 noktadan az → F-matrix tanımlanamaz, RANSAC reddedilir.
            ransac_inliers = np.zeros(len(pts1), dtype=bool)

        # RANSAC inlier mask'i (keep_disp subset'i üzerinden) full N uzunluğa
        # geri yay. keep_disp[i]=True olan i'lere ransac_inliers sırayla denk gelir.
        keep_ransac = np.zeros_like(keep_disp)
        keep_ransac[keep_disp] = ransac_inliers

        self.last_match_stats['after_ransac'] = int(keep_ransac.sum())
        self._lk_keep_ransac = keep_ransac

        # ---- PHASE F: survivors update + retire + refill + commit ----
        # Şu ana kadar state değişmedi — Phase B-E sadece intermediate
        # array'ler hesapladı. Burada gerçek mutation'lar yapılıyor:
        #   1) Hayatta kalan track'leri yeni keypoint + frame_id ile ilerlet.
        #   2) Düşen track'leri retire et (min_track_length koşulunu sağlarsa
        #      dead_tracks'e gider).
        #   3) Aktif sayı refill_threshold altına düştüyse Shi-Tomasi ile
        #      yeni köşeler ekle.
        #   4) prev_image'ı yeni frame ile değiştir.
        #   5) Intermediate _lk_* attribute'larını sil (clean state).
        new_active = []
        for i, tr in enumerate(self.active_tracks):
            if keep_ransac[i]:
                x, y = self._lk_next_pts[i]
                tr.frame_ids.append(frame_id)
                tr.keypoints.append((float(x), float(y)))
                new_active.append(tr)
            else:
                self._retire_track(tr)
        self.active_tracks = new_active

        # Refill — eşik altına düştüysek doldur.
        if len(self.active_tracks) < self.refill_threshold:
            mask = self._refill_mask(image.shape[:2])
            new_corners = self._detect_corners(image, mask=mask)
            needed = self.n_features - len(self.active_tracks)
            for c in new_corners[:needed]:
                self.active_tracks.append(
                    FeatureTrack(self.next_track_id, frame_id,
                                 (float(c[0]), float(c[1])))
                )
                self.next_track_id += 1

        # prev_image'ı bu frame'e taşı — sonraki çağrı için referans.
        self.prev_image = image

        # Intermediate state'i temizle (clean exit + bir sonraki çağrıda
        # yanlış kullanım koruması).
        del self._lk_prev_pts, self._lk_next_pts, self._lk_status
        del self._lk_keep_fb, self._lk_keep_disp, self._lk_keep_ransac