# ORB

import cv2
import numpy as np

class FeatureTrack:
    """
    Tek bir feature noktasının yaşam döngüsünü temsil eder.

    MSCKF.update() bu objeden iki şey okur:
      - frame_ids: hangi cam_state'lerde gözlemlendi
      - keypoints: her cam_state'te hangi pixel koordinatında

    Başka hiçbir şey gerekmiyor — pose, descriptor, ID gibi şeyler
    MSCKF'in işine yaramaz. Sadece "hangi frame'de hangi piksel".
    descriptor'u burada tutuyoruz çünkü TRACKER kendi içinde matching
    için kullanıyor (MSCKF'e gitmiyor).
    """
    def __init__(self, track_id, frame_id, kp, desc):
        self.track_id   = track_id
        self.frame_ids  = [frame_id]
        self.keypoints  = [kp]      # (x, y) tuple, pixel cinsinden
        self.descriptor = desc      # ORB BRIEF descriptor (32 byte / 256 bit)



class ORBTracker:
    def __init__(self, n_features=800,
                 grid_rows=4,
                 grid_cols=4,
                 ratio_thresh=0.75,
                 ransac_thresh=2.0,
                 min_track_length=3,
                 fast_threshold=8,
                 edge_threshold=10,
                 max_pixel_displacement=40.0):

        self.n_features       = n_features
        self.grid_rows        = grid_rows
        self.grid_cols        = grid_cols
        self.ratio_thresh     = ratio_thresh
        self.ransac_thresh    = ransac_thresh
        self.min_track_length = min_track_length   # MSCKF'nin güncellemesinde kullanılmak üzere bir parçacığın "ölü" olarak
        # emekli edilmesi için gereken minimum gözlem sayısı

        # Frame'ler arası maksimum izin verilen piksel kayması. Termal'de homojen
        # bölgeler RANSAC F-matrix'i degenerate ediyor (8 nokta da sahte bir
        # geometriye otururlarsa inlier sayılıyorlar). Motion prior olarak
        # descriptor-level filtre ile RANSAC arasına konuyor: prev_kp ile
        # curr_kp arasındaki Öklid mesafesi bu eşiği aşan match'ler atılıyor.
        # ~10 fps termal'de aracın tipik frame-içi kayması 5-20 px, 40 px makul
        # bir üst sınır. None verirsen kontrol kapanır.
        self.max_pixel_displacement = max_pixel_displacement

        self.n_per_cell = max(1, n_features // (grid_rows * grid_cols)) # Bu satır, görüntüyü hücrelere bölerek her hücredeki en güçlü ORB özelliklerini 
        # seçmek için kullanılan bir parametre hesaplar. 
        # Görüntü, grid_rows x grid_cols boyutlarında hücrelere bölünür ve her hücrede n_per_cell kadar ORB özelliği seçilir. 
        # Bu, özelliklerin görüntü genelinde daha eşit dağılmasını sağlar.
        # Örneğin, n_features=800, grid_rows=4 ve grid_cols=4 ise, toplam 16 hücre olur ve her hücrede en fazla 50 özellik seçilir (800 // 16 = 50).

        self.orb = cv2.ORB_create(nfeatures=self.n_per_cell * 2, # ORB'nin her hücredeki özellik sayısı için 2 katı kadar özellik çıkarmasını sağlar.
                                  fastThreshold=fast_threshold, # FAST köşe algılama algoritması için kullanılan bir parametredir. Bu değer, köşe olarak kabul edilecek noktaların belirlenmesinde kullanılan bir eşik değeridir. 
                                  edgeThreshold=edge_threshold # ORB'nin kenar bölgelerinde özellik algılamasını engellemek için kullanılan bir parametredir. 
                                  )  

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING) # ORB özellikleri ikili (binary) olduğu için Hamming mesafe ölçütü kullanılarak brute-force eşleştirici oluşturulur. crossCheck = True olarak ayarlanmaz çünkü Lowe ratio testi ve mutual check manuel olarak uygulanır.

        self.next_track_id = 0    # Takip edilen özelliklere benzersiz bir kimlik atamak için kullanılan bir sayaçtır. Her yeni özellik takip edildiğinde, bu sayaç artırılır ve yeni takip edilen özelliğe atanır.
        self.active_tracks = []   # Şu anda takip edilen noktaların listesi
        self.dead_tracks   = []   # Bu çerçevede eşleşme kaybedilen ve MSCKF güncellemesinde kullanılacak parçacıkların listesi

    def _extract_grid_orb(self, image):
        """ Görüntüyü bir ızgaraya bölerek her hücredeki en güçlü ORB özelliklerini seçer ve bunları tam görüntü koordinatlarına göre ayarlar.
        Bu yöntem, özelliklerin görüntü genelinde daha eşit dağılmasını sağlar."""
        h, w = image.shape  # Görüntünün yüksekliği (h) ve genişliği (w) alınır.
        all_kp, all_desc = [], [] # Tüm anahtar noktaları (keypoints) ve açıklayıcıları (descriptors) depolamak için boş listeler oluşturulur.
        
        # kplerin yapısı şöyledir: cv2.KeyPoint nesneleri, her biri bir anahtar nokta hakkında bilgi içerir (örneğin, konum, ölçek, açı, vb.).
        # Örnek değer: [<KeyPoint 0x7f8c9c0e5b80>, <KeyPoint 0x7f8c9c0e5c00>, <KeyPoint 0x7f8c9c0e5c80>, ...]
        # desc'ler ise, her anahtar nokta için ORB tarafından hesaplanan ikili açıklayıcıları içeren bir NumPy dizisidir.
        # Örnek değer: [[0 1 0 0 1 0 1 0 1 0 0 1 0 0 1 0], [1 0 1 0 0 1 0 1 0 1 1 0 1 0 0 1], ...]

        for r in range(self.grid_rows):
            for c in range(self.grid_cols): 
                
                # Örneğin 4x4'lük bir ızgara için, r ve c sırasıyla 0'dan 3'e kadar döner. Görsel boyutu h=480, w=640 ise, her hücre 120x160 piksel olur.

                y0 = r * h // self.grid_rows # Her hücrenin başlangıç ve bitiş y koordinatları hesaplanır.
                y1 = (r + 1) * h // self.grid_rows 
                x0 = c * w // self.grid_cols # Her hücrenin başlangıç ve bitiş x koordinatları hesaplanır.
                x1 = (c + 1) * w // self.grid_cols
                cell = image[y0:y1, x0:x1] # Görüntü, hesaplanan koordinatlara göre hücrelere bölünür.

                kp, desc = self.orb.detectAndCompute(cell, None) # ORB algoritması kullanılarak her hücredeki anahtar noktalar (keypoints) ve açıklayıcılar (descriptors) hesaplanır.
                if desc is None or len(kp) == 0:
                    continue

                # Keep only the strongest keypoints per cell (by Harris/FAST response).
                if len(kp) > self.n_per_cell:
                    order = np.argsort([-k.response for k in kp])[:self.n_per_cell] 
                    
                    # Her hücredeki anahtar noktaların Harris/FAST tepkisine göre sıralanır ve sadece n_per_cell kadar güçlü anahtar nokta seçilir. -k yazılma sebebi, np.argsort'un küçükten büyüğe sıralama yapmasıdır. En güçlü anahtar noktalar, en yüksek Harris/FAST tepkisine sahip olanlardır, bu yüzden negatif işaret kullanılarak sıralama tersine çevrilir. Burada order indisleri içerir. Örneğin, order = [2, 0, 1] ise, kp[2] en güçlü anahtar nokta, kp[0] ikinci güçlü ve kp[1] üçüncü güçlü anahtar noktadır. [:self.n_per_cell] ifadesi ise sadece n_per_cell kadar anahtar nokta seçilmesini sağlar.
                    
                    kp   = [kp[i] for i in order] # Sadece seçilen güçlü anahtar noktalar (keypoints) ve açıklayıcılar (descriptors) alınır.
                    # Şöyle çalışır: order'da kpleri sıralamıştık. kp[i] ile sıralanmış kp'leri alırız. Örneğin, order = [2, 0, 1] ise, kp[2], kp[0], kp[1] sırasıyla alınır.
                    desc = desc[order]

                # Adjust keypoint coordinates to the full image frame of reference.
                for k in kp:
                    k.pt = (k.pt[0] + x0, k.pt[1] + y0) # Her anahtar noktanın koordinatları, hücre içindeki konumlarından tam görüntü koordinatlarına dönüştürülür. Örneğin, eğer bir anahtar nokta hücre içinde (10, 15) konumunda ise ve hücre x0=160, y0=120 ile başlıyorsa, tam görüntü koordinatları (170, 135) olur.

                all_kp.extend(kp) # Seçilen anahtar noktalar (keypoints) ve açıklayıcılar (descriptors) tüm hücreler için birleştirilir. extend() metodu, listelerin birbirine eklenmesini sağlar. Örneğin, all_kp = [kp1, kp2] ve kp = [kp3, kp4] ise, all_kp.extend(kp) sonrası all_kp = [kp1, kp2, kp3, kp4] olur. 
                all_desc.append(desc) # desc'ler NumPy dizisi olduğu için append() kullanılır. Örneğin, all_desc = [desc1, desc2] ve desc = desc3 ise, all_desc.append(desc) sonrası all_desc = [desc1, desc2, desc3] olur.

        if len(all_kp) == 0:
            return [], None

        all_desc = np.vstack(all_desc) # Tüm hücrelerden elde edilen açıklayıcılar (descriptors) tek bir NumPy dizisi haline getirilir. np.vstack() fonksiyonu, verilen dizileri dikey olarak (satır bazında) birleştirir. Örneğin, all_desc = [desc1, desc2, desc3] ise, all_desc = np.vstack(all_desc) sonrası all_desc tek bir NumPy dizisi olur ve desc1, desc2, desc3'ün satırları bu yeni dizide ardışık olarak yer alır.

        return all_kp, all_desc
    
    def _match_descriptors(self, prev_desc, curr_desc):
        """
        Önceki frame'deki track descriptor'leri ile yeni frame'in
        descriptor'leri arasında çift-yönlü, ratio-gated matching.

        Args:
            prev_desc: (N_prev, 32) uint8 — active_tracks descriptor'leri
            curr_desc: (N_curr, 32) uint8 — yeni frame'in ORB descriptor'leri

        Returns:
            good_matches: list[cv2.DMatch], her biri prev→curr eşleşmesi
        """
        # En az 2 candidate gerekli (k=2 için), aksi halde Lowe ratio uygulayamayız.
        if prev_desc is None or curr_desc is None or len(curr_desc) < 2 or len(prev_desc) < 2:
            return []

        # 1) Forward (prev → curr, k=2): Lowe ratio için 2 NN lazım.
    
        matches_fwd = self.matcher.knnMatch(prev_desc, curr_desc, k=2)

        # 2) Backward (curr → prev, k=1): mutual check için 1 NN yeter.
        matches_bwd = self.matcher.knnMatch(curr_desc, prev_desc, k=1)
 
        # best_bwd[curr_idx] = en yakın prev_idx. Hızlı erişim için dict.
        best_bwd = {}
        for pair in matches_bwd:
            if len(pair) >= 1:
                best_bwd[pair[0].queryIdx] = pair[0].trainIdx

        # 3) + 4) Lowe ratio + mutual nearest neighbor
        good_matches = []
        for pair in matches_fwd:
            if len(pair) != 2: 
                continue
            m, n = pair  # m: 1-NN, n: 2-NN

            # pair yapısı şöyle: [DMatch(0, 10, 0, 0.5), DMatch(0, 20, 0, 0.8)] o zaman m = DMatch(0, 10, 0, 0.5) ve n = DMatch(0, 20, 0, 0.8) olur. 
            # Burada m.queryIdx = 0, m.trainIdx = 10, m.distance = 0.5 gibi değerlere sahiptir. 
            # queryIdx şudur: prev_desc içindeki descriptor'ün indeksi, trainIdx şudur: curr_desc içindeki descriptor'ün indeksi, distance ise iki descriptor arasındaki Hamming mesafesidir. Bunlar eşleşen descriptor çiftleri hakkında bilgi verir. k-1'inci framedeki descriptor prev_desc[m.queryIdx] ve k'inci framedeki descriptor curr_desc[m.trainIdx] arasındaki mesafe m.distance ile ifade edilir. 
            
            # Her zaman önceki frame descriptor'ü prev_desc ve m.queryIdx ile, yeni frame descriptor'ü curr_desc ve m.trainIdx ile erişilir. n ise ikinci en yakın eşleşmeyi temsil eder ve benzer şekilde n.queryIdx ve n.trainIdx ile erişilir. 

            # m ve n arasındaki fark ise şudur: m en yakın, n ise ikinci en yakın descriptor'ü temsil eder. Lowe ratio için bu iki mesafeye ihtiyacımız vardır.
            # Aralarındaki oran kontrol edilir. 


            # Lowe ratio test
            if m.distance >= self.ratio_thresh * n.distance: 
                continue 
            # Eğer m ile n arasındaki mesafe oranı eşik değerden büyükse bu eşleşme güvenilemez kabul edilir. Örneğin, m.distance = 0.5, n.distance = 0.8 ve ratio_thresh = 0.75 ise, 0.5 >= 0.75 * 0.8 → 0.5 >= 0.6 → False olur, bu eşleşme kabul edilir. Ancak eğer m.distance = 0.7 olsaydı, 0.7 >= 0.6 → True olurdu ve bu eşleşme reddedilirdi.

            # Mutual NN check: curr tarafından bakınca da m.queryIdx'e gidiyor mu?
            if best_bwd.get(m.trainIdx) != m.queryIdx:
                continue
            
            # Kısaca şudur bu kontrol de: En iyi eşleşme olan m.trainIdx (curr_desc içindeki descriptor'ün indeksi) geri yönde bakılınca m.queryIdx (prev_desc içindeki descriptor'ün indeksi) ile eşleşiyor mu? Eğer best_bwd.get(m.trainIdx) bize m.queryIdx'i veriyorsa, bu eşleşme karşılıklıdır ve güvenilir kabul edilir. Örneğin, m.trainIdx = 10 ise, best_bwd.get(10) bize 0 döndürüyorsa ve m.queryIdx de 0 ise, bu eşleşme karşılıklıdır. Ancak eğer best_bwd.get(10) bize 5 döndürüyorsa ve m.queryIdx 0 ise, bu eşleşme karşılıklı değildir ve reddedilir.

            good_matches.append(m)

        return good_matches

    # RANSAC
    def _RANSAC_outliers(self, good_matches, curr_kp):
        """
        Geometrik doğrulama için RANSAC tabanlı Fundamental matrix tahmini.
        
        Args:
            good_matches: list[cv2.DMatch], prev→curr eşleşmeleri (Lowe ratio + mutual check sonrası)
            Returns:
            inlier_mask: np.array(bool), good_matches ile aynı sırada, True ise inlier, False ise outlier
            """
        
        if len(good_matches) < 8:  # Fundamental matrix tahmini için en az 8 eşleşme gerekir
            return np.array([False] * len(good_matches))
        
        pts_prev = np.float32([self.active_tracks[m.queryIdx].keypoints[-1] for m in good_matches]) 
        # Eşleşen keypoint'lerin önceki frame'deki koordinatları alınır. Örneğin, self.active_tracks[m.queryIdx].keypoints[-1] ifadesi, m.queryIdx ile belirtilen aktif takipteki son keypoint'in koordinatlarını verir. Bu, prev_desc içindeki descriptor'ün hangi keypoint'e karşılık geldiğini bulmamızı sağlar.

        pts_curr = np.float32([curr_kp[m.trainIdx].pt for m in good_matches]) 
        # Eşleşen keypoint'lerin yeni frame'deki koordinatları alınır. Örneğin, curr_kp[m.trainIdx].pt ifadesi, m.trainIdx ile belirtilen yeni frame'deki keypoint'in koordinatlarını verir. Bu, curr_desc içindeki descriptor'ün hangi keypoint'e karşılık geldiğini bulmamızı sağlar.

        F, mask = cv2.findFundamentalMat(pts_prev, pts_curr, cv2.FM_RANSAC, self.ransac_thresh, 0.99) # RANSAC algoritması kullanılarak Fundamental matrix tahmini yapılır ve inlier/outlier maskesi elde edilir. self.ransac_thresh, RANSAC algoritmasının inlier olarak kabul edeceği maksimum mesafe eşiğidir. 0.99 ise RANSAC algoritmasının başarı olasılığıdır.

        if mask is None:
            return np.array([False] * len(good_matches)) # Eğer mask None ise, tüm eşleşmeler outlier olarak kabul edilir.
        return mask.ravel().astype(bool) # Mask, 1D boolean array'e dönüştürülür ve inlier'lar True, outlier'lar False olarak işaretlenir. Örneğin, mask = [[1], [0], [1], [1]] ise, mask.ravel() → [1, 0, 1, 1] ve mask.ravel().astype(bool) → [True, False, True, True] olur. Bu sonuç, good_matches listesi ile aynı sırada olup hangi eşleşmelerin inlier olduğunu gösterir.
    
    def _retire_track(self, track):
        """
            Bir track'in takipten çıkarılması (ölmesi) durumunda yapılacak işlemler. MSCKF.update() yalnızca bu tür ölü track'lerle ilgilenir, bu yüzden burada frame_ids ve keypoints bilgisi korunur.

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

    def process_frame(self, image, frame_id):
        """
        Her yeni frame'de çağrılır. Active track listesini günceller,
        yeterince uzun ölmüş track'leri dead_tracks'e koyar.
        
        Args:
            image:    yeni frame (uint8 grayscale, preprocessed)
            frame_id: bu frame'in benzersiz ID'si (MSCKF cam_state ile aynı)
        """
        # Her frame başında dead_tracks'i sıfırla. MSCKF.update() yalnız
        # bu frame'de retire olanlarla ilgilenir.
        self.dead_tracks = []

        # --- Adım 1: yeni frame'in feature'larını çıkar ---
        curr_kp, curr_desc = self._extract_grid_orb(image)

        # --- Edge case: bu frame'de hiç feature yok ---
        if curr_desc is None or len(curr_kp) == 0:
            for tr in self.active_tracks:
                self._retire_track(tr)
            self.active_tracks = []
            return

        # --- Edge case: ilk frame (henüz hiç active track yok) ---
        if len(self.active_tracks) == 0:
            for kp, desc in zip(curr_kp, curr_desc):
                self.active_tracks.append(
                    FeatureTrack(self.next_track_id, frame_id, kp.pt, desc) 
                    # kp.pt burada keypoint'in (x, y) koordinatlarını verir. Örneğin, kp.pt = (150.0, 200.0) gibi bir değer olabilir. kp kendisi ise şu formatta: <KeyPoint 0x7f8c9c0e5b80>, bu nesne içinde pt, size, angle, response gibi özellikler bulunur. Ancak burada sadece pt (koordinatlar) kullanılır.
                )
                self.next_track_id += 1
            return

        # --- Adım 2: prev descriptor'leri active_tracks'ten topla ---
        prev_desc = np.array([t.descriptor for t in self.active_tracks])

        # --- Adım 3: matching (Lowe + mutual NN) ---
        good_matches = self._match_descriptors(prev_desc, curr_desc)
        n_after_desc = len(good_matches)

        # --- Adım 3b: motion prior (max pixel displacement) ---
        # Termal'de homojen bölgelerde RANSAC F-matrix degenerate oluyor;
        # 8 sahte match kendi içinde tutarlı bir epipolar geometriye
        # yerleşebiliyor ve inlier sayılıyor. Bu yüzden RANSAC'a girmeden ÖNCE
        # fiziksel olarak makul olmayan büyük piksel sıçramalarını burada at.
        if self.max_pixel_displacement is not None and len(good_matches) > 0:
            max_d2 = self.max_pixel_displacement ** 2
            filtered = []
            for m in good_matches:
                p_prev = self.active_tracks[m.queryIdx].keypoints[-1]
                p_curr = curr_kp[m.trainIdx].pt
                dx = p_curr[0] - p_prev[0]
                dy = p_curr[1] - p_prev[1]
                if dx * dx + dy * dy <= max_d2:
                    filtered.append(m)
            good_matches = filtered
        n_after_disp = len(good_matches)

        # --- Adım 4: F-matrix RANSAC ile geometric outlier'ları ele ---
        inlier_mask = self._RANSAC_outliers(good_matches, curr_kp)
        inlier_matches = [m for m, ok in zip(good_matches, inlier_mask) if ok]
        n_after_ransac = len(inlier_matches)

        # Tanı: her aşamada kaç match kaldı?
        self.last_match_stats = {
            'after_desc':   n_after_desc,
            'after_disp':   n_after_disp,
            'after_ransac': n_after_ransac,
            'dropped_disp':   n_after_desc  - n_after_disp,
            'dropped_ransac': n_after_disp  - n_after_ransac,
        }

        # --- Adım 5: 3-yönlü sınıflandırma ---
        matched_prev = set()   # match alan prev track index'leri
        matched_curr = set()   # match alan curr keypoint index'leri
        new_active = []

        # SURVIVOR: extend existing tracks
        for m in inlier_matches:
            track = self.active_tracks[m.queryIdx]
            track.frame_ids.append(frame_id)
            track.keypoints.append(curr_kp[m.trainIdx].pt)
            # Descriptor refresh AÇIK: termal'de gradient hızla değişir,
            # descriptor'u en son görünüme güncellemek matching kararlılığını
            # arttırıyor. Drift riski var ama RANSAC + Lowe + mutual-NN
            # zincirleri kötü match'leri zaten yakalıyor.
            track.descriptor = curr_desc[m.trainIdx]
            new_active.append(track)
            matched_prev.add(m.queryIdx)
            matched_curr.add(m.trainIdx)

        # LOST: retire unmatched active tracks
        for i, track in enumerate(self.active_tracks):
            if i not in matched_prev:
                self._retire_track(track)

        # NEW: spawn from unmatched curr keypoints
        for i, (kp, desc) in enumerate(zip(curr_kp, curr_desc)):
            if i not in matched_curr:
                new_active.append(
                    FeatureTrack(self.next_track_id, frame_id, kp.pt, desc)
                )
                self.next_track_id += 1

        self.active_tracks = new_active
