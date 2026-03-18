# orb_vo_node.cpp — Satır Satır Açıklama

Bu node, CLAHE uygulanmış termal görüntülerden ORB feature extraction ve matching yaparak monoküler Visual Odometry (VO) hesaplar. Kamera pozisyonunu frame frame biriktirir ve TUM formatında trajectory dosyası yazar.

---

## Header Dosyaları (1-11)

```cpp
#include <rclcpp/rclcpp.hpp>                       // ROS2 C++ istemci kütüphanesi
#include <sensor_msgs/msg/image.hpp>               // Image mesaj tipi
#include <sensor_msgs/image_encodings.hpp>         // MONO8 encoding sabiti
#include <nav_msgs/msg/odometry.hpp>               // Odometry mesajı (pose + twist)
#include <nav_msgs/msg/path.hpp>                   // Path mesajı (pose dizisi, RViz'de görselleştirme)
#include <geometry_msgs/msg/pose_stamped.hpp>      // Tek bir timestamped pose
#include <cv_bridge/cv_bridge.h>                   // OpenCV ↔ ROS dönüşümü
#include <opencv2/opencv.hpp>                      // Temel OpenCV (findEssentialMat, recoverPose)
#include <opencv2/features2d.hpp>                  // ORB, BFMatcher
#include <std_msgs/msg/bool.hpp>                   // Bitti sinyali
#include <fstream>                                 // Trajectory dosyası yazma
```

---

## Constructor (13-53)

### ORB ve Matcher Oluşturma (17-18)
```cpp
orb_ = cv::ORB::create(1000, 1.2f, 8);
matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, true);
```
- **ORB(1000, 1.2, 8)**:
  - `nfeatures=1000`: Her frame'de en fazla 1000 keypoint tespit et.
  - `scaleFactor=1.2`: Görüntü piramidindeki ölçek faktörü. Her seviyede görüntü 1.2x küçültülür.
  - `nlevels=8`: Piramit seviye sayısı. Farklı ölçeklerdeki feature'ları yakalamak için.
- **BFMatcher(HAMMING, crossCheck=true)**:
  - `NORM_HAMMING`: ORB descriptor'ları binary olduğu için Hamming distance kullanılır (XOR + popcount).
  - `crossCheck=true`: A→B ve B→A eşleşmesi aynı sonucu vermelidir. Yanlış eşleşmeleri azaltır.

### Kamera İntrinsik Parametreleri (20-26)
```cpp
double fx = 406.33233091474426;    // Odak uzaklığı x (piksel cinsinden)
double fy = 406.9536696029995;     // Odak uzaklığı y
double cx = 311.51174613074784;    // Optik merkez x (principal point)
double cy = 241.75862889759748;    // Optik merkez y
K_ = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
dist_coeffs_ = (cv::Mat_<double>(1, 4) << -0.34952316, 0.10382263, -0.00014740, -0.00018344);
```
- **K_ (Intrinsic Matrix)**: Piksel koordinatlarını normalize kamera koordinatlarına dönüştürür.
  ```
  K = [fx  0  cx]
      [0  fy  cy]
      [0   0   1]
  ```
- **dist_coeffs_**: Radyal-tanjansiyel distortion katsayıları [k1, k2, p1, p2].
  - k1=-0.35: Güçlü negatif radyal distortion (barrel distortion).
  - k2=0.10: İkinci dereceden radyal düzeltme.
  - p1, p2 ≈ 0: Minimal tanjansiyel distortion.
- Bu değerler `firestereo.yaml` konfigürasyon dosyasından alınmıştır.

### Kümülatif Pose Başlatma (29-30)
```cpp
R_total_ = cv::Mat::eye(3, 3, CV_64F);     // 3x3 birim matris (başlangıç rotasyonu)
t_total_ = cv::Mat::zeros(3, 1, CV_64F);   // 3x1 sıfır vektör (başlangıç pozisyonu)
```
- Başlangıçta kamera orijinde, rotasyon yok.
- Her frame'de relative R,t eklenerek kümülatif pose güncellenir.

### Trajectory Dosyası (33-36)
```cpp
this->declare_parameter<std::string>("trajectory_file", "/tmp/vo_trajectory.txt");
trajectory_path_ = this->get_parameter("trajectory_file").as_string();
traj_file_.open(trajectory_path_);
traj_file_ << "# timestamp tx ty tz qx qy qz qw" << std::endl;
```
- TUM formatında trajectory çıktısı: `timestamp x y z qx qy qz qw`
- Bu format ground truth ile karşılaştırma için standarttır.

### Publisher ve Subscriber (39-50)
```cpp
odom_pub_  → "/vo/odometry"         // Her frame'de güncel pose
path_pub_  → "/vo/path"             // Tüm trajectory (RViz'de çizim için)
subscriber_ ← "/camera/thermal/image_clahe"  // CLAHE görüntülerini dinle
done_sub_   ← "/vo/sequence_complete"         // Bitti sinyalini dinle
```

---

## image_callback — Ana VO Döngüsü (63-173)

Her CLAHE görüntüsü geldiğinde çalışır. Tüm VO pipeline'ı burada gerçekleşir.

### 1. ROS → OpenCV Dönüşümü (64-66)
```cpp
cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
cv::Mat img = cv_ptr->image;
```
- 8-bit gri tonlamalı görüntü alınır.

### 2. ORB Feature Detection (69-71)
```cpp
std::vector<cv::KeyPoint> keypoints;
cv::Mat descriptors;
orb_->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
```
- `detectAndCompute`: Hem keypoint tespiti hem de descriptor hesaplaması tek seferde.
- **Keypoint**: Görüntüdeki ayırt edici nokta (köşe, kenar vb.), (x, y, size, angle) bilgisi içerir.
- **Descriptor**: Her keypoint'in 256-bit binary tanımlayıcısı. İki frame arasında eşleştirme için kullanılır.
- Termal görüntülerde RGB'ye kıyasla daha az keypoint bulunur (düşük texture).

### 3. İlk Frame Kontrolü (81-88)
```cpp
if (prev_descriptors_.empty()) {
    prev_keypoints_ = keypoints;
    prev_descriptors_ = descriptors.clone();
    write_pose(msg->header.stamp);          // Başlangıç pozisyonunu kaydet (0,0,0)
    publish_odometry(msg->header.stamp);
    return;
}
```
- VO frame-to-frame çalışır, ilk frame'de karşılaştırılacak önceki frame yoktur.
- Feature'lar saklanır, bir sonraki frame ile eşleştirilecektir.

### 4. Feature Matching (91-99)
```cpp
std::vector<cv::DMatch> matches;
matcher_->match(prev_descriptors_, descriptors, matches);
```
- Önceki frame'in descriptor'ları ile şimdiki frame'in descriptor'ları Hamming distance ile eşleştirilir.
- Cross-check: İki yönlü eşleşme tutarlılığı sağlanır.

### 5. Match Filtreleme (102-105)
```cpp
std::sort(matches.begin(), matches.end(),
    [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });
size_t keep = std::max(static_cast<size_t>(20), static_cast<size_t>(matches.size() * 0.6));
matches.resize(std::min(keep, matches.size()));
```
- Eşleşmeler distance'a göre sıralanır (küçük = daha iyi).
- En iyi %60'ı tutulur (minimum 20 eşleşme).
- Kötü eşleşmelerin çıkarılması pose tahmininin doğruluğunu artırır.

### 6. Eşleşen Noktaların Koordinatlarını Çıkarma (108-112)
```cpp
std::vector<cv::Point2f> pts_prev, pts_curr;
for (const auto& m : matches) {
    pts_prev.push_back(prev_keypoints_[m.queryIdx].pt);    // Önceki frame'deki nokta
    pts_curr.push_back(keypoints[m.trainIdx].pt);           // Şimdiki frame'deki nokta
}
```
- `queryIdx`: Önceki frame'deki keypoint indeksi.
- `trainIdx`: Şimdiki frame'deki eşleşen keypoint indeksi.

### 7. Lens Distortion Düzeltmesi (115-117)
```cpp
cv::undistortPoints(pts_prev, pts_prev_undist, K_, dist_coeffs_, cv::noArray(), K_);
cv::undistortPoints(pts_curr, pts_curr_undist, K_, dist_coeffs_, cv::noArray(), K_);
```
- Lens bozulmasını düzeltir. Barrel distortion nedeniyle kenarlar eğrilir, bu adım noktaları "ideal" konumlarına taşır.
- Son parametre `K_`: Düzeltilmiş noktaları tekrar piksel koordinatlarına dönüştürür.
- Essential Matrix hesabı için doğru nokta konumları gereklidir.

### 8. Essential Matrix Hesaplama (120-122)
```cpp
cv::Mat E = cv::findEssentialMat(pts_prev_undist, pts_curr_undist, K_,
                                  cv::RANSAC, 0.999, 1.0, inlier_mask);
```
- **Essential Matrix (E)**: İki kamera görünümü arasındaki rotasyon ve ötelenme bilgisini kodlar.
- Matematiksel olarak: `p2' * E * p1 = 0` (epipolar constraint).
- **RANSAC**: Random Sample Consensus — outlier'lara dayanıklı tahmin.
  - `0.999`: İstenen güven seviyesi (%99.9).
  - `1.0`: Piksel cinsinden inlier eşik değeri.
- `inlier_mask`: Hangi eşleşmelerin inlier olduğunu gösterir.

### 9. Pose Kurtarma (R, t) (141-142)
```cpp
cv::Mat R, t;
int recovered = cv::recoverPose(E, pts_prev_undist, pts_curr_undist, K_, R, t, inlier_mask);
```
- Essential Matrix'ten 4 olası (R, t) çözümü vardır. `recoverPose` fiziksel olarak anlamlı olanı seçer (noktalar kameranın önünde olmalıdır — cheirality check).
- **R**: 3x3 rotasyon matrisi (frame-to-frame rotasyon).
- **t**: 3x1 birim öteleme vektörü (yön bilgisi var ama **ölçek yok** — monoküler VO'nun temel sınırlaması).
- `recovered`: Cheirality check'i geçen nokta sayısı.

### 10. Kümülatif Pose Güncelleme (153-154)
```cpp
t_total_ = t_total_ + R_total_ * t;    // Ötelemeyi dünya koordinatlarına çevir ve ekle
R_total_ = R_total_ * R;                // Rotasyonu birikimli olarak çarp
```
- **Kritik formül**: `T_world = T_world * T_relative`
- `R_total_ * t`: Relative ötelemeyi şimdiki dünya yöneliminde ifade eder.
- Bu frame-to-frame birikim VO'nun temelidir. Her frame'deki küçük hatalar birikir (drift).

### 11. Kaydetme ve Yayınlama (157-158)
```cpp
write_pose(msg->header.stamp);         // TUM dosyasına yaz
publish_odometry(msg->header.stamp);   // ROS topic'lerine yayınla
```

---

## publish_odometry (176-202)

```cpp
void publish_odometry(const builtin_interfaces::msg::Time& stamp) {
    double qw, qx, qy, qz;
    rotation_to_quaternion(R_total_, qw, qx, qy, qz);   // 3x3 rotasyon → quaternion
    // ... Odometry mesajı doldur ve yayınla
    // ... Path mesajına yeni pose ekle ve yayınla
}
```
- Rotasyon matrisi quaternion'a çevrilir (ROS mesajları quaternion kullanır).
- Path mesajı kümülatiftir — her frame'de büyür, RViz'de tüm yol görselleştirilir.

---

## write_pose (204-215)

```cpp
void write_pose(const builtin_interfaces::msg::Time& stamp) {
    double ts = stamp.sec + stamp.nanosec * 1e-9;   // ROS time → double saniye
    traj_file_ << ts << " " << tx << " " << ty << " " << tz << " "
               << qx << " " << qy << " " << qz << " " << qw << std::endl;
}
```
- TUM formatı: `timestamp tx ty tz qx qy qz qw`
- Ground truth ile aynı formatta olduğu için doğrudan karşılaştırılabilir.

---

## rotation_to_quaternion (217-244)

3x3 rotasyon matrisini quaternion'a (qw, qx, qy, qz) dönüştürür.

```cpp
double trace = R(0,0) + R(1,1) + R(2,2);
```
- **Trace > 0 durumu** (en yaygın): Shepperd yöntemi ile doğrudan hesaplanır.
- **Trace ≤ 0 durumu**: Rotasyon matrisinin hangi diyagonal elemanı en büyükse ona göre dal seçilir. Bu, sayısal kararlılık için gereklidir (sıfıra bölme riski).
- 4 farklı dal var çünkü quaternion'un 4 bileşeninden her biri belirli rotasyonlarda dominant olabilir.

---

## Üye Değişkenler (254-274)

```cpp
// ROS iletişimi
subscriber_, done_sub_            // Görüntü ve bitti sinyali abonelikleri
odom_pub_, path_pub_              // Odometry ve Path yayıncıları

// Bilgisayar görüsü
orb_, matcher_                    // ORB detector ve BFMatcher
K_, dist_coeffs_                  // Kamera intrinsik ve distortion parametreleri

// Frame-to-frame takip
prev_keypoints_, prev_descriptors_   // Önceki frame'in feature'ları

// Kümülatif pose
R_total_, t_total_                // Dünya koordinatlarında toplam rotasyon ve öteleme

// Çıktı
traj_file_, path_msg_            // Dosya yazıcı ve Path mesajı
frame_count_                     // İşlenen frame sayacı
```

---

## Algoritma Özeti

```
Her yeni CLAHE frame geldiğinde:
  1. ORB ile keypoint + descriptor çıkar
  2. Önceki frame ile BFMatcher eşleştir
  3. En iyi %60 match'i tut
  4. Lens distortion'ı düzelt
  5. Essential Matrix hesapla (RANSAC)
  6. recoverPose ile R, t çıkar
  7. T_total = T_total * T_relative (pose biriktir)
  8. Trajectory dosyasına ve ROS topic'lerine yaz
```

## Bilinen Sınırlamalar

- **Scale ambiguity**: Monoküler VO'da gerçek ölçek bilinemez. t vektörü her zaman birim uzunluktadır.
- **Drift**: Frame-to-frame hatalar birikir, özellikle rotasyonlarda.
- **Pure rotation**: Öteleme olmadan Essential Matrix dejenere olur, pose tahmini başarısız olur.