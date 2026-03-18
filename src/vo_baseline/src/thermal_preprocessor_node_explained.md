# thermal_preprocessor_node.cpp — Satır Satır Açıklama

Bu node, 16-bit termal görüntüleri alıp CLAHE (Contrast Limited Adaptive Histogram Equalization) uygulayarak 8-bit'e dönüştürür. Feature extraction için görüntüyü hazırlar.

---

## Header Dosyaları (1-7)

```cpp
#include <rclcpp/rclcpp.hpp>                    // ROS2 C++ istemci kütüphanesi
#include <sensor_msgs/msg/image.hpp>            // Image mesaj tipi
#include <sensor_msgs/image_encodings.hpp>      // MONO8, MONO16 encoding sabitleri
#include <std_msgs/msg/header.hpp>              // Header (timestamp + frame_id)
#include <std_msgs/msg/bool.hpp>                // Bool (sequence_complete sinyali)
#include <cv_bridge/cv_bridge.h>                // OpenCV ↔ ROS dönüşümü
#include <opencv2/opencv.hpp>                   // OpenCV (normalize, CLAHE)
```

---

## Constructor (9-26)

```cpp
class ThermalPreprocessorNode : public rclcpp::Node {
public:
    ThermalPreprocessorNode() : Node("thermal_preprocessor_node"), frame_count_(0) {
```
- Node adı `"thermal_preprocessor_node"`, işlenen frame sayacı 0'dan başlar.

### CLAHE Nesnesi Oluşturma (12)
```cpp
clahe_ = cv::createCLAHE(2.0, cv::Size(8, 8));
```
- **clipLimit = 2.0**: Histogram'daki kontrast amplifikasyonunun üst sınırı. Yüksek değer = daha fazla kontrast ama daha fazla gürültü.
- **tileGridSize = (8, 8)**: Görüntü 8x8 bloğa bölünür, her blokta ayrı histogram equalization yapılır. Bu "adaptive" kısmıdır — global yerine lokal kontrast artırımı sağlar.

### Publisher (14-15)
```cpp
publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
    "/camera/thermal/image_clahe", 10);
```
- İşlenmiş 8-bit görüntüleri `/camera/thermal/image_clahe` topic'ine yayınlar.
- ORB VO node bu topic'i dinler.

### Subscriber — Görüntü (17-19)
```cpp
subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/camera/thermal/image_raw", 10,
    std::bind(&ThermalPreprocessorNode::image_callback, this, std::placeholders::_1));
```
- `/camera/thermal/image_raw` topic'inden 16-bit termal görüntüleri dinler.
- Her mesaj geldiğinde `image_callback` çağrılır.
- `std::bind` + `std::placeholders::_1`: Üye fonksiyonu callback olarak bağlar.

### Subscriber — Bitti Sinyali (21-23)
```cpp
done_sub_ = this->create_subscription<std_msgs::msg::Bool>(
    "/vo/sequence_complete", 10,
    std::bind(&ThermalPreprocessorNode::done_callback, this, std::placeholders::_1));
```
- `image_publisher_node` tüm görüntüleri bitirince bu sinyali gönderir.

---

## image_callback (29-56)

Her yeni 16-bit görüntü geldiğinde çalışır.

### ROS Mesajını OpenCV Mat'e Dönüştürme (31-32)
```cpp
cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO16);
cv::Mat img_16bit = cv_ptr->image;
```
- `toCvShare`: ROS mesajını kopyalamadan OpenCV Mat olarak paylaşır (verimli).
- `MONO16`: 16-bit tek kanallı görüntü formatını bekler.
- Sonuç: `img_16bit` → CV_16UC1, 640x512, piksel değerleri 0-65535 aralığında.

### Normalizasyon: 16-bit → 8-bit (34-35)
```cpp
cv::Mat img_8bit;
cv::normalize(img_16bit, img_8bit, 0, 255, cv::NORM_MINMAX, CV_8UC1);
```
- `NORM_MINMAX`: Minimum değeri 0'a, maksimum değeri 255'e eşler, aradakileri lineer interpole eder.
- 16-bit'teki geniş dinamik aralık 8-bit'e sıkıştırılır.
- Bu adım CLAHE'den önce gereklidir çünkü CLAHE sadece 8-bit görüntülerde çalışır.

### CLAHE Uygulama (37-38)
```cpp
cv::Mat img_clahe;
clahe_->apply(img_8bit, img_clahe);
```
- CLAHE algoritması:
  1. Görüntüyü 8x8 bloğa böler
  2. Her blokta histogram equalization uygular
  3. clipLimit ile aşırı kontrast amplifikasyonunu sınırlar
  4. Blok sınırlarını bilinear interpolasyon ile yumuşatır
- Sonuç: Lokal kontrastı artırılmış 8-bit görüntü. Termal görüntülerdeki sıcaklık farklılıkları daha belirgin hale gelir.

### Yayınlama (40-44)
```cpp
std_msgs::msg::Header header = msg->header;     // Orijinal timestamp'i koru
auto processed_msg =
    cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, img_clahe).toImageMsg();
publisher_->publish(*processed_msg);
```
- **Önemli**: Orijinal header (timestamp + frame_id) korunur. Böylece VO node doğru timestamp ile çalışır.
- Encoding artık `MONO8` (8-bit tek kanal).

### İlerleme Logu (45-49)
```cpp
frame_count_++;
if (frame_count_ % 100 == 0) {
    RCLCPP_INFO(this->get_logger(), "Processed %d frames", frame_count_);
}
```
- Her 100 frame'de bir log basarak node'un çalıştığını doğrular.

### Hata Yakalama (51-55)
```cpp
} catch (const cv_bridge::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
} catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Standard exception: %s", e.what());
}
```
- Encoding uyumsuzluğu veya bozuk mesajlarda crash yerine hata logu basar.

---

## done_callback (58-63)

```cpp
void done_callback(const std_msgs::msg::Bool::ConstSharedPtr& msg) {
    if (msg->data) {
        RCLCPP_INFO(this->get_logger(), "Sequence complete. Processed %d frames total. Shutting down.", frame_count_);
        rclcpp::shutdown();
    }
}
```
- `image_publisher_node`'dan gelen bitti sinyalini alır ve node'u kapatır.
- Toplam işlenen frame sayısını loglar.

---

## Üye Değişkenler (65-69)

```cpp
rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscriber_;   // Gelen görüntü aboneliği
rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr done_sub_;         // Bitti sinyali aboneliği
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;       // CLAHE çıktı yayıncısı
cv::Ptr<cv::CLAHE> clahe_;                                             // CLAHE nesnesi (smart pointer)
int frame_count_;                                                       // İşlenen frame sayacı
```

---

## Veri Akışı

```
/camera/thermal/image_raw (16-bit MONO16)
        ↓
   toCvShare → cv::Mat (CV_16UC1)
        ↓
   cv::normalize → 8-bit (CV_8UC1)
        ↓
   CLAHE → lokal kontrast artırılmış 8-bit
        ↓
   cv_bridge → ROS Image (MONO8)
        ↓
/camera/thermal/image_clahe
        ↓
   orb_vo_node dinler
```