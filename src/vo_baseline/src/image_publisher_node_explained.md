# image_publisher_node.cpp — Satır Satır Açıklama

Bu node, diskten 16-bit termal görüntüleri okuyarak ROS2 topic'ine yayınlar. FireStereo gibi offline dataset'leri pipeline üzerinde çalıştırmak için kullanılır.

---

## Header Dosyaları (1-10)

```cpp
#include <rclcpp/rclcpp.hpp>                    // ROS2 C++ istemci kütüphanesi (Node, Publisher, Timer vb.)
#include <sensor_msgs/msg/image.hpp>            // sensor_msgs/Image mesaj tipi (görüntü taşımak için)
#include <sensor_msgs/image_encodings.hpp>      // MONO8, MONO16 gibi encoding sabitleri
#include <std_msgs/msg/header.hpp>              // Header mesajı (timestamp + frame_id)
#include <std_msgs/msg/bool.hpp>                // Bool mesajı (sequence_complete sinyali için)
#include <cv_bridge/cv_bridge.h>                // OpenCV Mat <-> ROS Image dönüşümü
#include <opencv2/opencv.hpp>                   // OpenCV temel fonksiyonları (imread vb.)
#include <filesystem>                           // C++17 dosya sistemi (dizin okuma, dosya adı parse)
#include <algorithm>                            // std::sort
#include <fstream>                              // Dosya okuma (timestamps.txt)
```

---

## Sınıf Tanımı ve Constructor (12-74)

```cpp
class ImagePublisherNode : public rclcpp::Node {
```
- `rclcpp::Node`'dan türetilen ROS2 node sınıfı.

### Constructor (14-74)

```cpp
ImagePublisherNode() : Node("image_publisher_node"), current_index_(0) {
```
- Node adı `"image_publisher_node"`, yayınlanacak görüntünün indeksi 0'dan başlar.

#### Parametre Tanımları (15-21)
```cpp
this->declare_parameter<std::string>("image_dir", "");         // Görüntü dizini yolu
this->declare_parameter<std::string>("timestamp_file", "");    // Timestamp dosyası yolu
this->declare_parameter<double>("publish_rate", 30.0);         // Yayın hızı (Hz)
```
- ROS2 parametre sistemi kullanılır. Çalıştırırken `--ros-args -p image_dir:=/path/to/images` şeklinde verilir.
- Varsayılan yayın hızı 30 Hz.

#### Parametre Doğrulama (23-27)
```cpp
if (image_dir.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Parameter 'image_dir' is required.");
    rclcpp::shutdown();
    return;
}
```
- `image_dir` verilmezse hata verip kapanır.

#### Görüntü Dosyalarını Yükleme (30-35)
```cpp
for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
    if (entry.path().extension() == ".png") {
        image_paths_.push_back(entry.path().string());
    }
}
std::sort(image_paths_.begin(), image_paths_.end());
```
- `std::filesystem::directory_iterator` ile dizindeki tüm `.png` dosyaları toplanır.
- Alfabetik/numerik sıralama yapılır (`01998.png`, `01999.png`, `02000.png`, ...).

#### Timestamp Dosyasını Yükleme (45-54)
```cpp
if (!timestamp_file.empty()) {
    std::ifstream ts_file(timestamp_file);
    std::string line;
    while (std::getline(ts_file, line)) {
        if (!line.empty()) {
            timestamps_.push_back(std::stoull(line));
        }
    }
}
```
- `timestamps.txt` dosyasındaki her satır milisaniye cinsinden epoch timestamp'tir.
- Tüm timestamp'ler bir vektöre yüklenir. Bu dizideki indeks, frame numarasına karşılık gelir.

#### Dosya Adından Frame İndeksi Çıkarma (57-60)
```cpp
for (const auto& path : image_paths_) {
    std::string stem = std::filesystem::path(path).stem().string();
    frame_indices_.push_back(static_cast<size_t>(std::stoul(stem)));
}
```
- `01998.png` → stem = `"01998"` → `std::stoul("01998")` → indeks `1998`
- Bu indeks, `timestamps_[1998]` şeklinde doğru timestamp'e erişmek için kullanılır.
- **Kritik**: Dosya adları 0'dan başlamaz (01998'den başlar), ama timestamps.txt 0. satırdan başlar. Bu yüzden dosya adı doğrudan indeks olarak kullanılır.

#### Publisher ve Timer Kurulumu (62-73)
```cpp
publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
    "/camera/thermal/image_raw", 10);           // Ana görüntü topic'i, queue size=10
done_pub_ = this->create_publisher<std_msgs::msg::Bool>(
    "/vo/sequence_complete", 10);               // Bitti sinyali topic'i

auto period = std::chrono::duration<double>(1.0 / rate);  // 30Hz → ~33ms
timer_ = this->create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(period),
    std::bind(&ImagePublisherNode::timer_callback, this));
```
- `create_wall_timer`: Belirtilen periyodda `timer_callback` fonksiyonunu çağırır.
- Her timer tick'inde bir görüntü yayınlanır.

---

## Timer Callback (77-117)

Bu fonksiyon her tick'te çağrılır ve sıradaki görüntüyü yayınlar.

### Tüm Görüntüler Bittiğinde (78-87)
```cpp
if (current_index_ >= image_paths_.size()) {
    std_msgs::msg::Bool done_msg;
    done_msg.data = true;
    done_pub_->publish(done_msg);               // Diğer node'lara "bitti" sinyali gönder
    RCLCPP_INFO(this->get_logger(), "All %zu images published. Shutting down.", image_paths_.size());
    rclcpp::shutdown();                         // Bu node'u kapat
    return;
}
```
- Son görüntü yayınlandıktan sonra `/vo/sequence_complete` topic'ine `true` gönderilir.
- Preprocessor ve VO node bu sinyali alıp kendilerini kapatır.

### Görüntüyü Diskten Okuma (89-94)
```cpp
cv::Mat img = cv::imread(image_paths_[current_index_], cv::IMREAD_UNCHANGED);
```
- `IMREAD_UNCHANGED`: 16-bit PNG dosyasını orijinal formatında (CV_16UC1) okur.
- 8-bit olarak okursaydık termal bilgi kaybı olurdu.

### Timestamp Atama (96-107)
```cpp
size_t frame_idx = frame_indices_[current_index_];      // Dosya adından gelen indeks
if (!timestamps_.empty() && frame_idx < timestamps_.size()) {
    uint64_t ts_ms = timestamps_[frame_idx];             // Milisaniye epoch
    header.stamp.sec = static_cast<int32_t>(ts_ms / 1000);        // Saniye kısmı
    header.stamp.nanosec = static_cast<uint32_t>((ts_ms % 1000) * 1000000);  // Nanosaniye
} else {
    header.stamp = this->get_clock()->now();              // Fallback: şimdiki zaman
}
```
- Milisaniye timestamp'i ROS2'nin `sec` + `nanosec` formatına dönüştürülür.
- Örnek: `1679940973658` ms → sec=`1679940973`, nanosec=`658000000`

### Encoding Belirleme ve Yayınlama (109-116)
```cpp
std::string encoding = (img.type() == CV_16UC1)
    ? sensor_msgs::image_encodings::MONO16
    : sensor_msgs::image_encodings::MONO8;

auto msg = cv_bridge::CvImage(header, encoding, img).toImageMsg();
publisher_->publish(*msg);
current_index_++;
```
- `cv_bridge::CvImage`: OpenCV Mat'i ROS Image mesajına dönüştürür.
- Encoding otomatik belirlenir (16-bit ise MONO16, 8-bit ise MONO8).

---

## Üye Değişkenler (119-125)

```cpp
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;     // Görüntü yayıncısı
rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr done_pub_;          // Bitti sinyali yayıncısı
rclcpp::TimerBase::SharedPtr timer_;                                 // Periyodik timer
std::vector<std::string> image_paths_;                               // Sıralı görüntü dosya yolları
std::vector<uint64_t> timestamps_;                                   // Tüm timestamp'ler (indeksli)
std::vector<size_t> frame_indices_;                                  // Her dosyanın frame indeksi
size_t current_index_;                                               // Şu an yayınlanacak indeks
```

---

## Main Fonksiyonu (128-133)

```cpp
int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);                                    // ROS2 başlat
    rclcpp::spin(std::make_shared<ImagePublisherNode>());        // Node'u çalıştır, timer callback'leri işle
    rclcpp::shutdown();                                          // Temiz kapanış
    return 0;
}
```
- `rclcpp::spin`: Node'u event loop'a sokar, `shutdown()` çağrılana kadar çalışır.

---

## Veri Akışı

```
Disk (PNG dosyaları) → imread(UNCHANGED) → cv_bridge → /camera/thermal/image_raw topic
                                                         ↓
                                              thermal_preprocessor_node dinler
```