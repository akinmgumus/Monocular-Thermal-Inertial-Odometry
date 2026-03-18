# orb_tracker_node.cpp — Satır Satır Açıklama

Bu node, basit bir ORB feature detection test node'udur. CLAHE görüntülerinden ORB keypoint'leri tespit eder ve sayısını loglar. Pose tahmini yapmaz — pipeline'ın feature extraction kısmını doğrulamak için yazılmıştır. Daha sonra `orb_vo_node` ile tam VO pipeline'ına genişletilmiştir.

---

## Header Dosyaları (1-6)

```cpp
#include <rclcpp/rclcpp.hpp>                    // ROS2 C++ istemci kütüphanesi
#include <sensor_msgs/msg/image.hpp>            // Image mesaj tipi
#include <sensor_msgs/image_encodings.hpp>      // MONO8 encoding sabiti
#include <cv_bridge/cv_bridge.h>                // OpenCV ↔ ROS dönüşümü
#include <opencv2/opencv.hpp>                   // Temel OpenCV
#include <opencv2/features2d.hpp>               // ORB feature detector
```

---

## Constructor (8-19)

```cpp
class OrbTrackerNode : public rclcpp::Node {
public:
    OrbTrackerNode() : Node("orb_tracker_node") {
```

### ORB Detector Oluşturma (12)
```cpp
orb_ = cv::ORB::create(500, 1.2f, 8);
```
- `nfeatures=500`: En fazla 500 keypoint (vo_node'daki 1000'den az — test amaçlı).
- `scaleFactor=1.2f`: Görüntü piramidindeki ölçek faktörü.
- `nlevels=8`: Piramit seviye sayısı.

### Subscriber (14-16)
```cpp
subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/camera/thermal/image_clahe", 10,
    std::bind(&OrbTrackerNode::image_callback, this, std::placeholders::_1));
```
- CLAHE uygulanmış görüntüleri dinler.
- `orb_vo_node` ile aynı topic'i dinler.

---

## image_callback (22-38)

```cpp
void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    try {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
        cv::Mat img = cv_ptr->image;
```
- ROS mesajını 8-bit OpenCV Mat'e dönüştürür.

```cpp
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        orb_->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

        RCLCPP_INFO(this->get_logger(), "Detected %zu keypoints", keypoints.size());
```
- ORB keypoint ve descriptor hesaplar.
- Her frame'de tespit edilen keypoint sayısını loglar.
- **Yalnızca detection yapar, matching/pose estimation yok.**

---

## Üye Değişkenler (40-41)

```cpp
rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscriber_;   // Görüntü aboneliği
cv::Ptr<cv::ORB> orb_;                                                  // ORB detector
```

---

## Main (44-49)

```cpp
int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OrbTrackerNode>());
    rclcpp::shutdown();
    return 0;
}
```

---

## orb_vo_node ile Karşılaştırma

| Özellik | orb_tracker_node | orb_vo_node |
|---------|-----------------|-------------|
| Keypoint sayısı | 500 | 1000 |
| Feature matching | Yok | BFMatcher (Hamming, cross-check) |
| Pose estimation | Yok | Essential Matrix → recoverPose |
| Pose birikim | Yok | Var (R_total, t_total) |
| Trajectory çıktı | Yok | TUM format dosyası |
| ROS çıktı topic | Yok | /vo/odometry, /vo/path |
| Kullanım | Test/doğrulama | Tam VO pipeline |

Bu node, pipeline'ın erken aşamasında "termal görüntülerde ORB ne kadar keypoint buluyor?" sorusunu yanıtlamak için yazılmıştır.