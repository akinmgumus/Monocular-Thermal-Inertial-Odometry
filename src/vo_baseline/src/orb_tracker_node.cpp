#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

class OrbTrackerNode : public rclcpp::Node {
public:
    OrbTrackerNode() : Node("orb_tracker_node") {
        // ORB detector: 500 keypoints, scale factor 1.2, 8 pyramid levels
        orb_ = cv::ORB::create(500, 1.2f, 8);

        subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/thermal/image_clahe", 10,
            std::bind(&OrbTrackerNode::image_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "ORB Tracker Node initialized. Waiting for CLAHE images...");
    }

private:
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
            cv::Mat img = cv_ptr->image;

            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            orb_->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

            RCLCPP_INFO(this->get_logger(), "Detected %zu keypoints", keypoints.size());

        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Standard exception: %s", e.what());
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscriber_;
    cv::Ptr<cv::ORB> orb_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OrbTrackerNode>());
    rclcpp::shutdown();
    return 0;
}
