#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/bool.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class ThermalPreprocessorNode : public rclcpp::Node {
public:
    ThermalPreprocessorNode() : Node("thermal_preprocessor_node"), frame_count_(0) {
        clahe_ = cv::createCLAHE(2.0, cv::Size(8, 8));

        publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/camera/thermal/image_clahe", 10);

        subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/thermal/image_raw", 10,
            std::bind(&ThermalPreprocessorNode::image_callback, this, std::placeholders::_1));

        done_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/vo/sequence_complete", 10,
            std::bind(&ThermalPreprocessorNode::done_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Thermal Preprocessor Node initialized. Waiting for 16-bit images...");
    }

private:
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO16);
            cv::Mat img_16bit = cv_ptr->image;

            cv::Mat img_8bit;
            cv::normalize(img_16bit, img_8bit, 0, 255, cv::NORM_MINMAX, CV_8UC1);

            cv::Mat img_clahe;
            clahe_->apply(img_8bit, img_clahe);

            std_msgs::msg::Header header = msg->header;
            auto processed_msg =
                cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, img_clahe).toImageMsg();

            publisher_->publish(*processed_msg);
            frame_count_++;

            if (frame_count_ % 100 == 0) {
                RCLCPP_INFO(this->get_logger(), "Processed %d frames", frame_count_);
            }

        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Standard exception: %s", e.what());
        }
    }

    void done_callback(const std_msgs::msg::Bool::ConstSharedPtr& msg) {
        if (msg->data) {
            RCLCPP_INFO(this->get_logger(), "Sequence complete. Processed %d frames total. Shutting down.", frame_count_);
            rclcpp::shutdown();
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscriber_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr done_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    cv::Ptr<cv::CLAHE> clahe_;
    int frame_count_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ThermalPreprocessorNode>());
    rclcpp::shutdown();
    return 0;
}