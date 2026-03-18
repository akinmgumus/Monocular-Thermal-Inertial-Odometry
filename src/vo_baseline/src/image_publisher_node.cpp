#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/bool.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <algorithm>
#include <fstream>

class ImagePublisherNode : public rclcpp::Node {
public:
    ImagePublisherNode() : Node("image_publisher_node"), current_index_(0) {
        this->declare_parameter<std::string>("image_dir", "");
        this->declare_parameter<std::string>("timestamp_file", "");
        this->declare_parameter<double>("publish_rate", 30.0);

        std::string image_dir = this->get_parameter("image_dir").as_string();
        std::string timestamp_file = this->get_parameter("timestamp_file").as_string();
        double rate = this->get_parameter("publish_rate").as_double();

        if (image_dir.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Parameter 'image_dir' is required.");
            rclcpp::shutdown();
            return;
        }

        // Load and sort image paths
        for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
            if (entry.path().extension() == ".png") {
                image_paths_.push_back(entry.path().string());
            }
        }
        std::sort(image_paths_.begin(), image_paths_.end());

        if (image_paths_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No PNG images found in: %s", image_dir.c_str());
            rclcpp::shutdown();
            return;
        }

        // Load all timestamps — file contains timestamps for ALL frames in the sequence.
        // Each image filename (e.g. "01998.png") is used as an index into this array.
        if (!timestamp_file.empty()) {
            std::ifstream ts_file(timestamp_file);
            std::string line;
            while (std::getline(ts_file, line)) {
                if (!line.empty()) {
                    timestamps_.push_back(std::stoull(line));
                }
            }
            RCLCPP_INFO(this->get_logger(), "Loaded %zu timestamps", timestamps_.size());
        }

        // Extract frame indices from filenames for timestamp lookup
        for (const auto& path : image_paths_) {
            std::string stem = std::filesystem::path(path).stem().string();
            frame_indices_.push_back(static_cast<size_t>(std::stoul(stem)));
        }

        publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/camera/thermal/image_raw", 10);
        done_pub_ = this->create_publisher<std_msgs::msg::Bool>(
            "/vo/sequence_complete", 10);

        auto period = std::chrono::duration<double>(1.0 / rate);
        timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&ImagePublisherNode::timer_callback, this));

        RCLCPP_INFO(this->get_logger(),
            "Image Publisher initialized. %zu images at %.1f Hz", image_paths_.size(), rate);
    }

private:
    void timer_callback() {
        if (current_index_ >= image_paths_.size()) {
            // Signal downstream nodes that the sequence is complete
            std_msgs::msg::Bool done_msg;
            done_msg.data = true;
            done_pub_->publish(done_msg);

            RCLCPP_INFO(this->get_logger(), "All %zu images published. Shutting down.", image_paths_.size());
            rclcpp::shutdown();
            return;
        }

        cv::Mat img = cv::imread(image_paths_[current_index_], cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            RCLCPP_WARN(this->get_logger(), "Failed to read: %s", image_paths_[current_index_].c_str());
            current_index_++;
            return;
        }

        std_msgs::msg::Header header;
        header.frame_id = "thermal_left";

        // Look up timestamp using frame index from filename
        size_t frame_idx = frame_indices_[current_index_];
        if (!timestamps_.empty() && frame_idx < timestamps_.size()) {
            uint64_t ts_ms = timestamps_[frame_idx];
            header.stamp.sec = static_cast<int32_t>(ts_ms / 1000);
            header.stamp.nanosec = static_cast<uint32_t>((ts_ms % 1000) * 1000000);
        } else {
            header.stamp = this->get_clock()->now();
        }

        std::string encoding = (img.type() == CV_16UC1)
            ? sensor_msgs::image_encodings::MONO16
            : sensor_msgs::image_encodings::MONO8;

        auto msg = cv_bridge::CvImage(header, encoding, img).toImageMsg();
        publisher_->publish(*msg);

        current_index_++;
    }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr done_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<std::string> image_paths_;
    std::vector<uint64_t> timestamps_;
    std::vector<size_t> frame_indices_;
    size_t current_index_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImagePublisherNode>());
    rclcpp::shutdown();
    return 0;
}