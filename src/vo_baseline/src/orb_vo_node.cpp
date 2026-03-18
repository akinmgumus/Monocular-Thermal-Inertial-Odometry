#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <std_msgs/msg/bool.hpp>
#include <fstream>

class OrbVoNode : public rclcpp::Node {
public:
    OrbVoNode() : Node("orb_vo_node"), frame_count_(0) {
        // ORB detector
        orb_ = cv::ORB::create(1000, 1.2f, 8);
        matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, true);

        // Camera intrinsics from firestereo.yaml
        double fx = 406.33233091474426;
        double fy = 406.9536696029995;
        double cx = 311.51174613074784;
        double cy = 241.75862889759748;
        K_ = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        dist_coeffs_ = (cv::Mat_<double>(1, 4) << -0.34952316, 0.10382263, -0.00014740, -0.00018344);

        // Initialize cumulative pose as identity
        R_total_ = cv::Mat::eye(3, 3, CV_64F);
        t_total_ = cv::Mat::zeros(3, 1, CV_64F);

        // Output trajectory file
        this->declare_parameter<std::string>("trajectory_file", "/tmp/vo_trajectory.txt");
        trajectory_path_ = this->get_parameter("trajectory_file").as_string();
        traj_file_.open(trajectory_path_);
        traj_file_ << "# timestamp tx ty tz qx qy qz qw" << std::endl;

        // Publishers
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/vo/odometry", 10);
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/vo/path", 10);
        path_msg_.header.frame_id = "odom";

        // Subscribers
        subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/thermal/image_clahe", 10,
            std::bind(&OrbVoNode::image_callback, this, std::placeholders::_1));

        done_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/vo/sequence_complete", 10,
            std::bind(&OrbVoNode::done_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "ORB VO Node initialized. Trajectory output: %s", trajectory_path_.c_str());
    }

    ~OrbVoNode() {
        if (traj_file_.is_open()) {
            traj_file_.close();
            RCLCPP_INFO(this->get_logger(), "Trajectory saved to %s (%d frames)", trajectory_path_.c_str(), frame_count_);
        }
    }

private:
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
            cv::Mat img = cv_ptr->image;

            // Detect ORB features
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            orb_->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

            if (keypoints.empty() || descriptors.empty()) {
                RCLCPP_WARN(this->get_logger(), "No keypoints detected, skipping frame");
                return;
            }

            frame_count_++;

            // First frame: just store and return
            if (prev_descriptors_.empty()) {
                prev_keypoints_ = keypoints;
                prev_descriptors_ = descriptors.clone();
                write_pose(msg->header.stamp);
                publish_odometry(msg->header.stamp);
                RCLCPP_INFO(this->get_logger(), "First frame captured with %zu keypoints", keypoints.size());
                return;
            }

            // Match features between previous and current frame
            std::vector<cv::DMatch> matches;
            matcher_->match(prev_descriptors_, descriptors, matches);

            if (matches.size() < 10) {
                RCLCPP_WARN(this->get_logger(), "Too few matches (%zu), skipping frame", matches.size());
                prev_keypoints_ = keypoints;
                prev_descriptors_ = descriptors.clone();
                return;
            }

            // Filter matches by distance threshold (keep best 60%)
            std::sort(matches.begin(), matches.end(),
                [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });
            size_t keep = std::max(static_cast<size_t>(20), static_cast<size_t>(matches.size() * 0.6));
            matches.resize(std::min(keep, matches.size()));

            // Extract matched point coordinates
            std::vector<cv::Point2f> pts_prev, pts_curr;
            for (const auto& m : matches) {
                pts_prev.push_back(prev_keypoints_[m.queryIdx].pt);
                pts_curr.push_back(keypoints[m.trainIdx].pt);
            }

            // Undistort points
            std::vector<cv::Point2f> pts_prev_undist, pts_curr_undist;
            cv::undistortPoints(pts_prev, pts_prev_undist, K_, dist_coeffs_, cv::noArray(), K_);
            cv::undistortPoints(pts_curr, pts_curr_undist, K_, dist_coeffs_, cv::noArray(), K_);

            // Compute Essential Matrix
            cv::Mat inlier_mask;
            cv::Mat E = cv::findEssentialMat(pts_prev_undist, pts_curr_undist, K_,
                                              cv::RANSAC, 0.999, 1.0, inlier_mask);

            if (E.empty()) {
                RCLCPP_WARN(this->get_logger(), "Essential matrix estimation failed");
                prev_keypoints_ = keypoints;
                prev_descriptors_ = descriptors.clone();
                return;
            }

            // Count inliers
            int inlier_count = cv::countNonZero(inlier_mask);
            if (inlier_count < 10) {
                RCLCPP_WARN(this->get_logger(), "Too few inliers (%d)", inlier_count);
                prev_keypoints_ = keypoints;
                prev_descriptors_ = descriptors.clone();
                return;
            }

            // Recover rotation and translation from Essential Matrix
            cv::Mat R, t;
            int recovered = cv::recoverPose(E, pts_prev_undist, pts_curr_undist, K_, R, t, inlier_mask);

            if (recovered < 10) {
                RCLCPP_WARN(this->get_logger(), "recoverPose: too few points (%d)", recovered);
                prev_keypoints_ = keypoints;
                prev_descriptors_ = descriptors.clone();
                return;
            }

            // Accumulate pose: T_total = T_total * T_current
            // Monocular VO has scale ambiguity - using unit translation
            t_total_ = t_total_ + R_total_ * t;
            R_total_ = R_total_ * R;

            // Publish and save
            write_pose(msg->header.stamp);
            publish_odometry(msg->header.stamp);

            RCLCPP_INFO(this->get_logger(),
                "Frame %d | matches: %zu | inliers: %d | pos: [%.2f, %.2f, %.2f]",
                frame_count_, matches.size(), inlier_count,
                t_total_.at<double>(0), t_total_.at<double>(1), t_total_.at<double>(2));

            // Update previous frame
            prev_keypoints_ = keypoints;
            prev_descriptors_ = descriptors.clone();

        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
        }
    }

    void publish_odometry(const builtin_interfaces::msg::Time& stamp) {
        // Convert rotation matrix to quaternion
        double qw, qx, qy, qz;
        rotation_to_quaternion(R_total_, qw, qx, qy, qz);

        // Odometry message
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = "odom";
        odom.child_frame_id = "base_link";
        odom.pose.pose.position.x = t_total_.at<double>(0);
        odom.pose.pose.position.y = t_total_.at<double>(1);
        odom.pose.pose.position.z = t_total_.at<double>(2);
        odom.pose.pose.orientation.x = qx;
        odom.pose.pose.orientation.y = qy;
        odom.pose.pose.orientation.z = qz;
        odom.pose.pose.orientation.w = qw;
        odom_pub_->publish(odom);

        // Path message (append pose)
        geometry_msgs::msg::PoseStamped pose;
        pose.header = odom.header;
        pose.pose = odom.pose.pose;
        path_msg_.poses.push_back(pose);
        path_msg_.header.stamp = stamp;
        path_pub_->publish(path_msg_);
    }

    void write_pose(const builtin_interfaces::msg::Time& stamp) {
        double qw, qx, qy, qz;
        rotation_to_quaternion(R_total_, qw, qx, qy, qz);
        double ts = stamp.sec + stamp.nanosec * 1e-9;
        traj_file_ << std::fixed << std::setprecision(6)
                    << ts << " "
                    << t_total_.at<double>(0) << " "
                    << t_total_.at<double>(1) << " "
                    << t_total_.at<double>(2) << " "
                    << qx << " " << qy << " " << qz << " " << qw
                    << std::endl;
    }

    void rotation_to_quaternion(const cv::Mat& R, double& qw, double& qx, double& qy, double& qz) {
        double trace = R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2);
        if (trace > 0) {
            double s = 0.5 / std::sqrt(trace + 1.0);
            qw = 0.25 / s;
            qx = (R.at<double>(2, 1) - R.at<double>(1, 2)) * s;
            qy = (R.at<double>(0, 2) - R.at<double>(2, 0)) * s;
            qz = (R.at<double>(1, 0) - R.at<double>(0, 1)) * s;
        } else if (R.at<double>(0, 0) > R.at<double>(1, 1) && R.at<double>(0, 0) > R.at<double>(2, 2)) {
            double s = 2.0 * std::sqrt(1.0 + R.at<double>(0, 0) - R.at<double>(1, 1) - R.at<double>(2, 2));
            qw = (R.at<double>(2, 1) - R.at<double>(1, 2)) / s;
            qx = 0.25 * s;
            qy = (R.at<double>(0, 1) + R.at<double>(1, 0)) / s;
            qz = (R.at<double>(0, 2) + R.at<double>(2, 0)) / s;
        } else if (R.at<double>(1, 1) > R.at<double>(2, 2)) {
            double s = 2.0 * std::sqrt(1.0 + R.at<double>(1, 1) - R.at<double>(0, 0) - R.at<double>(2, 2));
            qw = (R.at<double>(0, 2) - R.at<double>(2, 0)) / s;
            qx = (R.at<double>(0, 1) + R.at<double>(1, 0)) / s;
            qy = 0.25 * s;
            qz = (R.at<double>(1, 2) + R.at<double>(2, 1)) / s;
        } else {
            double s = 2.0 * std::sqrt(1.0 + R.at<double>(2, 2) - R.at<double>(0, 0) - R.at<double>(1, 1));
            qw = (R.at<double>(1, 0) - R.at<double>(0, 1)) / s;
            qx = (R.at<double>(0, 2) + R.at<double>(2, 0)) / s;
            qy = (R.at<double>(1, 2) + R.at<double>(2, 1)) / s;
            qz = 0.25 * s;
        }
    }

    void done_callback(const std_msgs::msg::Bool::ConstSharedPtr& msg) {
        if (msg->data) {
            RCLCPP_INFO(this->get_logger(), "Sequence complete. %d frames processed. Trajectory saved to %s",
                frame_count_, trajectory_path_.c_str());
            rclcpp::shutdown();
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscriber_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr done_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

    cv::Ptr<cv::ORB> orb_;
    cv::Ptr<cv::BFMatcher> matcher_;
    cv::Mat K_, dist_coeffs_;

    // Previous frame data
    std::vector<cv::KeyPoint> prev_keypoints_;
    cv::Mat prev_descriptors_;

    // Cumulative pose
    cv::Mat R_total_, t_total_;

    // Trajectory output
    std::string trajectory_path_;
    std::ofstream traj_file_;
    nav_msgs::msg::Path path_msg_;
    int frame_count_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OrbVoNode>());
    rclcpp::shutdown();
    return 0;
}
