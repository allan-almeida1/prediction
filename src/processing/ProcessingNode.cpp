#include "ProcessingNode.hpp"

void ros_logger(const std::string &message)
{
    ROS_INFO("%s", message.c_str());
}

ProcessingNode::ProcessingNode(ros::NodeHandle *nh)
{
    this->nh = nh;
    // this->processing = Processing();
    this->img_sub = nh->subscribe("/image_raw_bin", 100, &ProcessingNode::imageCallback, this);
    this->path_pub = nh->advertise<prediction::Path>("/prediction/path", 100);
    nh->param("/processing/window_size", this->window_size, 7);
    nh->param("/processing/n_points", this->n_points, 8);
    nh->param("/processing/order", this->order, 2);
    nh->param("/processing/min_samples", this->min_samples, 3);
    nh->param("/processing/threshold", this->threshold, 10);
    nh->param("/processing/max_iterations", this->max_iterations, 200);
    nh->param("/processing/width", this->width, 320);
    nh->param("/processing/height", this->height, 176);
    this->resolution = processing::Resolution(this->width, this->height);
}

ProcessingNode::~ProcessingNode() {}

void ProcessingNode::imageCallback(const sensor_msgs::Image::Ptr &img)
{
    execTime.startTimer();

    cv::Mat image;
    image = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8)->image;

    cv::resize(image, image, cv::Size(320, 176), 0, 0, CV_INTER_LINEAR);

    cv::Mat original_img = image;

    // Preprocessing step
    processing::preprocess(image);

    // Processing step
    auto start = std::chrono::high_resolution_clock::now();

    processing::binarize(image);

    std::vector<cv::Point> coordinates = processing::findActivePixels(image);

    if (coordinates.size() > 30)
    {
        // Fit a second order polynomial to data using RANSAC
        Eigen::VectorXd coefficients = processing::ransacFit(coordinates, this->order, this->min_samples, this->threshold, this->max_iterations);
        // Eigen::VectorXd coefficients = processing::leastSquaresFit(coordinates, this->order);

        // Append coefficients to vector for average calculation
        ransac_results.push_back(coefficients);

        if (ransac_results.size() > this->window_size)
        {
            ransac_results.pop_front();
        }

        // Calculate average
        Eigen::VectorXd filtered_coefficients = Eigen::VectorXd::Zero(this->order + 1);
        for (uint16_t i = 0; i < ransac_results.size(); ++i)
        {
            filtered_coefficients += ransac_results[i];
        }
        filtered_coefficients /= ransac_results.size();

        // Calculate points to draw the curve
        processing::Curve curve = processing::calculateCurve(filtered_coefficients, this->resolution, this->n_points);

        execTime.stopTimer("Processing step", &ros_logger);

        // Prepare Path info and publish to topic
        prediction::Path path;
        path.Header.stamp = ros::Time::now();
        std::vector<double> v_coefficients(
            filtered_coefficients.data(), filtered_coefficients.data() + filtered_coefficients.size());
        path.coefficients = v_coefficients;
        prediction::Limits y_limits;
        y_limits.y_max = curve.y_max;
        y_limits.y_min = curve.y_min;
        path.limits = y_limits;
        path.resolution.width = this->resolution.width;
        path.resolution.height = this->resolution.height;
        path_pub.publish(path);

        processing::drawCurve(curve, original_img);
    }
    else
    {
        execTime.stopTimer("Processing step", &ros_logger);
    }
}