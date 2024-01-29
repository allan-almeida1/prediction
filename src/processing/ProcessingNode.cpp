#include "ProcessingNode.hpp"

ProcessingNode::ProcessingNode(ros::NodeHandle *nh)
{
    this->processing = Processing();
    this->img_sub = nh->subscribe("/image_raw_bin", 100, &ProcessingNode::imageCallback, this);
}

ProcessingNode::~ProcessingNode() {}

void ProcessingNode::imageCallback(const sensor_msgs::Image::Ptr &img)
{
    processing.startTimer();

    cv::Mat image;
    image = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8)->image;

    cv::resize(image, image, cv::Size(320, 176), 0, 0, CV_INTER_LINEAR);

    cv::Mat original_img = image;

    // Preprocessing step
    processing.preprocess(image);

    // Processing step
    auto start = std::chrono::high_resolution_clock::now();

    processing.binarize(image);

    std::vector<cv::Point> coords = processing.findActivePixels(image);

    arma::mat best_cluster = processing.dbscan(coords, 3.0, 10UL);

    Eigen::VectorXd coefficients = processing.leastSquaresFit(best_cluster);

    std::vector<cv::Point> points = processing.calculateCurve(coefficients, best_cluster, 20);

    processing.stopTimer("Processing step");

    processing.drawCurve(points, original_img);
}