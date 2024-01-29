#include "Processing.hpp"

Processing::Processing(std::pair<uint32_t, uint32_t> resolution, bool verbose)
{
    this->resolution = resolution;
    this->VERBOSE = verbose;
}
Processing::~Processing() {}

/*------------------------------------*/
/* --------- PUBLIC METHODS --------- */
/*------------------------------------*/

cv::Mat Processing::openImage(const cv::String &path)
{
    cv::Mat img = cv::imread(path);
    return img;
}

void Processing::showImage(cv::Mat image, cv::String title /* = "Image"*/)
{
    double min_val, max_val;
    cv::Point min_loc, max_loc;

    cv::minMaxLoc(image, &min_val, &max_val, &min_loc, &max_loc);

    cv::Mat img = image;

    if (max_val <= 1.0)
    {
        img *= 255;
    }
    cv::imshow(title, img);
    cv::waitKey(1);
}

void Processing::preprocess(cv::Mat &image)
{
    cv::Mat out_img;
    if (image.type() != CV_8UC1)
    {
        cv::cvtColor(image, out_img, cv::COLOR_BGR2GRAY);
        image = out_img;
    }
    image = image / 255.0;
}

void Processing::binarize(cv::Mat &image, double threshold /* =  0.6*/)
{
    cv::Mat out_img;
    cv::threshold(image, out_img, threshold, 1, cv::THRESH_BINARY);
    image = out_img;
}

std::vector<cv::Point> Processing::findActivePixels(const cv::Mat &image)
{
    std::vector<cv::Point> active_pixels_coordinates;
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            if (image.at<uchar>(y, x) == 1.0)
            {
                active_pixels_coordinates.push_back(cv::Point(x, y));
            }
        }
    }
    return active_pixels_coordinates;
}

arma::mat Processing::dbscan(const std::vector<cv::Point> &dataset, const double eps, const size_t min_points)
{
    mlpack::DBSCAN<> dbscan(eps, min_points);
    arma::Row<size_t> labels;
    arma::mat data(2, dataset.size());

    for (size_t i = 0; i < dataset.size(); ++i)
    {
        data(0, i) = (double)dataset[i].x;
        data(1, i) = (double)dataset[i].y;
    }

    size_t n_clusters = dbscan.Cluster(data, labels);
    if (VERBOSE)
    {
        std::cout << n_clusters << " clusters\n";
    }

    // If only 1 cluster, return it
    if (n_clusters == 1)
    {
        return data;
    }

    // Get the best cluster
    size_t best_cluster;
    double max_value = 0;
    for (size_t i = 0; i < n_clusters; ++i)
    {
        if (VERBOSE)
        {
            std::cout << ((arma::mat)data.cols(arma::find(labels == i))).n_cols << " points found in cluster " << i << std::endl;
        }
        arma::uvec cluster_indices = arma::find(labels == i);
        arma::mat x_coords = (arma::mat)data.cols(cluster_indices);
        double max = x_coords.row(1).max();
        if (max > max_value)
        {
            max_value = max;
            best_cluster = i;
        }
    }
    // ... and return it
    return (arma::mat)data.cols(arma::find(labels == best_cluster));
}

void Processing::drawCluster(const arma::mat &cluster)
{
    cv::Mat image(this->resolution.second, this->resolution.first, CV_64F, 0.0);
    for (arma::uword i = 0; i < cluster.n_cols; ++i)
    {
        uint16_t x = static_cast<uint16_t>(cluster(0, i));
        uint16_t y = static_cast<uint16_t>(cluster(1, i));
        image.at<double>(y, x) = 255;
    }
    cv::imshow("Best cluster", image);
    cv::waitKey(1);
}

Eigen::VectorXd Processing::leastSquaresFit(const arma::mat &data)
{

    Eigen::MatrixXd A(data.n_cols, 3);
    Eigen::VectorXd y(data.n_cols);
    for (arma::uword i = 0; i < data.n_cols; ++i)
    {
        double x = data(1, i);
        A(i, 0) = 1.0;
        A(i, 1) = x;
        A(i, 2) = x * x;
        y(i) = data(0, i);
    }
    return A.colPivHouseholderQr().solve(y);
}

std::vector<cv::Point> Processing::calculateCurve(const Eigen::VectorXd &coefficients, const arma::mat &dataset, const uint16_t &n_points)
{
    double max_y = dataset.row(1).max();
    double min_y = dataset.row(1).min();
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(n_points, min_y, max_y);
    Eigen::VectorXd y = coefficients[0] * Eigen::VectorXd::Ones(x.size()) +
                        coefficients[1] * x +
                        coefficients[2] * x.cwiseProduct(x);
    std::vector<cv::Point> points;
    for (size_t i = 0; i < n_points; ++i)
    {
        int16_t x_px = static_cast<int16_t>(x[i]);
        int16_t y_px = static_cast<int16_t>(y[i]);
        if (x_px < 0)
            x_px = 0;
        if (y_px < 0)
            y_px = 0;
        points.push_back(cv::Point(y_px, x_px));
    }
    return points;
}

void Processing::drawCurve(const std::vector<cv::Point> &points, cv::Mat &image)
{
    std::vector<std::vector<cv::Point>> polylines{points};
    cv::polylines(image, polylines, false, cv::Scalar(255, 0, 0), 1);
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    cv::resize(image, image, cv::Size(640, 352));
    cv::imshow("Estimated curve", image);
    cv::waitKey(1);
}

void Processing::startTimer()
{
    this->start = std::chrono::high_resolution_clock::now();
}

void Processing::stopTimer(const char *description)
{
    this->end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(this->end - this->start);
    ROS_INFO("%s time (ms): %4f", description, (float)duration.count() / 1000);
    // std::cout << description << " took " << ((float)duration.count() / 1000) << " milliseconds" << std::endl;
}