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

Eigen::VectorXd Processing::leastSquaresFit(const std::vector<cv::Point> &coordinates)
{
    Eigen::MatrixXd A(coordinates.size(), 3);
    Eigen::VectorXd x(coordinates.size());
    for (uint16_t i = 0; i < coordinates.size(); ++i)
    {
        double y = coordinates[i].y;
        A(i, 0) = 1.0;
        A(i, 1) = y;
        A(i, 2) = std::pow(y, 2);
        x(i) = coordinates[i].x;
    }
    return A.colPivHouseholderQr().solve(x);
}

Eigen::VectorXd Processing::ransacFit(const std::vector<cv::Point> &coordinates, const int &min_samples, const int &threshold, const int &max_iterations)
{
    // Seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    Eigen::VectorXd best_model = Eigen::VectorXd::Zero(3);
    std::vector<cv::Point> best_inliers;
    std::vector<cv::Point> picked_samples;
    std::vector<int> picked_indexes;
    int data_len = coordinates.size();
    std::uniform_int_distribution<int> dist(0, data_len - 1);

    for (uint16_t iter = 0; iter < max_iterations; ++iter)
    {
        // Pick `min_samples` random samples from given data
        if (picked_indexes.size() > 0)
        {
            picked_indexes.clear();
            picked_samples.clear();
        }
        while (picked_indexes.size() < min_samples)
        {
            int random_index = dist(gen);
            if (!std::binary_search(picked_indexes.begin(), picked_indexes.end(), random_index))
            {
                picked_indexes.push_back(random_index);
                picked_samples.push_back(coordinates[random_index]);
            }
        }
        Eigen::VectorXd model = Processing::leastSquaresFit(picked_samples);
        std::vector<cv::Point> inliers = Processing::findInliers(coordinates, model, threshold);
        if (inliers.size() > best_inliers.size())
        {
            best_inliers = inliers;
            best_model = model;
            // Skip criteria (90% of points are inliers)
            if (inliers.size() >= (data_len * 0.9))
            {
                if (VERBOSE)
                {
                    std::cout << "Found a good model with "
                              << (inliers.size() * 100 / data_len)
                              << " %% of inliers after " << iter
                              << " iterations" << std::endl;
                }
                break;
            }
        }
    }
    return best_model;
}

std::vector<cv::Point> Processing::calculateCurve(const Eigen::VectorXd &coefficients, const uint16_t &n_points)
{
    int16_t min_y = 0;
    int16_t max_y;

    // Calculate curve y limits

    // From bottom of the image, go upwards until x coordinate is within range (0, width)
    for (int16_t i = this->resolution.second; i > 0; --i)
    {
        double ans = coefficients[0] + coefficients[1] * i + coefficients[2] * i * i;
        if (ans > 0 && ans < this->resolution.first)
        {
            max_y = i;
            break;
        }
    }

    // From top of the image, go downwards until x coordinate is within range (0, width)
    for (int16_t i = 0; i < this->resolution.second; ++i)
    {
        double ans = coefficients[0] + coefficients[1] * i + coefficients[2] * i * i;
        if (ans > 0 && ans < this->resolution.first)
        {
            min_y = i;
            break;
        }
    }

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
    Eigen::VectorXd idx = Eigen::VectorXd::LinSpaced(8, 0, points.size() - 1);
    Eigen::VectorXi indices = idx.cast<int>();
    for (int i = 0; i < indices.size(); ++i)
    {
        cv::circle(image, points.at(indices[i]), 1, cv::Scalar(0, 0, 255), 2);
    }
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
}

/*------------------------------------*/
/* -------- PRIVATE METHODS --------- */
/*------------------------------------*/

std::vector<cv::Point> Processing::findInliers(const std::vector<cv::Point> &coordinates, const Eigen::VectorXd &model, const int &threshold)
{
    std::vector<cv::Point> inliers;
    for (uint32_t i = 0; i < coordinates.size(); ++i)
    {
        float predicted_value = model[0] + model[1] * coordinates[i].y + model[2] * std::pow(coordinates[i].y, 2);
        if (std::abs(coordinates[i].x - predicted_value) <= threshold)
        {
            inliers.push_back(coordinates[i]);
        }
    }
    return inliers;
}