#include "Processing.hpp"

namespace processing
{

    cv::Mat openImage(const cv::String &path)
    {
        cv::Mat img = cv::imread(path);
        return img;
    }

    void showImage(cv::Mat image, cv::String title /* = "Image"*/)
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

    void preprocess(cv::Mat &image)
    {
        cv::Mat out_img;
        if (image.type() != CV_8UC1)
        {
            cv::cvtColor(image, out_img, cv::COLOR_BGR2GRAY);
            image = out_img;
        }
        image = image / 255.0;
    }

    void binarize(cv::Mat &image, double threshold /* =  0.6*/)
    {
        cv::Mat out_img;
        cv::threshold(image, out_img, threshold, 1, cv::THRESH_BINARY);
        image = out_img;
    }

    std::vector<cv::Point> findActivePixels(const cv::Mat &image)
    {
        std::vector<cv::Point> active_pixels_coordinates;
        cv::findNonZero(image, active_pixels_coordinates);
        return active_pixels_coordinates;
    }

    arma::mat dbscan(const std::vector<cv::Point> &dataset, const double eps, const size_t min_points)
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

    void drawCluster(const arma::mat &cluster, Resolution resolution)
    {
        cv::Mat image(resolution.height, resolution.width, CV_64F, 0.0);
        for (arma::uword i = 0; i < cluster.n_cols; ++i)
        {
            uint16_t x = static_cast<uint16_t>(cluster(0, i));
            uint16_t y = static_cast<uint16_t>(cluster(1, i));
            image.at<double>(y, x) = 255;
        }
        cv::imshow("Best cluster", image);
        cv::waitKey(1);
    }

    Eigen::VectorXd leastSquaresFit(const std::vector<cv::Point> &coordinates, const int &order)
    {
        Eigen::MatrixXd A(coordinates.size(), order + 1);
        Eigen::VectorXd x(coordinates.size());
        for (uint16_t i = 0; i < coordinates.size(); ++i)
        {
            double y = coordinates[i].y;
            A(i, 0) = 1.0;
            for (uint16_t j = 1; j < order + 1; ++j)
            {
                A(i, j) = std::pow(y, j);
            }
            x(i) = coordinates[i].x;
        }
        return A.colPivHouseholderQr().solve(x);
    }

    Eigen::VectorXd ransacFit(const std::vector<cv::Point> &coordinates, const int &order, const int &min_samples, const int &threshold, const int &max_iterations)
    {

        Eigen::VectorXd best_model = Eigen::VectorXd::Zero(3);
        std::vector<cv::Point> best_inliers;
        std::vector<cv::Point> picked_samples;
        std::vector<int> picked_indexes;
        std::random_device rd;
        std::mt19937 gen(rd());
        int data_len = coordinates.size();

        std::uniform_int_distribution<int> dist(0, data_len - 1);

#pragma omp parallel for shared(best_model, best_inliers) default(none) schedule(static) // Add OpenMP pragma for parallelization
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
                if (std::find(picked_indexes.begin(), picked_indexes.end(), random_index) == picked_indexes.end())
                {
                    picked_indexes.push_back(random_index);
                    picked_samples.push_back(coordinates[random_index]);
                }
            }
            Eigen::VectorXd model = leastSquaresFit(picked_samples, order);
            std::vector<cv::Point> inliers = findInliers(coordinates, model, threshold);
#pragma omp critical
            {
                if (inliers.size() > best_inliers.size())
                {
                    best_inliers = inliers;
                    best_model = model;
                    // Skip criteria (90% of points are inliers)
                    if (inliers.size() >= (data_len * 0.9))
                    {
                        break;
                    }
                }
            }
        }

        Eigen::VectorXd post_ransac_best_model = leastSquaresFit(best_inliers, order);

        return post_ransac_best_model;
    }

    Curve calculateCurve(const Eigen::VectorXd &coefficients, Resolution resolution, const uint16_t &n_points)
    {
        int16_t min_y = 0;
        int16_t max_y;
        Curve curve;

        // Calculate curve y limits

        // From bottom of the image, go upwards until x coordinate is within range (0, width)
        for (int16_t i = resolution.height; i > 0; --i)
        {
            double ans = 0;
            for (int16_t j = 0; j < coefficients.size(); ++j)
            {
                ans += coefficients[j] * std::pow(i, j);
            }
            if (ans > 0 && ans < resolution.width)
            {
                max_y = i;
                break;
            }
        }

        curve.y_max = max_y;

        // From top of the image, go downwards until x coordinate is within range (0, width)
        for (int16_t i = 0; i < resolution.height; ++i)
        {
            double ans = 0;
            for (int16_t j = 0; j < coefficients.size(); ++j)
            {
                ans += coefficients[j] * std::pow(i, j);
            }
            if (ans > 0 && ans < resolution.width)
            {
                min_y = i;
                break;
            }
        }

        curve.y_min = min_y;

        Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(n_points, min_y, max_y);
        Eigen::VectorXd x = Eigen::VectorXd::Zero(y.size());
        for (int16_t i = 0; i < coefficients.size(); ++i)
        {
            x += coefficients[i] * (y.array().pow(i)).matrix();
        }
        std::vector<cv::Point> points;
        for (size_t i = 0; i < n_points; ++i)
        {
            int16_t x_px = static_cast<int16_t>(x[i]);
            int16_t y_px = static_cast<int16_t>(y[i]);
            if (x_px < 0)
                x_px = 0;
            if (y_px < 0)
                y_px = 0;
            points.push_back(cv::Point(x_px, y_px));
        }
        curve.points = points;
        return curve;
    }

    void drawCurve(const Curve &curve, cv::Mat &image)
    {
        std::vector<std::vector<cv::Point>> polylines{curve.points};
        cv::polylines(image, polylines, false, cv::Scalar(255, 0, 0), 2);
        for (int i = 0; i < curve.points.size(); ++i)
        {
            cv::circle(image, curve.points.at(i), 2, cv::Scalar(0, 0, 255), 2);
        }
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        cv::resize(image, image, cv::Size(640, 352), 0, 0, cv::INTER_LINEAR);
        // cv::resize(image, image, cv::Size(480, 264));
        cv::imshow("Estimated curve", image);
        cv::waitKey(1);
    }

    std::vector<cv::Point> findInliers(const std::vector<cv::Point> &coordinates, const Eigen::VectorXd &model, const int &threshold)
    {
        std::vector<cv::Point> inliers;
        for (uint32_t i = 0; i < coordinates.size(); ++i)
        {
            float predicted_value = 0;
            for (int16_t j = 0; j < model.size(); ++j)
            {
                predicted_value += model[j] * std::pow(coordinates[i].y, j);
            }
            if (std::abs(coordinates[i].x - predicted_value) <= threshold)
            {
                inliers.push_back(coordinates[i]);
            }
        }
        return inliers;
    }

}; // namespace processing