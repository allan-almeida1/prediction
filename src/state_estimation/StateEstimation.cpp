/**
 * @author Allan Souza Almeida
 * @date Feb 5 2024
 */

#include "StateEstimation.hpp"

namespace state_estimation
{

    States getStates(const prediction::Path::Ptr &path, double Z_cal, double theta_cal, AngleUnit angle_unit)
    {
        int h = int(path->resolution.height * 3 / 4);
        States states;
        double delta = 0.5;
        double x_px = 0;
        double x_plus = 0;
        double x_minus = 0;
        for (int16_t i = 0; i < path->coefficients.size(); ++i)
        {
            x_minus += path->coefficients[i] * std::pow(h - delta, i);
            x_px += path->coefficients[i] * std::pow(h, i);
            x_plus += path->coefficients[i] * std::pow(h + delta, i);
        }

        // Translate to the center of the image
        x_px -= path->resolution.width / 2;

        // Calibrate Z
        states.Z = x_px * Z_cal;

        double delta_y = 2 * delta;
        double delta_x = x_plus - x_minus;
        double angle;
        if (angle_unit == AngleUnit::RADIANS)
        {
            // Calculate derivative and calibrate
            angle = -std::atan2(delta_x, delta_y) * theta_cal;
        }
        else if (angle_unit == AngleUnit::DEGREES)
        {
            // Calculate derivative, calibrate and convert to degrees
            angle = -std::atan2(delta_x, delta_y) * theta_cal * 180 / M_PI;
        }
        else
        {
            std::cerr << "Invalid angle unit specified. Choose either AngleUnit::RADIANS or AngleUnit::DEGREES." << std::endl;
            angle = 0.0;
        }
        states.theta = angle;
        return states;
    }

    double calculateCurvature(const prediction::Path::Ptr &path, double curv_calc)
    {
        int h = int(path->resolution.height * 3 / 4);
        double dx_dy = 0.0;
        double d2x_dy2 = 0.0;
        for (int16_t i = 0; i < path->coefficients.size(); ++i)
        {
            dx_dy += path->coefficients[i] * i * std::pow(h, i - 1);
            d2x_dy2 += path->coefficients[i] * i * (i - 1) * std::pow(h, i - 2);
        }
        return d2x_dy2 / std::sqrt(std::pow(1 + std::pow(dx_dy, 2), 3)) / curv_calc;
    }

}; // namespace state_estimation