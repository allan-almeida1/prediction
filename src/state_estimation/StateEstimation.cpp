/**
 * @author Allan Souza Almeida
 * @date Feb 5 2024
 */

#include "StateEstimation.hpp"

StateEstimation::StateEstimation(double angle_offset, double displacement_offset, double angle_calibration_constant)
{
    this->angle_offset = angle_offset;
    this->displacement_offset = displacement_offset;
    this->angle_calibration_constant = angle_calibration_constant;
}

StateEstimation::~StateEstimation() {}

double StateEstimation::getLateralDisplacement(const prediction::Path::Ptr &path)
{
    int h = int(path->resolution.height / 2);
    double x_px = 0;
    for (int16_t i = 0; i < path->coefficients.size(); ++i)
    {
        x_px += path->coefficients[i] * std::pow(h, i);
    }
    // Translate to the center of the image
    x_px -= path->resolution.width / 2;
    return x_px;
}

double StateEstimation::getAngle(const prediction::Path::Ptr &path, AngleUnit unit)
{
    int h = int(path->resolution.height / 2);
    double delta = 0.5;
    double x_plus, x_minus;
    for (int16_t i = 0; i < path->coefficients.size(); ++i)
    {
        x_plus += path->coefficients[i] * std::pow(h + delta, i);
        x_minus += path->coefficients[i] * std::pow(h - delta, i);
    }
    double delta_y = 2 * delta;
    double delta_x = x_plus - x_minus;
    double angle;
    if (unit == AngleUnit::RADIANS)
    {
        angle = -std::atan2(delta_x, delta_y) + this->angle_offset +
                this->angle_calibration_constant * StateEstimation::getLateralDisplacement(path);
    }
    else if (unit == AngleUnit::DEGREES)
    {
        angle = (-std::atan2(delta_x, delta_y) + this->angle_offset +
                 this->angle_calibration_constant * StateEstimation::getLateralDisplacement(path)) *
                180 / M_PI;
    }
    else
    {
        std::cerr << "Invalid angle unit specified. Choose either AngleUnit::RADIANS or AngleUnit::DEGREES." << std::endl;
        angle = 0.0;
    }
    return angle;
}