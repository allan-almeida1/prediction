/**
 * @author Allan Souza Almeida
 * @date Feb 5 2025
 */

#pragma once

#include <prediction/Path.h>
#include <cmath>

/**
 * @brief Simple enum class to choose between radians or degrees
 */
enum class AngleUnit
{
    RADIANS,
    DEGREES
};

/**
 * @brief States Z (lateral displacement) and theta (angular error)
 */
struct States
{
    double Z;
    double theta;

    States() : Z(0.0), theta(0.0) {}
    States(double _Z, double _theta) : Z(_Z), theta(_theta) {}
};

/**
 * @brief This class implements methods for estimating states from a given Path
 */
namespace state_estimation
{

    /**
     * @brief Get states Z (lateral displacement) and theta (angular error) for the median point
     * of the curve (h = height/2)
     *
     * @param path The Path to get the states from
     * @param Z_cal Constant term to callibrate the lateral displacement (Z)
     * @param theta_cal Constant term to callibrate the angular error (theta)
     * @param angle_unit Angle unit, either AngleUnit::DEGREES or AngleUnit::RADIANS (defaults to AngleUnit::RADIANS)
     *
     * @return States
     */
    States getStates(const prediction::Path::Ptr &path, double Z_cal, double theta_cal, AngleUnit angle_unit = AngleUnit::RADIANS);

    /**
     * @brief Get curvature for the median point of the curve (h = height/2)
     *
     * @param path The Path to calculate central curvature
     * @param curv_cal Constant term to convert curvature from rad/px to rad/m
     *
     * @return Curvature for h = height/2
     */
    double calculateCurvature(const prediction::Path::Ptr &path, double curv_cal);

}; // namespace state_estimation
