/**
 * @author Allan Souza Almeida
 * @date Feb 5 2025
 */

#pragma once

#ifndef STATE_ESTIMATION_HPP
#define STATE_ESTIMATION_HPP

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
 * @brief This class implements methods for estimating states from a given Path
 */
class StateEstimation
{
public:
    StateEstimation(double angle_offset = 0, double displacement_offset = 0, double angle_calibration_constant = 0);
    ~StateEstimation();

    /**
     * @brief Get the lateral displacement (Z)
     *
     * @param path The Path to get the lateral displacement from
     *
     * @return Lateral displacement (Z)
     */
    double getLateralDisplacement(const prediction::Path::Ptr &path);

    /**
     * @brief Get the angle between real robot frame and the Serret-Frenet frame (theta)
     *
     * @param path The Path to get the angle from
     * @param unit Angle unit, either AngleUnit::DEGREES or AngleUnit::RADIANS (defaults to AngleUnit::RADIANS)
     *
     * @return Angle (theta)
     */
    double getAngle(const prediction::Path::Ptr &path, AngleUnit unit = AngleUnit::RADIANS);

private:
    double angle_offset;               // Used to fix angle offset during calibration
    double displacement_offset;        // Used to fix angle offset during calibration
    double angle_calibration_constant; // Used to fix angle distortion caused by perspective
};

#endif // STATE_ESTIMATION_HPP