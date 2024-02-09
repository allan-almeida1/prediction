/**
 * @author Allan Souza Almeida
 * @date Feb 8 2024
 */

#pragma once

#include <chrono>
#include <iostream>
#include <functional>

typedef void (*LoggerFunction)(const std::string &);

/**
 * @brief This class implements methods to calculate execution time of tasks
 */
class ExecTime
{
private:
    std::chrono::_V2::system_clock::time_point start; // start time for duration calculation
    std::chrono::_V2::system_clock::time_point end;   // end time for duration calculation

public:
    /**
     * @brief Start timer
     */
    void startTimer();

    /**
     * @brief Stop timer
     *
     * @param description Description of operation being timed
     * @param logger A logger function to print the message to terminal
     */
    void stopTimer(
        const std::string &description = "Operation", LoggerFunction logger = [](const std::string &msg)
                                                      { std::cout << msg << std::endl; });
};