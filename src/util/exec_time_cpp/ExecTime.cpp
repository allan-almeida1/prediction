/**
 * @author Allan Souza Almeida
 * @date Feb 8 2024
 */

#include "ExecTime.hpp"

void ExecTime::startTimer()
{
    this->start = std::chrono::high_resolution_clock::now();
}

void ExecTime::stopTimer(const std::string &description, LoggerFunction logger)
{
    this->end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(this->end - this->start);
    std::string message = description + " time (ms): " + std::to_string((float)duration.count() / 1000);
    logger(message);
}
