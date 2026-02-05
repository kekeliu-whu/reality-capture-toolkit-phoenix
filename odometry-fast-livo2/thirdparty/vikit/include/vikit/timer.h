#ifndef TIMER_H
#define TIMER_H

#include <chrono>

namespace vk
{

class Timer
{
private:
  using Clock = std::chrono::high_resolution_clock;
  Clock::time_point start_time_;
  double time_;
  double accumulated_;
public:

  /// The constructor directly starts the timer.
  Timer() :
    time_(0.0),
    accumulated_(0.0)
  {
    start();
  }

  ~Timer()
  {}

  inline void start()
  {
    accumulated_ = 0.0;
    start_time_ = Clock::now();
  }

  inline void resume()
  {
    start_time_ = Clock::now();
  }

  inline double stop()
  {
    auto end_time = Clock::now();
    auto duration = std::chrono::duration<double>(end_time - start_time_);
    time_ = duration.count() + accumulated_;
    accumulated_ = time_;
    return time_;
  }

  inline double getTime() const
  {
    return time_;
  }

  inline void reset()
  {
    time_ = 0.0;
    accumulated_ = 0.0;
  }

  static double getCurrentTime()
  {
    auto now = Clock::now();
    auto duration = std::chrono::duration<double>(now.time_since_epoch());
    return duration.count();
  }

  static double getCurrentSecond()
  {
    auto now = Clock::now();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch());
    return static_cast<double>(seconds.count());
  }

};

} // end namespace vk

#endif
