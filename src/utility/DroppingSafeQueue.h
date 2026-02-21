#ifndef DROPPING_SAFE_QUEUE_H
#define DROPPING_SAFE_QUEUE_H

#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T> class DroppingSafeQueue {
public:
  DroppingSafeQueue() = default;

  void push(const T &item) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(item);
    condition_.notify_one();
  }

  bool try_pop(T &item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    item = queue_.front();
    queue_.pop();
    return true;
  }

  void pop(T &item) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return !queue_.empty(); });
    item = queue_.front();
    queue_.pop();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
};

#endif // DROPPING_SAFE_QUEUE_H