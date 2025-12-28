#pragma once
#include <thread>
#include <vector>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <atomic>

namespace cpu {

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();

    void enqueue(std::function<void()> task);
    void wait_for_all();

private:
    void worker_thread();

    std::vector<std::thread> workers_;
    std::vector<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_;
    size_t active_tasks_;
    std::mutex task_mutex_;
    std::condition_variable task_complete_;
};

}
