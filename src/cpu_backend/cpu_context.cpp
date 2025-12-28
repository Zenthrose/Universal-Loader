#include "cpu_context.h"

namespace cpu {

ThreadPool::ThreadPool(size_t num_threads) : stop_(false), active_tasks_(0) {
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&ThreadPool::worker_thread, this);
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (auto& worker : workers_) {
        worker.join();
    }
}

void ThreadPool::enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        tasks_.push_back(std::move(task));
        ++active_tasks_;
    }
    condition_.notify_one();
}

void ThreadPool::wait_for_all() {
    std::unique_lock<std::mutex> lock(task_mutex_);
    task_complete_.wait(lock, [this] { return active_tasks_ == 0; });
}

void ThreadPool::worker_thread() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) {
                return;
            }
            task = std::move(tasks_.back());
            tasks_.pop_back();
        }
        task();
        {
            std::unique_lock<std::mutex> lock(task_mutex_);
            --active_tasks_;
            if (active_tasks_ == 0) {
                task_complete_.notify_all();
            }
        }
    }
}

}
