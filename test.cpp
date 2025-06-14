#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <vector>
#include <functional>
#include <memory>
#include <string>

// --------- 假设已有的 llama_model 和推理相关 ---------
struct llama_model {
    // 模型权重数据等
    void info() { std::cout << "Model info: shared weights\n"; }
};

struct llama_context {
    llama_model * model;
    // 推理缓存、状态等，比如KV缓存，token历史
    std::string session_id; // 用于标识用户会话（示例）
    // 初始化上下文
    llama_context(llama_model * m, std::string id) : model(m), session_id(id) {}
};

std::string llama_infer(llama_context & ctx, const std::string & prompt) {
    // 这里写你的推理调用逻辑，示范返回固定字符串
    return "Response to '" + prompt + "' from session " + ctx.session_id;
}
// -----------------------------------------------------

// 线程安全的任务队列与线程池
class ThreadPool {
public:
    ThreadPool(size_t n_threads) : done(false) {
        for (size_t i = 0; i < n_threads; ++i) {
            workers.emplace_back([this]() { this->worker_thread(); });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            done = true;
        }
        cv.notify_all();
        for (auto & t : workers) {
            if (t.joinable()) t.join();
        }
    }

    void enqueue(std::function<void()> task) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            tasks.push(std::move(task));
        }
        cv.notify_one();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mutex_;
    std::condition_variable cv;
    bool done;

    void worker_thread() {
        while (true) {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv.wait(lock, [this]() { return done || !tasks.empty(); });

                if (done && tasks.empty())
                    return;

                task = std::move(tasks.front());
                tasks.pop();
            }

            task();
        }
    }
};

// 全局模型指针（共享模型）
llama_model * global_model = nullptr;

// 业务层：收到请求，放入线程池执行
void handle_inference_request(ThreadPool & pool, const std::string & session_id, const std::string & prompt) {
    pool.enqueue([session_id, prompt]() {
        // 每个请求创建独立上下文
        llama_context ctx(global_model, session_id);

        // 推理
        std::string output = llama_infer(ctx, prompt);

        // 返回结果（这里简单打印）
        std::cout << "[Thread " << std::this_thread::get_id() << "] "
                  << "Session " << session_id << ": " << output << std::endl;
    });
}

int main() {
    // 1. 初始化全局模型（只加载一次）
    global_model = new llama_model();
    global_model->info();

    // 2. 启动线程池
    ThreadPool pool(4); // 4个工作线程
    // 3. 模拟多用户多请求并发调用
    handle_inference_request(pool, "user1", "Hello, llama!");
    handle_inference_request(pool, "user2", "What's the weather?");
    handle_inference_request(pool, "user1", "Tell me a joke.");
    handle_inference_request(pool, "user3", "Explain quantum physics.");

    // 等待任务完成（简化做法：主线程sleep）
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // 4. 清理
    delete global_model;
    return 0;
}
