#include <iostream>
#include <utility>
#include <vector>
#include <random>

struct Worker {
    std::vector<float> snapshot_weights;
    std::vector<std::pair<float, float>> batch;
    std::vector<float> calculate_gradient(){
        float grad_w = 0;
        float grad_b = 0;
        float w = snapshot_weights[0];
        float b = snapshot_weights[1];
        for ( size_t i = 0; i <batch.size(); ++i){
            float error = w * batch[i].first + b - batch[i].second;
            grad_w += 2 *error * batch[i].first;
            grad_b += 2 * error; 
        }
        grad_w = grad_w / batch.size();
        grad_b = grad_b/ batch.size();
        return {grad_w, grad_b};
    }
};
class server { 
    private:
    std::vector<float> global_weights;
    float lr = 0.01f;
    public:
    std::vector<float> pull(){
        return global_weights;
    }
    explicit server(std::vector<float> initial_weights) : global_weights{std::move(initial_weights)} {}

bool push(const std::vector<float>& new_gradients){
        if (global_weights.size() == new_gradients.size()){
            for (size_t index = 0; index < new_gradients.size(); ++index){
                global_weights[index] = global_weights[index] - (lr * new_gradients[index]);
              }
              return true;
        }
        return false;
    };
};
std::vector<std::pair<float, float>> make_data(size_t n, unsigned seed = 42) {
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> x_dist(-10.0f, 10.0f);
    std::normal_distribution<float> noise(0.0f, 1.0f);

    std::vector<std::pair<float, float>> data;
    data.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        float x = x_dist(engine);
        float y = 3.0f * x + 2.0f + noise(engine);
        data.push_back({x, y});
    }
    return data;
}


int main(){
    server first_server{{0.0f, 0.0f}};
    Worker first_worker;
    first_worker.batch = make_data(100);
    for (int i = 0; i< 1000; ++i){
    first_worker.snapshot_weights = first_server.pull();
    first_server.push(first_worker.calculate_gradient());
    }
    std::vector<float> final_snapshot = first_server.pull();
    std:: cout << "W: " << final_snapshot[0] << " B: " << final_snapshot[1] << "\n";
    return 0;
}
