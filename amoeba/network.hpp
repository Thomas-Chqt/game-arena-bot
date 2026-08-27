#pragma once

#include "mcts.hpp"

#include <mlx/mlx.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <functional>
#include <numeric>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace amoeba_bot
{

std::string childName(std::string_view parent, std::string_view child);
std::string indexedName(std::string_view parent, size_t index);

template<typename T>
concept Module = requires(const T& module, mlx::core::array input, std::span<const mlx::core::array> parameters)
{
    module(std::move(input), parameters);
};

template<typename T>
struct NetworkName
{
    static_assert(false, "");
};

struct Prediction
{
    mlx::core::array policy; // [batch, moveIdCount]
    mlx::core::array value;  // [batch]
};

// Training keeps its data as MLX arrays so the complete forward-and-loss graph
// remains differentiable. Batch construction itself stays in training.cpp.
struct TrainingBatch
{
    mlx::core::array input;
    mlx::core::array legal; // [batchSize, moveIdCount]
    mlx::core::array policyTarget;
    mlx::core::array valueTarget;
};

struct Loss
{
    mlx::core::array total;
    mlx::core::array policy;
    mlx::core::array value;
};

struct LossAndGrad
{
    Loss loss;
    // There is one gradient tensor for each tensor in the supplied parameter
    // vector, in exactly the same order.
    std::vector<mlx::core::array> gradients;
};

class Network
{
    struct ParameterDefinition
    {
        std::string name;
        mlx::core::Shape shape;
    };

    using CompiledForward =
        std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)>;

public:
    virtual ~Network() = default;

    // MCTS-facing inference. This fixed Amoeba adapter batches Board objects,
    // masks illegal moves, and returns probabilities indexed by absolute move id.
    void operator()(std::span<const Board* const> boards,
                    std::span<Evaluation> outputs) const;

    // Validation only needs the losses and therefore avoids calculating
    // gradients. Training calls valueAndGrad with the parameter values that the
    // optimizer is currently updating.
    Loss loss(const std::vector<mlx::core::array>& parameters,
              const TrainingBatch& batch, float weightDecay) const;
    LossAndGrad valueAndGrad(const std::vector<mlx::core::array>& parameters,
                             const TrainingBatch& batch, float weightDecay) const;

    // Modules call this from their constructors and retain the returned index.
    // The actual tensor stays here in Network; a module owns only its structure.
    size_t addParameter(std::string parameterName, mlx::core::array value)
    {
        assert(std::ranges::find(m_layout, parameterName, &ParameterDefinition::name) == m_layout.end());
        assert(value.ndim() > 0);
        assert(value.dtype() == mlx::core::float32);

        const size_t index = m_parameters.size();
        m_layout.push_back({std::move(parameterName), value.shape()});
        m_parameters.push_back(std::move(value));
        return index;
    }

    const std::vector<mlx::core::array>& parameters() const { return m_parameters; }

    // MLX arrays are immutable graph values, so Adam produces a new vector after
    // every step instead of changing tensors in place.
    void replaceParameters(std::vector<mlx::core::array> parameters)
    {
        if (parameters.size() != m_parameters.size())
            throw std::runtime_error(std::format("expected {} parameter tensors, got {}",
                                                 m_parameters.size(), parameters.size()));

        for (size_t index = 0; index < parameters.size(); ++index)
        {
            if (parameters[index].shape() != m_layout[index].shape)
                throw std::runtime_error(std::format("parameter {} has shape different from {}", index,
                                                     m_layout[index].name));
            if (parameters[index].dtype() != mlx::core::float32)
                throw std::runtime_error(std::format("parameter {} ({}) is not float32", index,
                                                     m_layout[index].name));
        }
        m_parameters = std::move(parameters);
    }

    size_t parameterCount() const
    {
        return std::accumulate(m_parameters.begin(), m_parameters.end(), size_t{0}, [](size_t total, const mlx::core::array& parameter) {
            return total + parameter.size();
        });
    }

    void printSummary() const
    {
        for (size_t index = 0; index < m_layout.size(); ++index)
        {
            std::print("{}  [", m_layout[index].name);
            for (size_t dimension = 0; dimension < m_layout[index].shape.size(); ++dimension)
                std::print("{}{}", dimension == 0 ? "" : ", ", m_layout[index].shape[dimension]);
            std::println("]  {}", m_parameters[index].size());
        }
        std::println("Total parameters: {}", parameterCount());
    }

    void save(const std::filesystem::path& checkpoint) const
    {
        std::unordered_map<std::string, mlx::core::array> tensors;
        for (size_t index = 0; index < m_layout.size(); ++index)
            tensors.emplace(m_layout[index].name, m_parameters[index]);
        mlx::core::save_safetensors(checkpoint.string(), tensors, {{"network", m_name}});
    }

protected:
    // The base is constructed before the derived module members, so setting the
    // seed here makes their subsequent parameter initialization reproducible.
    Network(const char* name, uint64_t seed)
        : m_name(name)
    {
        assert(name != nullptr);
        assert(*name != '\0');
        mlx::core::random::seed(seed);
    }

    // A compiled callable captures `this`, so copies retain parameter state but
    // deliberately compile their own callable on first use.
    Network(const Network& other)
        : m_name(other.m_name)
        , m_layout(other.m_layout)
        , m_parameters(other.m_parameters)
    {
    }

    Network& operator=(const Network& other)
    {
        if (this != &other)
        {
            m_name = other.m_name;
            m_layout = other.m_layout;
            m_parameters = other.m_parameters;
            m_compiledForward = {};
        }
        return *this;
    }

    void materializeParameters() { mlx::core::eval(m_parameters); }

    void load(const std::filesystem::path& checkpoint)
    {
        auto [tensors, metadata] = mlx::core::load_safetensors(checkpoint.string());

        const auto storedName = metadata.find("network");
        if (storedName == metadata.end() || storedName->second != m_name)
            throw std::runtime_error(std::format("{}: checkpoint is not for network {}", checkpoint.string(), m_name));
        if (tensors.size() != m_layout.size())
            throw std::runtime_error(std::format("{}: expected {} tensors, found {}", checkpoint.string(),
                                                 m_layout.size(), tensors.size()));

        std::vector<mlx::core::array> loaded;
        loaded.reserve(m_layout.size());
        for (size_t index = 0; index < m_layout.size(); ++index)
        {
            const auto found = tensors.find(m_layout[index].name);
            if (found == tensors.end())
                throw std::runtime_error(std::format("{}: no tensor named {}", checkpoint.string(), m_layout[index].name));
            if (found->second.shape() != m_layout[index].shape)
                throw std::runtime_error(std::format("{}: {} has the wrong shape", checkpoint.string(), m_layout[index].name));
            if (found->second.dtype() != mlx::core::float32)
                throw std::runtime_error(std::format("{}: {} is not float32", checkpoint.string(), m_layout[index].name));
            loaded.push_back(found->second);
        }
        m_parameters = std::move(loaded);
        mlx::core::eval(m_parameters);
    }

    virtual Prediction forward(
        mlx::core::array input, std::span<const mlx::core::array> parameters) const = 0;

private:
    Prediction evaluate(mlx::core::array input,
                        std::span<const mlx::core::array> parameters) const;
    std::vector<mlx::core::array> computeLoss(
        const std::vector<mlx::core::array>& parameters,
        const TrainingBatch& batch, float weightDecay) const;
    void createCompiledForward() const;

    // Metadata that belongs together is one structure. Values remain a flat
    // vector because MLX transformations exchange parameters in that format.
    const char* m_name;
    std::vector<ParameterDefinition> m_layout;
    std::vector<mlx::core::array> m_parameters;
    mutable CompiledForward m_compiledForward;
};

// Modules only need their construction-time dependency to accept new parameter
// tensors. The inference and training API belongs to the Network base itself.
template<typename T>
concept NetworkType = requires(T& network, std::string name,
                               mlx::core::array parameter)
{
    { network.addParameter(std::move(name), std::move(parameter)) } -> std::same_as<size_t>;
};

// A fully connected layer. For every vector on the input's last axis it computes
// output = input * weight + bias. Leading axes such as batch and token are kept.
template<size_t Input, size_t Output, bool Bias = true>
class Linear
{
public:
    template<NetworkType N>
    Linear(N& network, std::string_view name)
        : m_weightIndex(network.addParameter(
              childName(name, "weight"),
              mlx::core::random::normal(
                  {static_cast<int>(Input), static_cast<int>(Output)}, mlx::core::float32,
                  0.0f, 1.0f / std::sqrt(static_cast<float>(Input)))))
        , m_biasIndex(addBias(network, name))
    {
    }

    mlx::core::array operator()(mlx::core::array input,
                                std::span<const mlx::core::array> parameters) const
    {
        assert(input.ndim() > 0);
        assert(input.shape(input.ndim() - 1) == Input);
        assert(m_weightIndex < parameters.size());

        mlx::core::array output = mlx::core::matmul(input, parameters[m_weightIndex]);
        if constexpr (Bias)
        {
            assert(m_biasIndex < parameters.size());
            output = output + parameters[m_biasIndex];
        }
        return output;
    }

private:
    template<NetworkType N>
    static size_t addBias(N& network, std::string_view name)
    {
        if constexpr (Bias)
            return network.addParameter(
                childName(name, "bias"),
                mlx::core::zeros({static_cast<int>(Output)}, mlx::core::float32));
        else
            return 0;
    }

    // Indices into Network::m_parameters, assigned during construction.
    size_t m_weightIndex;
    size_t m_biasIndex; // Unused when Bias is false.
};

// Normalizes each Width-element token, then applies a learned scale and shift.
// This keeps the numerical scale presented to the next module predictable.
template<size_t Width>
class LayerNorm
{
public:
    template<NetworkType N>
    LayerNorm(N& network, std::string_view name)
        : m_scaleIndex(network.addParameter(
              childName(name, "scale"),
              mlx::core::ones({static_cast<int>(Width)}, mlx::core::float32)))
        , m_shiftIndex(network.addParameter(
              childName(name, "shift"),
              mlx::core::zeros({static_cast<int>(Width)}, mlx::core::float32)))
    {
    }

    mlx::core::array operator()(mlx::core::array input,
                                std::span<const mlx::core::array> parameters) const
    {
        assert(input.ndim() > 0);
        assert(input.shape(input.ndim() - 1) == Width);
        assert(m_scaleIndex < parameters.size());
        assert(m_shiftIndex < parameters.size());
        return mlx::core::fast::layer_norm(input, parameters[m_scaleIndex], parameters[m_shiftIndex], 1e-5f);
    }

private:
    size_t m_scaleIndex;
    size_t m_shiftIndex;
};

// Activations have no trainable parameters. Their templated constructors only
// let Sequential construct every child through the same (network, name) form.
struct Relu
{
    Relu() = default;
    template<NetworkType N> Relu(N&, std::string_view) {}
    mlx::core::array operator()(mlx::core::array input, std::span<const mlx::core::array>) const
    {
        return mlx::core::maximum(input, mlx::core::array(0.0f));
    }
};

struct Gelu
{
    Gelu() = default;
    template<NetworkType N> Gelu(N&, std::string_view) {}
    mlx::core::array operator()(mlx::core::array input, std::span<const mlx::core::array>) const
    {
        return input * 0.5f * (mlx::core::erf(input * 0.70710678118654752f) + 1.0f);
    }
};

struct Sigmoid
{
    Sigmoid() = default;
    template<NetworkType N> Sigmoid(N&, std::string_view) {}
    mlx::core::array operator()(mlx::core::array input, std::span<const mlx::core::array>) const
    {
        return mlx::core::sigmoid(input);
    }
};

struct Tanh
{
    Tanh() = default;
    template<NetworkType N> Tanh(N&, std::string_view) {}
    mlx::core::array operator()(mlx::core::array input, std::span<const mlx::core::array>) const
    {
        return mlx::core::tanh(input);
    }
};

template<int Axis>
struct Softmax
{
    Softmax() = default;
    template<NetworkType N> Softmax(N&, std::string_view) {}
    mlx::core::array operator()(mlx::core::array input, std::span<const mlx::core::array>) const
    {
        return mlx::core::softmax(input, Axis);
    }
};

// Applies its children from left to right:
// Sequential<A, B, C>(x) is exactly C(B(A(x))). The tuple owns concrete module
// types, so this composition introduces no virtual calls.
template<typename... Modules>
class Sequential
{
public:
    template<NetworkType N>
    Sequential(N& network, std::string_view name)
        : Sequential(network, name, std::index_sequence_for<Modules...>{})
    {
    }

    mlx::core::array operator()(mlx::core::array input,
                                std::span<const mlx::core::array> parameters) const
    {
        return apply<0>(std::move(input), parameters);
    }

private:
    template<NetworkType N, size_t... Indices>
    Sequential(N& network, std::string_view name, std::index_sequence<Indices...>)
        : m_modules{Modules{network, indexedName(name, Indices)}...}
    {
    }

    template<size_t Index>
    mlx::core::array apply(mlx::core::array input,
                           std::span<const mlx::core::array> parameters) const
    {
        if constexpr (Index == sizeof...(Modules))
            return input;
        else
            return apply<Index + 1>(std::get<Index>(m_modules)(std::move(input), parameters), parameters);
    }

    std::tuple<Modules...> m_modules;
};

// Owns Count separate copies of one module type and applies them in order. The
// copies have different parameter names and therefore learn different values;
// this is how a stack of transformer blocks is represented.
template<size_t Count, typename ModuleType>
class Repeat
{
public:
    template<NetworkType N>
    Repeat(N& network, std::string_view name)
        : m_modules(makeModules(network, name, std::make_index_sequence<Count>{}))
    {
    }

    mlx::core::array operator()(mlx::core::array input,
                                std::span<const mlx::core::array> parameters) const
    {
        return apply<0>(std::move(input), parameters);
    }

private:
    template<NetworkType N, size_t... Indices>
    static std::array<ModuleType, Count> makeModules(N& network, std::string_view name,
                                                      std::index_sequence<Indices...>)
    {
        return {ModuleType{network, indexedName(name, Indices)}...};
    }

    template<size_t Index>
    mlx::core::array apply(mlx::core::array input,
                           std::span<const mlx::core::array> parameters) const
    {
        if constexpr (Index == Count)
            return input;
        else
            return apply<Index + 1>(m_modules[Index](std::move(input), parameters), parameters);
    }

    std::array<ModuleType, Count> m_modules;
};

// A residual connection asks a module to learn a correction instead of replacing
// its input completely: output = input + module(input).
template<typename ModuleType>
class Residual
{
public:
    template<NetworkType N>
    Residual(N& network, std::string_view name)
        : m_module(network, name)
    {
    }

    mlx::core::array operator()(mlx::core::array input,
                                std::span<const mlx::core::array> parameters) const
    {
        const mlx::core::array residual = input;
        return residual + m_module(std::move(input), parameters);
    }

private:
    ModuleType m_module;
};

// Multi-head self-attention over TokenCount tokens. Each token builds a query
// (what am I looking for?), a key (what information do I contain?), and a value
// (what information should I send?). Dot products between queries and keys decide
// how strongly every token reads every other token.
//
// RelationMap adds application-specific relationships to those scores, such as
// whether two board positions are identical, unrelated, or a known distance apart.
template<size_t TokenCount, size_t Width, size_t HeadCount, typename RelationMap>
class RelationSelfAttention
{
    static_assert(Width % HeadCount == 0, "attention width must be divisible by its head count");
    static_assert(RelationMap::tokenCount == TokenCount, "relation map token count does not match attention");

public:
    template<NetworkType N>
    RelationSelfAttention(N& network, std::string_view name)
        : m_query(network, childName(name, "query"))
        , m_key(network, childName(name, "key"))
        , m_value(network, childName(name, "value"))
        , m_output(network, childName(name, "output"))
        , m_relationBiasIndex(network.addParameter(
              childName(name, "relation_bias"),
              mlx::core::zeros(
                  {static_cast<int>(HeadCount), static_cast<int>(RelationMap::bucketCount)},
                  mlx::core::float32)))
    {
    }

    mlx::core::array operator()(mlx::core::array input,
                                std::span<const mlx::core::array> parameters) const
    {
        assert(input.ndim() == 3);
        assert(input.shape(1) == TokenCount);
        assert(input.shape(2) == Width);
        assert(m_relationBiasIndex < parameters.size());

        const int batchSize = input.shape(0);

        // [batch, token, width] -> [batch, head, token, width/head]. Each head
        // gets its own smaller representation and can learn a different relation.
        const mlx::core::array query = splitHeads(m_query(input, parameters));
        const mlx::core::array key = splitHeads(m_key(input, parameters));
        const mlx::core::array value = splitHeads(m_value(std::move(input), parameters));

        // Look up one learned bias per (head, relation bucket), then arrange it
        // as one additional score for every source/destination token pair.
        const mlx::core::array bias = mlx::core::reshape(
            mlx::core::take(parameters[m_relationBiasIndex], RelationMap::indices(), 1),
            {1, static_cast<int>(HeadCount), static_cast<int>(TokenCount), static_cast<int>(TokenCount)});

        // Softmax turns the scores into read weights that sum to one. Dividing by
        // sqrt(width/head) prevents large dot products from saturating softmax.
        const float scale = 1.0f / std::sqrt(static_cast<float>(Width) / static_cast<float>(HeadCount));
        mlx::core::array scores = mlx::core::matmul(query, mlx::core::transpose(key, {0, 1, 3, 2}));
        scores = mlx::core::softmax(scores * scale + bias, -1);

        // Weighted sums of value vectors collect the requested information. The
        // heads are then joined and mixed once more by the output Linear layer.
        const mlx::core::array attended = mergeHeads(mlx::core::matmul(scores, value), batchSize);
        return m_output(attended, parameters);
    }

private:
    static mlx::core::array splitHeads(const mlx::core::array& input)
    {
        return mlx::core::transpose(
            mlx::core::reshape(input,
                               {input.shape(0), static_cast<int>(TokenCount), static_cast<int>(HeadCount),
                                static_cast<int>(Width / HeadCount)}),
            {0, 2, 1, 3});
    }

    static mlx::core::array mergeHeads(const mlx::core::array& input, int batchSize)
    {
        return mlx::core::reshape(mlx::core::transpose(input, {0, 2, 1, 3}),
                                  {batchSize, static_cast<int>(TokenCount), static_cast<int>(Width)});
    }

    Linear<Width, Width, false> m_query;
    Linear<Width, Width, false> m_key;
    Linear<Width, Width, false> m_value;
    Linear<Width, Width, false> m_output;
    size_t m_relationBiasIndex;
};

struct AdamConfig
{
    float meanDecay = 0.9f;
    float varianceDecay = 0.999f;
    float epsilon = 1e-8f;
};

// Adam is gradient descent with a running estimate of each parameter's usual
// gradient direction and magnitude. This lets one learning rate work across
// parameter tensors whose raw gradient scales differ substantially.
class Adam
{
public:
    explicit Adam(const std::vector<mlx::core::array>& parameters, AdamConfig config = {});

    std::vector<mlx::core::array> updateParameters(const std::vector<mlx::core::array>& parameters,
                                                    const std::vector<mlx::core::array>& gradients, float rate);

    int steps() const { return m_steps; }

private:
    AdamConfig m_config;
    std::vector<mlx::core::array> m_mean;
    std::vector<mlx::core::array> m_variance;
    int m_steps = 0;
};

} // namespace amoeba_bot
