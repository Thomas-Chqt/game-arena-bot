#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "mcts.hpp"

#include <mlx/mlx.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
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

namespace amoeba
{

// Describes how a parameter tensor gets its initial values. Initialization is
// performed once, when Network constructs its root module. FanInNormal scales
// random weights by the input width so that early activations do not explode as
// they pass through several layers.
struct Initialization
{
    enum class Kind { zeros, ones, fanInNormal, normal };

    static constexpr Initialization Zeros() { return {Kind::zeros, 0.0f}; }
    static constexpr Initialization Ones() { return {Kind::ones, 0.0f}; }
    static constexpr Initialization FanInNormal() { return {Kind::fanInNormal, 0.0f}; }
    static constexpr Initialization Normal(float standardDeviation) { return {Kind::normal, standardDeviation}; }

    Kind kind;
    float standardDeviation;
};

std::string childName(std::string_view parent, std::string_view child);
std::string indexedName(std::string_view parent, size_t index);

// There is no Module base class. A type is a module when it can be called with
// an input tensor and the parameter tensors used for this particular forward
// pass. This keeps every internal call statically typed and inlineable.
template<typename T>
concept Module = requires(const T& module, mlx::core::array input, std::span<const mlx::core::array> parameters)
{
    module(std::move(input), parameters);
};

template<const char* Name, Module Root>
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
    // This exact name is written into the checkpoint. It identifies both the
    // architecture and its dimensions, not a particular set of trained values.
    inline static constexpr const char* name = Name;

    // Constructing Root recursively constructs every child module. Each module
    // calls addParameter(), so construction also defines the complete parameter
    // layout and initializes the tensors in that same stable order.
    explicit Network(uint64_t seed) : Network(seed, true) {}

    explicit Network(const std::filesystem::path& checkpoint) : Network(0, false)
    {
        load(checkpoint);
    }

    // Training uses this overload. MLX supplies a replacement parameter vector
    // to value_and_grad, and the same module tree evaluates with those values.
    std::vector<mlx::core::array> operator()(
        mlx::core::array input, std::span<const mlx::core::array> parameters) const
    {
        assert(parameters.size() == m_parameters.size());
        for (size_t index = 0; index < parameters.size(); ++index)
        {
            assert(parameters[index].shape() == m_layout[index].shape);
            assert(parameters[index].dtype() == mlx::core::float32);
        }

        // MLX compile accepts one flat vector. The board is entry 0 and every
        // parameter follows in the layout order. Making parameters inputs is
        // essential: captured arrays would be frozen as constants by compile().
        std::vector<mlx::core::array> inputs;
        inputs.reserve(parameters.size() + 1);
        inputs.push_back(std::move(input));
        inputs.insert(inputs.end(), parameters.begin(), parameters.end());
        return m_compiledForward(inputs);
    }

    // Inference uses the parameters currently owned by this Network.
    std::vector<mlx::core::array> operator()(mlx::core::array input) const
    {
        return (*this)(std::move(input), m_parameters);
    }

    // Modules call this from their constructors and retain the returned index.
    // The actual tensor stays here in Network; a module owns only its structure.
    size_t addParameter(std::string parameterName, mlx::core::Shape shape, Initialization initialization)
    {
        assert(std::ranges::find(m_layout, parameterName, &ParameterDefinition::name) == m_layout.end());
        assert(!shape.empty());

        mlx::core::array value = [&]
        {
            switch (initialization.kind)
            {
                case Initialization::Kind::zeros:
                    return mlx::core::zeros(shape, mlx::core::float32);
                case Initialization::Kind::ones:
                    return mlx::core::ones(shape, mlx::core::float32);
                case Initialization::Kind::fanInNormal:
                    return mlx::core::random::normal(shape, mlx::core::float32, 0.0f, 1.0f / std::sqrt(static_cast<float>(shape.front())));
                case Initialization::Kind::normal:
                    assert(initialization.standardDeviation > 0.0f);
                    return mlx::core::random::normal(shape, mlx::core::float32, 0.0f, initialization.standardDeviation);
            }
            std::unreachable();
        }();

        const size_t index = m_parameters.size();
        m_layout.push_back({std::move(parameterName), std::move(shape)});
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
        mlx::core::save_safetensors(checkpoint.string(), tensors, {{"network", name}});
    }

private:
    Network(uint64_t seed, bool materializeRandomParameters)
        : m_root(createRoot(*this, seed))
        , m_compiledForward(compileForward(m_root))
    {
        // A newly initialized network should own concrete values immediately.
        // The checkpoint constructor skips this because load() replaces every
        // random tensor before any of those temporary values are needed.
        if (materializeRandomParameters)
            mlx::core::eval(m_parameters);
    }

    // The random seed must be set before Root registers and initializes its first
    // parameter. A helper is needed because member construction precedes the
    // Network constructor body.
    static Root createRoot(Network& network, uint64_t seed)
    {
        mlx::core::random::seed(seed);
        return Root{network, ""};
    }

    static CompiledForward compileForward(Root root)
    {
        // Store this function once for the lifetime of the Network. Its first
        // real call traces and compiles the graph; later calls with the same
        // number, shapes and dtypes reuse MLX's cached executable. Root is copied
        // into the lambda so copying a Network never leaves a dangling `this`.
        std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)> forward =
            [root = std::move(root)](const std::vector<mlx::core::array>& inputs)
        {
            assert(!inputs.empty());
            return root(inputs.front(), std::span{inputs}.subspan(1));
        };
        return mlx::core::compile(std::move(forward));
    }

    void load(const std::filesystem::path& checkpoint)
    {
        auto [tensors, metadata] = mlx::core::load_safetensors(checkpoint.string());

        const auto storedName = metadata.find("network");
        if (storedName == metadata.end() || storedName->second != name)
            throw std::runtime_error(std::format("{}: checkpoint is not for network {}", checkpoint.string(), name));
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

    // Metadata that belongs together is one structure. Values remain a flat
    // vector because MLX transformations exchange parameters in that format.
    std::vector<ParameterDefinition> m_layout;
    std::vector<mlx::core::array> m_parameters;
    Root m_root;
    CompiledForward m_compiledForward;
};

// A fully connected layer. For every vector on the input's last axis it computes
// output = input * weight + bias. Leading axes such as batch and token are kept.
template<size_t Input, size_t Output, bool Bias = true>
class Linear
{
public:
    template<typename NetworkType>
    Linear(NetworkType& network, std::string_view name)
        : m_weightIndex(network.addParameter(childName(name, "weight"), {static_cast<int>(Input), static_cast<int>(Output)}, Initialization::FanInNormal()))
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
    template<typename NetworkType>
    static size_t addBias(NetworkType& network, std::string_view name)
    {
        if constexpr (Bias)
            return network.addParameter(childName(name, "bias"), {static_cast<int>(Output)},
                                        Initialization::Zeros());
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
    template<typename NetworkType>
    LayerNorm(NetworkType& network, std::string_view name)
        : m_scaleIndex(network.addParameter(childName(name, "scale"), {static_cast<int>(Width)},
                                            Initialization::Ones()))
        , m_shiftIndex(network.addParameter(childName(name, "shift"), {static_cast<int>(Width)},
                                            Initialization::Zeros()))
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
    template<typename NetworkType> Relu(NetworkType&, std::string_view) {}
    mlx::core::array operator()(mlx::core::array input, std::span<const mlx::core::array>) const
    {
        return mlx::core::maximum(input, mlx::core::array(0.0f));
    }
};

struct Gelu
{
    Gelu() = default;
    template<typename NetworkType> Gelu(NetworkType&, std::string_view) {}
    mlx::core::array operator()(mlx::core::array input, std::span<const mlx::core::array>) const
    {
        return input * 0.5f * (mlx::core::erf(input * 0.70710678118654752f) + 1.0f);
    }
};

struct Sigmoid
{
    Sigmoid() = default;
    template<typename NetworkType> Sigmoid(NetworkType&, std::string_view) {}
    mlx::core::array operator()(mlx::core::array input, std::span<const mlx::core::array>) const
    {
        return mlx::core::sigmoid(input);
    }
};

struct Tanh
{
    Tanh() = default;
    template<typename NetworkType> Tanh(NetworkType&, std::string_view) {}
    mlx::core::array operator()(mlx::core::array input, std::span<const mlx::core::array>) const
    {
        return mlx::core::tanh(input);
    }
};

template<int Axis>
struct Softmax
{
    Softmax() = default;
    template<typename NetworkType> Softmax(NetworkType&, std::string_view) {}
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
    template<typename NetworkType>
    Sequential(NetworkType& network, std::string_view name)
        : Sequential(network, name, std::index_sequence_for<Modules...>{})
    {
    }

    mlx::core::array operator()(mlx::core::array input,
                                std::span<const mlx::core::array> parameters) const
    {
        return apply<0>(std::move(input), parameters);
    }

private:
    template<typename NetworkType, size_t... Indices>
    Sequential(NetworkType& network, std::string_view name, std::index_sequence<Indices...>)
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
    template<typename NetworkType>
    Repeat(NetworkType& network, std::string_view name)
        : m_modules(makeModules(network, name, std::make_index_sequence<Count>{}))
    {
    }

    mlx::core::array operator()(mlx::core::array input,
                                std::span<const mlx::core::array> parameters) const
    {
        return apply<0>(std::move(input), parameters);
    }

private:
    template<typename NetworkType, size_t... Indices>
    static std::array<ModuleType, Count> makeModules(NetworkType& network, std::string_view name,
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
    template<typename NetworkType>
    Residual(NetworkType& network, std::string_view name)
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
// RelationMap adds game geometry to those scores. For Amoeba it tells attention
// whether two tokens are the same hex, unrelated, or separated by a particular
// direction and distance.
template<size_t TokenCount, size_t Width, size_t HeadCount, typename RelationMap>
class RelationSelfAttention
{
    static_assert(Width % HeadCount == 0, "attention width must be divisible by its head count");
    static_assert(RelationMap::tokenCount == TokenCount, "relation map token count does not match attention");

public:
    template<typename NetworkType>
    RelationSelfAttention(NetworkType& network, std::string_view name)
        : m_query(network, childName(name, "query"))
        , m_key(network, childName(name, "key"))
        , m_value(network, childName(name, "value"))
        , m_output(network, childName(name, "output"))
        , m_relationBiasIndex(network.addParameter(
              childName(name, "relation_bias"),
              {static_cast<int>(HeadCount), static_cast<int>(RelationMap::bucketCount)},
              Initialization::Zeros()))
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

// Every hex produces six move logits and six sow/split logits. Flattening the
// resulting [37, 12] tensor gives exactly the game's 444 policy indices.
inline constexpr int policyOutputsPerHex = directionCount * 2;

// One relation number for every ordered pair of board hexes:
//   0      same hex
//   1      different hexes that do not share a straight line
//   2..37  one of six directions and one of six distances
// Attention uses this fixed table to select a learned score bias.
struct AmoebaRelationMap
{
    static constexpr size_t tokenCount = hexCount;
    static constexpr size_t bucketCount = 2 + directionCount * maximumMovableStackHeight;

    inline static constexpr auto buckets = []
    {
        std::array<std::array<uint8_t, tokenCount>, tokenCount> table{};
        for (auto& row : table)
            row.fill(1); // Different hexes that do not share a line.

        for (size_t source = 0; source < tokenCount; ++source)
        {
            table[source][source] = 0;
            for (uint8_t direction = 0; direction < directionCount; ++direction)
            {
                for (uint8_t distance = 1; distance <= maximumMovableStackHeight; ++distance)
                {
                    const int8_t destination =
                        destinationHex(static_cast<uint8_t>(source), direction, distance);
                    if (destination >= 0)
                    {
                        table[source][destination] = static_cast<uint8_t>(
                            2 + direction * maximumMovableStackHeight + distance - 1);
                    }
                }
            }
        }
        return table;
    }();

    // MLX's take() consumes a one-dimensional index tensor, so flatten the
    // compile-time [source][destination] table once in source-major order.
    inline static constexpr auto flattened = []
    {
        std::array<int32_t, tokenCount * tokenCount> result{};
        for (size_t source = 0; source < tokenCount; ++source)
        {
            for (size_t destination = 0; destination < tokenCount; ++destination)
                result[source * tokenCount + destination] = buckets[source][destination];
        }
        return result;
    }();

    static const mlx::core::array& indices()
    {
        static const mlx::core::array value(flattened.data(),
                                     mlx::core::Shape{static_cast<int>(flattened.size())}, mlx::core::int32);
        return value;
    }
};

template<size_t Width, size_t HeadCount>
class TransformerBlock
{
public:
    template<typename NetworkType>
    TransformerBlock(NetworkType& network, std::string_view name)
        : m_norm1(network, childName(name, "norm1"))
        , m_attention(network, childName(name, "attention"))
        , m_norm2(network, childName(name, "norm2"))
        , m_feedForward(network, childName(name, "feed_forward"))
    {
    }

    mlx::core::array operator()(mlx::core::array input,
                                std::span<const mlx::core::array> parameters) const
    {
        // First let every hex read information from the complete board.
        input = input + m_attention(m_norm1(input, parameters), parameters);

        // Then independently transform each hex's resulting feature vector.
        input = input + m_feedForward(m_norm2(input, parameters), parameters);
        return input;
    }

private:
    // This is the pre-normalization transformer form:
    // x = x + attention(norm(x)); x = x + mlp(norm(x)). The residual additions
    // preserve the old representation while each submodule learns a correction.
    LayerNorm<Width> m_norm1;
    RelationSelfAttention<hexCount, Width, HeadCount, AmoebaRelationMap> m_attention;
    LayerNorm<Width> m_norm2;
    Sequential<Linear<Width, Width * 4>, Gelu, Linear<Width * 4, Width>> m_feedForward;
};

struct Prediction
{
    mlx::core::array policy; // Raw move logits: [batch, 444].
    mlx::core::array value;  // Expected outcome for the side to move: [batch].
};

// The game-specific root module. Network owns the tensors; this class defines
// how those tensors are connected into a computation from encoded boards to a
// policy and a value.
template<size_t BlockCount, size_t Width, size_t HeadCount>
class AmoebaRoot
{
    static_assert(BlockCount > 0, "an Amoeba network needs at least one transformer block");
    static_assert(Width > 0, "an Amoeba network width must be positive");
    static_assert(HeadCount > 0, "an Amoeba network needs at least one attention head");
    static_assert(Width % HeadCount == 0, "network width must be divisible by its attention-head count");

public:
    template<typename NetworkType>
    AmoebaRoot(NetworkType& network, std::string_view)
        : m_embed(network, "embed")
        , m_positionIndex(network.addParameter(
              "position", {hexCount, static_cast<int>(Width)}, Initialization::Normal(0.02f)))
        , m_blocks(network, "blocks")
        , m_finalNorm(network, "final_norm")
        , m_policy(network, "policy")
        , m_value(network, "value")
    {
    }

    std::vector<mlx::core::array> operator()(
        mlx::core::array inputBatch, std::span<const mlx::core::array> parameters) const
    {
        assert(inputBatch.ndim() == 2);
        assert(inputBatch.shape(1) == encodedBoardSize);
        assert(inputBatch.dtype() == mlx::core::float32);
        assert(m_positionIndex < parameters.size());

        const int batchSize = inputBatch.shape(0);
        constexpr int hexFeatureCount = hexCount * featuresPerHex;

        // The encoder's first section contains 59 features for each of 37 hexes.
        // Reshape it so each hex becomes a token in the sequence.
        const mlx::core::array perHexFeatures = mlx::core::reshape(
            mlx::core::slice(inputBatch, {0, 0}, {batchSize, hexFeatureCount}),
            {batchSize, hexCount, featuresPerHex});

        // The remaining 8 values describe the whole position. Copy them onto
        // every token because side-to-move and history affect every board hex.
        const mlx::core::array globalFeatures = mlx::core::broadcast_to(
            mlx::core::reshape(
                mlx::core::slice(inputBatch, {0, hexFeatureCount}, {batchSize, encodedBoardSize}),
                {batchSize, 1, globalFeatureCount}),
            {batchSize, hexCount, globalFeatureCount});

        // Each of the 37 board hexes becomes one Width-element token. Attention
        // then lets every token read every other token before the two task heads
        // turn the shared representation into move logits and a position value.
        mlx::core::array tokens = m_embed(
            mlx::core::concatenate({perHexFeatures, globalFeatures}, 2), parameters);

        // A learned vector identifies each physical hex. Without it, attention
        // would see the tokens as an unordered collection.
        tokens = tokens + parameters[m_positionIndex];
        tokens = m_blocks(std::move(tokens), parameters);
        tokens = m_finalNorm(std::move(tokens), parameters);

        // One 12-value policy block per token; reshape preserves the Move::id
        // ordering described by policyOutputsPerHex.
        const mlx::core::array policy = mlx::core::reshape(m_policy(tokens, parameters),
                                                            {batchSize, moveIdCount});

        // Value concerns the entire position, so average the 37 token vectors
        // into one vector before mapping it to a scalar in [-1, 1].
        const mlx::core::array pooled = mlx::core::mean(tokens, std::vector<int>{1}, false);
        const mlx::core::array value = mlx::core::reshape(m_value(pooled, parameters), {batchSize});
        // MLX's C++ compile API represents every compiled function's outputs as
        // a vector. Entry 0 is the policy and entry 1 is the value.
        return {policy, value};
    }

private:
    Linear<featuresPerHex + globalFeatureCount, Width> m_embed;
    size_t m_positionIndex;
    Repeat<BlockCount, TransformerBlock<Width, HeadCount>> m_blocks;
    LayerNorm<Width> m_finalNorm;
    Linear<Width, policyOutputsPerHex> m_policy;
    Sequential<Linear<Width, Width>, Gelu, Linear<Width, 1>, Tanh> m_value;
};

// This is the active concrete architecture. Its name includes every dimension
// that changes checkpoint shapes; changing its size deliberately creates an
// incompatible network and therefore requires a new name.
inline constexpr char networkName[] = "amoeba-relation-transformer-v2-6x128x8";
using AmoebaNetwork = Network<networkName, AmoebaRoot<6, 128, 8>>;

class NetworkEvaluator
{
public:
    explicit NetworkEvaluator(const AmoebaNetwork& network)
        : m_network(network)
    {
    }

    void evaluate(std::span<const Board* const> boards, std::span<Evaluation> outputs);

private:
    const AmoebaNetwork& m_network;
};

struct TrainingBatch
{
    mlx::core::array input;        // Encoded boards: [batch, encodedBoardSize].
    mlx::core::array legal;        // 1 for legal policy entries, otherwise 0.
    mlx::core::array policyTarget; // Normalized MCTS visit counts: [batch, 444].
    mlx::core::array valueTarget;  // Final outcomes from side-to-move viewpoint.
};

TrainingBatch makeTrainingBatch(std::span<const Board* const> boards,
                                std::span<const VisitCounts> visits,
                                std::span<const float> outcomes);

std::vector<mlx::core::array> computeLoss(const AmoebaNetwork& network,
                                   const std::vector<mlx::core::array>& parameters,
                                   const TrainingBatch& batch, float weightDecay);

} // namespace amoeba

#endif // NETWORK_HPP
