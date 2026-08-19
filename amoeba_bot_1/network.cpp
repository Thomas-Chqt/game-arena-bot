#include "network.hpp"

#include <cmath>
#include <format>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace bot
{

namespace
{

// x * 0.5 * (1 + erf(x / sqrt(2))). The only nonlinearity in the trunk besides
// the value head's tanh; without one, six stacked blocks would collapse
// algebraically into a single matrix.
mlx::core::array gelu(const mlx::core::array& x)
{
    return x * 0.5f * (mlx::core::erf(x * 0.70710678118654752f) + 1.0f);
}

// [batch, hex, width] -> [batch, head, hex, width/head], so each head attends
// over the 37 hexes independently.
mlx::core::array splitHeads(const mlx::core::array& x, int heads)
{
    return mlx::core::transpose(mlx::core::reshape(x, {x.shape(0), amoeba::kNumHexes, heads, x.shape(2) / heads}), {0, 2, 1, 3});
}

mlx::core::array mergeHeads(const mlx::core::array& x)
{
    return mlx::core::reshape(mlx::core::transpose(x, {0, 2, 1, 3}), {x.shape(0), amoeba::kNumHexes, x.shape(1) * x.shape(3)});
}

// One learned scalar per (head, bucket), gathered into the [head, hex, hex] shape
// the attention scores are in. This is where the board's geometry enters: the
// network is told which hexes lie on a line and how far apart, instead of having
// to recover it from game outcomes.
mlx::core::array bucketIndices()
{
    std::array<int32_t, amoeba::kNumHexes * amoeba::kNumHexes> flat{};
    for (int i = 0; i < amoeba::kNumHexes; ++i) {
        for (int j = 0; j < amoeba::kNumHexes; ++j) {
            flat[i * amoeba::kNumHexes + j] = kBucket[i][j];
        }
    }

    return mlx::core::array(flat.data(), mlx::core::Shape{amoeba::kNumHexes * amoeba::kNumHexes}, mlx::core::int32);
}

mlx::core::array initialTensor(const std::string& name, const mlx::core::Shape& shape)
{
    if (name.ends_with(".scale"))
        return mlx::core::ones(shape, mlx::core::float32);

    // Both the layer-norm shifts and every Linear bias start here, as does the
    // relative-position bias: no positional preference until training finds one.
    if (name.ends_with(".shift") || name.ends_with(".bias"))
        return mlx::core::zeros(shape, mlx::core::float32);

    // The position embedding is added to a token, not multiplied through it, so
    // fan-in scaling does not apply. Keep it small enough to nudge rather than
    // drown the features it is added to.
    if (name == "position")
        return mlx::core::random::normal(shape, mlx::core::float32, 0.0f, 0.02f);

    // Every remaining tensor is a matrix whose output sums shape[0] inputs.
    // 1/sqrt(fan-in) keeps that sum at roughly unit scale, so activations
    // neither vanish nor explode on the way through the blocks.
    const float fanIn = static_cast<float>(shape.front());
    return mlx::core::random::normal(shape, mlx::core::float32, 0.0f, 1.0f / std::sqrt(fanIn));
}

} // namespace

std::vector<std::pair<std::string, mlx::core::Shape>> Network::layout(NetworkShape shape)
{
    const int width  = shape.width;
    const int hidden = width * 4;

    // Linear weights are stored [in, out] so the forward pass is a plain matmul
    // with no transpose.
    std::vector<std::pair<std::string, mlx::core::Shape>> entries{
        {"embed.weight", {amoeba::kHexFeatures + amoeba::kGlobalFeatures, width}},
        {"embed.bias",   {width}},
        {"position",     {amoeba::kNumHexes, width}},
    };

    for (int block = 0; block < shape.blocks; ++block)
    {
        const std::string prefix = std::format("block{}.", block);
        entries.emplace_back(prefix + "norm1.scale", mlx::core::Shape{width});
        entries.emplace_back(prefix + "norm1.shift", mlx::core::Shape{width});
        entries.emplace_back(prefix + "attn.query", mlx::core::Shape{width, width});
        entries.emplace_back(prefix + "attn.key", mlx::core::Shape{width, width});
        entries.emplace_back(prefix + "attn.value", mlx::core::Shape{width, width});
        entries.emplace_back(prefix + "attn.out", mlx::core::Shape{width, width});
        entries.emplace_back(prefix + "attn.bias", mlx::core::Shape{shape.heads, kPositionBuckets});
        entries.emplace_back(prefix + "norm2.scale", mlx::core::Shape{width});
        entries.emplace_back(prefix + "norm2.shift", mlx::core::Shape{width});
        entries.emplace_back(prefix + "mlp.in", mlx::core::Shape{width, hidden});
        entries.emplace_back(prefix + "mlp.in.bias", mlx::core::Shape{hidden});
        entries.emplace_back(prefix + "mlp.out", mlx::core::Shape{hidden, width});
        entries.emplace_back(prefix + "mlp.out.bias", mlx::core::Shape{width});
    }

    entries.emplace_back("policy.weight", mlx::core::Shape{width, kPolicyPerHex});
    entries.emplace_back("policy.bias", mlx::core::Shape{kPolicyPerHex});
    entries.emplace_back("value.hidden", mlx::core::Shape{width, width});
    entries.emplace_back("value.hidden.bias", mlx::core::Shape{width});
    entries.emplace_back("value.out", mlx::core::Shape{width, 1});
    entries.emplace_back("value.out.bias", mlx::core::Shape{1});

    return entries;
}

Network::Network(NetworkShape shape, uint64_t seed) : m_shape(shape)
{
    // Seeding once and building in layout() order is what makes a generation
    // reproducible from its seed alone.
    mlx::core::random::seed(seed);

    for (const auto& [name, tensorShape] : layout(shape))
    {
        m_names.push_back(name);
        m_params.push_back(initialTensor(name, tensorShape));
    }
    mlx::core::eval(m_params);
}

Network::Network(const std::filesystem::path& checkpoint)
{
    // MLX spells paths as std::string, so the conversion lives here rather than
    // spreading through every caller.
    const auto [tensors, metadata] = mlx::core::load_safetensors(checkpoint.string());

    m_shape.blocks = std::stoi(metadata.at("blocks"));
    m_shape.width  = std::stoi(metadata.at("width"));
    m_shape.heads  = std::stoi(metadata.at("heads"));

    // Reading through layout() rather than over the file means a checkpoint from
    // a different architecture is rejected here instead of at the first matmul.
    for (const auto& [name, tensorShape] : layout(m_shape))
    {
        const auto found = tensors.find(name);
        if (found == tensors.end())
            throw std::runtime_error(std::format("{}: no tensor named {}", checkpoint.string(), name));
        if (found->second.shape() != tensorShape)
            throw std::runtime_error(std::format("{}: {} has the wrong shape", checkpoint.string(), name));

        m_names.push_back(name);
        m_params.push_back(found->second);
    }
    mlx::core::eval(m_params);
}

void Network::save(const std::filesystem::path& checkpoint) const
{
    std::unordered_map<std::string, mlx::core::array> tensors;
    for (size_t i = 0; i < m_names.size(); ++i)
        tensors.emplace(m_names[i], m_params[i]);

    mlx::core::save_safetensors(checkpoint.string(), tensors, {
        {"blocks", std::format("{}", m_shape.blocks)},
        {"width",  std::format("{}", m_shape.width)},
        {"heads",  std::format("{}", m_shape.heads)}
    });
}

size_t Network::parameterCount() const
{
    return std::accumulate(m_params.begin(), m_params.end(), size_t{0}, [](size_t total, const mlx::core::array& tensor) {
        return total + tensor.size();
    });
}

void Network::replaceParameters(std::vector<mlx::core::array> params)
{
    if (params.size() != m_params.size())
        throw std::runtime_error(std::format("expected {} parameter tensors, got {}", m_params.size(), params.size()));
    m_params = std::move(params);
}

Prediction forward(const std::vector<mlx::core::array>& params, NetworkShape shape, const mlx::core::array& input)
{
    // Read strictly in layout() order. Each tensor gets its own local because the
    // order in which C++ evaluates two reads inside one expression is unspecified.
    size_t cursor = 0;
    const auto next = [&params, &cursor]() -> const mlx::core::array& { return params[cursor++]; };

    const int     batch    = input.shape(0);
    constexpr int hexBlock = amoeba::kNumHexes * amoeba::kHexFeatures;

    const mlx::core::array hexes = mlx::core::reshape(
        mlx::core::slice(input, {0, 0}, {batch, hexBlock}),
        {batch, amoeba::kNumHexes, amoeba::kHexFeatures}
    );

    // The globals go to every token: staleness or being in check changes what
    // every hex on the board means.
    const mlx::core::array globals = mlx::core::broadcast_to(
        mlx::core::reshape(
            mlx::core::slice(input, {0, hexBlock}, {batch, amoeba::kEncodedSize}),
            {batch, 1, amoeba::kGlobalFeatures}),
        {batch, amoeba::kNumHexes, amoeba::kGlobalFeatures}
    );

    const mlx::core::array& embedWeight = next();
    const mlx::core::array& embedBias   = next();
    const mlx::core::array& position    = next();

    mlx::core::array x = mlx::core::matmul(mlx::core::concatenate({hexes, globals}, 2), embedWeight);
    x = x + embedBias;
    x = x + position;

    const mlx::core::array buckets  = bucketIndices();
    const int headDim  = shape.width / shape.heads;
    const float scoreScale = 1.0f / std::sqrt(static_cast<float>(headDim));

    for (int block = 0; block < shape.blocks; ++block)
    {
        const mlx::core::array& norm1Scale = next();
        const mlx::core::array& norm1Shift = next();
        const mlx::core::array& queryWeight = next();
        const mlx::core::array& keyWeight = next();
        const mlx::core::array& valueWeight = next();
        const mlx::core::array& outWeight = next();
        const mlx::core::array& positionBias = next();

        const mlx::core::array normed = mlx::core::fast::layer_norm(x, norm1Scale, norm1Shift, 1e-5f);

        const mlx::core::array query = splitHeads(mlx::core::matmul(normed, queryWeight), shape.heads);
        const mlx::core::array key = splitHeads(mlx::core::matmul(normed, keyWeight), shape.heads);
        const mlx::core::array value = splitHeads(mlx::core::matmul(normed, valueWeight), shape.heads);

        const mlx::core::array bias = mlx::core::reshape(
            mlx::core::take(positionBias, buckets, 1),
            {1, shape.heads, amoeba::kNumHexes, amoeba::kNumHexes}
        );

        mlx::core::array scores = mlx::core::matmul(query, mlx::core::transpose(key, {0, 1, 3, 2}));
        scores = scores * scoreScale;
        scores = mlx::core::softmax(scores + bias, -1);

        const mlx::core::array attended = mergeHeads(mlx::core::matmul(scores, value));
        x = x + mlx::core::matmul(attended, outWeight);

        const mlx::core::array& norm2Scale = next();
        const mlx::core::array& norm2Shift = next();
        const mlx::core::array& mlpIn = next();
        const mlx::core::array& mlpInBias  = next();
        const mlx::core::array& mlpOut = next();
        const mlx::core::array& mlpOutBias = next();

        const mlx::core::array hidden = mlx::core::matmul(mlx::core::fast::layer_norm(x, norm2Scale, norm2Shift, 1e-5f), mlpIn);
        const mlx::core::array activated = gelu(hidden + mlpInBias);
        x = x + (mlx::core::matmul(activated, mlpOut) + mlpOutBias);
    }

    const mlx::core::array& policyWeight = next();
    const mlx::core::array& policyBias = next();

    // [batch, hex, 12] flattens straight onto Move::id, because the 12 run
    // direction-major and splitting-minor and id is (hex * 6 + dir) * 2 + split.
    const mlx::core::array logits = mlx::core::matmul(x, policyWeight) + policyBias;
    const mlx::core::array policy = mlx::core::reshape(logits, {batch, amoeba::kNumMoveIds});

    const mlx::core::array& valueWeight = next();
    const mlx::core::array& valueBias = next();
    const mlx::core::array& valueOutWeight = next();
    const mlx::core::array& valueOutBias = next();

    // Value is a property of the whole position, so the 37 tokens collapse to one.
    const mlx::core::array pooled = mlx::core::mean(x, std::vector<int>{1}, false);
    const mlx::core::array hidden = gelu(mlx::core::matmul(pooled, valueWeight) + valueBias);
    const mlx::core::array scalar = mlx::core::matmul(hidden, valueOutWeight) + valueOutBias;

    return {policy, mlx::core::tanh(mlx::core::reshape(scalar, {batch}))};
}

void NetworkEvaluator::evaluate(std::span<const amoeba::Board* const> boards, std::span<Evaluation> out)
{
    const int batch = static_cast<int>(boards.size());

    std::vector<float> encoded(static_cast<size_t>(batch) * amoeba::kEncodedSize);
    std::vector<float> mask(static_cast<size_t>(batch) * amoeba::kNumMoveIds);

    for (int i = 0; i < batch; ++i)
    {
        const size_t base = static_cast<size_t>(i) * amoeba::kEncodedSize;
        amoeba::encode(*boards[i], std::span<float, amoeba::kEncodedSize>(encoded.data() + base, amoeba::kEncodedSize));

        const std::span<const uint16_t, amoeba::kNumMoveIds> toAbsolute = amoeba::policyToAbsolute(boards[i]->whiteToMove);
        const size_t maskBase = static_cast<size_t>(i) * amoeba::kNumMoveIds;
        for (int slot = 0; slot < amoeba::kNumMoveIds; ++slot)
            mask[maskBase + slot] = boards[i]->isLegal(toAbsolute[slot]) ? 1.0f : 0.0f;
    }

    const mlx::core::array input(encoded.data(), mlx::core::Shape{batch, amoeba::kEncodedSize}, mlx::core::float32);
    const Prediction prediction = forward(m_network.parameters(), m_network.shape(), input);

    // A large finite penalty rather than -inf: a terminal position has no legal
    // moves at all, and softmax over an all -inf row is NaN, which would spread
    // into every parameter that touched it. The search creates no edges there, so
    // the resulting uniform row is never read.
    const mlx::core::array legal(mask.data(), mlx::core::Shape{batch, amoeba::kNumMoveIds}, mlx::core::float32);
    const mlx::core::array masked = mlx::core::where(mlx::core::greater(legal, mlx::core::array(0.0f)), prediction.policy, mlx::core::array(-1e9f));

    // The search sums the priors of the legal moves and divides, so it needs
    // probabilities; raw logits would give it negative priors.
    const mlx::core::array probabilities = mlx::core::softmax(masked, -1);
    mlx::core::eval({probabilities, prediction.value});

    const float* policyData = probabilities.data<float>();
    const float* valueData = prediction.value.data<float>();

    for (int i = 0; i < batch; ++i)
    {
        const std::span<const uint16_t, amoeba::kNumMoveIds> toAbsolute = amoeba::policyToAbsolute(boards[i]->whiteToMove);
        const size_t base = static_cast<size_t>(i) * amoeba::kNumMoveIds;

        for (int slot = 0; slot < amoeba::kNumMoveIds; ++slot)
            out[i].policy[toAbsolute[slot]] = policyData[base + slot];
        out[i].value = valueData[i];
    }
}

} // namespace bot
