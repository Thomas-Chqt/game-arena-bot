#include "network.hpp"

#include <cassert>
#include <format>
#include <numeric>
#include <stdexcept>

namespace amoeba
{

// Module composition produces names such as blocks.2.attention.query.weight.
// These names make the flat tensor vector readable and form the checkpoint keys.
std::string childName(std::string_view parent, std::string_view child)
{
    return parent.empty() ? std::string{child} : std::format("{}.{}", parent, child);
}

std::string indexedName(std::string_view parent, size_t index)
{
    return childName(parent, std::format("{}", index));
}

Adam::Adam(const std::vector<mlx::core::array>& parameters, AdamConfig config)
    : m_config(config)
{
    // Adam keeps two tensors with the same shape as every trained parameter:
    // a moving mean of gradients and a moving mean of squared gradients.
    for (const mlx::core::array& tensor : parameters)
    {
        m_mean.push_back(mlx::core::zeros(tensor.shape(), tensor.dtype()));
        m_variance.push_back(mlx::core::zeros(tensor.shape(), tensor.dtype()));
    }
}

std::vector<mlx::core::array> Adam::updateParameters(const std::vector<mlx::core::array>& parameters,
                                                      const std::vector<mlx::core::array>& gradients, float rate)
{
    if (parameters.size() != m_mean.size() || parameters.size() != gradients.size())
        throw std::runtime_error(std::format("Adam was built for {} tensors but got {} parameters and {} gradients",
                                             m_mean.size(), parameters.size(), gradients.size()));
    ++m_steps;

    // Both moving averages start at zero. These factors remove that initial bias
    // toward zero, which would otherwise make the first updates too small.
    const float meanScale = 1.0f - std::pow(m_config.meanDecay, m_steps);
    const float varianceScale = 1.0f - std::pow(m_config.varianceDecay, m_steps);

    std::vector<mlx::core::array> updated;
    updated.reserve(parameters.size());
    for (size_t index = 0; index < parameters.size(); ++index)
    {
        // Smooth noisy minibatch gradients into a direction and typical size.
        m_mean[index] = m_mean[index] * m_config.meanDecay + gradients[index] * (1.0f - m_config.meanDecay);
        m_variance[index] = m_variance[index] * m_config.varianceDecay
            + mlx::core::square(gradients[index]) * (1.0f - m_config.varianceDecay);

        // Divide by the typical gradient magnitude before taking the step.
        // epsilon only prevents division by zero for a parameter with no gradient.
        const mlx::core::array typicalGradient =
            mlx::core::sqrt(m_variance[index] / varianceScale) + m_config.epsilon;
        updated.push_back(parameters[index] - (m_mean[index] / meanScale) / typicalGradient * rate);
    }
    return updated;
}

} // namespace amoeba
