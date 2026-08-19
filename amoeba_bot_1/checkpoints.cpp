#include "checkpoints.hpp"

#include <charconv>
#include <cstdlib>
#include <format>
#include <string>

namespace bot
{

namespace
{

constexpr const char* kBestLink  = "best";
constexpr const char* kExtension = ".safetensors";

std::optional<int> generationOf(const std::filesystem::path& file)
{
    const std::string name = file.filename().string();
    if (!name.starts_with("gen-") || !name.ends_with(kExtension))
        return std::nullopt;

    const char* first = name.data() + 4;
    const char* last  = name.data() + name.size() - std::string_view(kExtension).size();

    int value{};
    if (std::from_chars(first, last, value).ptr != last)
        return std::nullopt;
    return value;
}

} // namespace

Checkpoints Checkpoints::fromEnvironment()
{
    const char* directory = std::getenv("AMOEBA_CHECKPOINTS");
    return Checkpoints(directory == nullptr ? "checkpoints" : directory);
}

Checkpoints::Checkpoints(std::filesystem::path directory) : m_directory(std::move(directory))
{
    std::filesystem::create_directories(m_directory);
}

std::filesystem::path Checkpoints::forGeneration(int generation) const
{
    return m_directory / std::format("gen-{:04}{}", generation, kExtension);
}

std::optional<std::filesystem::path> Checkpoints::best() const
{
    const std::filesystem::path link = m_directory / kBestLink;
    if (!std::filesystem::exists(link))
        return std::nullopt;
    return std::filesystem::canonical(link);
}

void Checkpoints::promote(int generation) const
{
    const std::filesystem::path staging = m_directory / std::format("{}.staging", kBestLink);
    std::filesystem::remove(staging);
    std::filesystem::create_symlink(forGeneration(generation).filename(), staging);
    std::filesystem::rename(staging, m_directory / kBestLink);
}

int Checkpoints::latestGeneration() const
{
    int latest = -1;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(m_directory))
        if (const std::optional<int> generation = generationOf(entry.path()))
            latest = std::max(latest, *generation);
    return latest;
}

} // namespace bot
