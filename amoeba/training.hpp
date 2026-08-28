#pragma once

#include <filesystem>
#include <string_view>

namespace amoeba_bot
{

int runTraining(const std::filesystem::path& weights,
                std::string_view networkIdentifier = {});

} // namespace amoeba_bot
