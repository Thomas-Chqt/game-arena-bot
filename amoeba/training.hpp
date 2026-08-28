#pragma once

#include <filesystem>

namespace amoeba_bot
{

int runTraining(const std::filesystem::path& weights,
                const std::filesystem::path& executable);
int runSelfPlayWorker(const std::filesystem::path& weights);
int runEvaluationWorker(const std::filesystem::path& weights);

} // namespace amoeba_bot
