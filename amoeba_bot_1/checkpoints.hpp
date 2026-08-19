#pragma once

// ---------------------------------------------------------------------------
// The one thing amoeba_bot_1 and amoeba_bot_1_train share at runtime: a
// directory of model files and a `best` link naming the one to play with.
//
// The trainer writes a numbered file per generation and only ever moves `best`
// when a candidate has actually beaten the incumbent over a match. The player
// only ever reads `best`. Nothing else coordinates the two, which is why they
// can be started, killed and restarted independently.
//
// Training does not improve monotonically - a newer generation is regularly
// weaker than the one before it. Pointing the player at the newest file instead
// of the gated one would make its rating a random walk over that noise.
// ---------------------------------------------------------------------------

#include <filesystem>
#include <optional>

namespace bot
{

class Checkpoints
{
public:
    // AMOEBA_CHECKPOINTS, or ./checkpoints. Creates the directory.
    static Checkpoints fromEnvironment();

    explicit Checkpoints(std::filesystem::path directory);

    const std::filesystem::path& directory() const { return m_directory; }

    std::filesystem::path forGeneration(int generation) const;

    // Nothing has been trained yet until the trainer promotes generation 0.
    std::optional<std::filesystem::path> best() const;

    // Repoints `best`. Atomic - written beside the link and renamed over it -
    // so a player loading a model at that moment gets the old one or the new
    // one, never a half-written link.
    void promote(int generation) const;

    // -1 when the directory holds no generations at all.
    int latestGeneration() const;

private:
    std::filesystem::path m_directory;
};

} // namespace bot
