group "default" {
  targets = ["amoeba-cuda"]
}

target "amoeba-cuda" {
  context    = "."
  dockerfile = "docker/Dockerfile"
  platforms  = ["linux/amd64"]
  tags = [
    "ghcr.io/thomas-chqt/game-arena-bot:latest",
  ]
}
