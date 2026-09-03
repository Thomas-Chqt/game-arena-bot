group "default" {
  targets = ["amoeba-cuda"]
}

target "mlx-cuda" {
  context    = "."
  dockerfile = "docker/Dockerfile.mlx"
  platforms  = ["linux/amd64"]
  tags = [
    "game-arena-bot:mlx-v0.32.2",
  ]
}

target "amoeba-cuda" {
  context    = "."
  dockerfile = "docker/Dockerfile"
  platforms  = ["linux/amd64"]
  tags = [
    "ghcr.io/thomas-chqt/game-arena-bot:latest",
  ]
}
