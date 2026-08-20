# Repository Guidelines

This repository implements a competitive bot for the **Amoeba** board game on Game Arena. The goal is to implement the **AlphaZero algorithm**: a neural network learns from self-play and guides Monte Carlo Tree Search (MCTS) when choosing moves.

The authoritative game rules are in [`amoeba/amoeba-reference.md`](amoeba/amoeba-reference.md). Read that document before changing game logic, move generation, terminal conditions, or board encoding.

## Technology Stack

The project is written in **C++23** and built with **CMake**. It targets **macOS on Apple silicon** and runs natively on the ARM64 architecture. Neural-network training and inference use Apple's **MLX** framework. Online play and communication with Game Arena use the **Game Arena C SDK**.

## Working With the Project Owner

The project owner is highly experienced with C++, but has limited AI and machine-learning experience. They understand the basics of neurons, weights, and training with gradient descent, but do not yet have much knowledge beyond those foundations. When discussing or implementing the AlphaZero, neural-network, or training parts of the project, explain the relevant AI concepts and the reasoning behind design choices in clear, concrete terms. Do not assume familiarity with machine-learning terminology; define it when needed and connect explanations to the code so the project owner can understand and evaluate the work. There is no need to explain ordinary C++ concepts unless asked.
