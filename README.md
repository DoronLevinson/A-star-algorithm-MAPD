# Multi-Agent Package Delivery (MAPD) with A* Algorithm

This project implements a multi-agent package delivery (MAPD) problem where agents and saboteurs operate in a grid environment. Agents aim to deliver packages to their respective destinations while avoiding blocked and fragile edges, whereas saboteurs aim to block paths by breaking fragile edges. The A* algorithm is used to compute optimal paths for the agents.

![image](https://github.com/user-attachments/assets/a81d220b-688c-4865-bb96-acd9e8218a48)


## Overview

In this MAPD scenario, agents must deliver packages from pickup locations to delivery locations within certain time constraints, while avoiding blocked paths. Fragile edges, which can be broken by saboteurs, create additional obstacles in the environment. The A* algorithm is used to compute the optimal paths for the agents, considering the dynamic nature of the environment.

## A* Algorithm

The A* algorithm is a widely-used pathfinding and graph traversal algorithm, especially for scenarios where the goal is to find the shortest path from a start node to a goal node. It is an informed search algorithm that efficiently balances between exploring paths that seem promising and paths that are necessary to ensure the optimal solution.

How A* Works
A* uses a cost function, f(n), to evaluate each node n in the search space:

![image](https://github.com/user-attachments/assets/14278f8c-71fe-4a51-914a-510c35e50090)


Where:

* g(n) is the actual cost to reach the node n from the start node. This is the sum of the step costs incurred along the path.
* h(n) is the heuristic estimate of the cost to reach the goal node from n. It serves as a "guess" of the remaining cost and helps prioritize nodes that appear closer to the goal.

A* maintains a priority queue (or open set) of nodes to explore, always expanding the node with the lowest f(n) value. As long as the heuristic h(n) is admissible (i.e., never overestimates the true cost to the goal), A* is guaranteed to find the shortest possible path.

In each step, the algorithm:

1. Picks the node with the smallest f(n) from the priority queue.
2. Expands that node by exploring its neighbors, updating their g(n) and f(n) values.
3. Repeats until it reaches the goal or exhausts the search space.

### Real-Time A* and Unlimited A* Differences

* **Unlimited A*** explores the entire search space until a solution is found. The agent waits for the algorithm to find the optimal path before moving, making it slower in large environments but ensuring the best route is taken.
* **Real-Time A*** limits how much of the search space is explored before the agent moves. For example, the agent may move after expanding 10 nodes, even if the solution isn't complete. This allows the agent to act quickly, re-evaluating the environment as it moves. While this sacrifices optimality, it ensures fast decisions in dynamic, time-constrained environments.

## Input Explanation

The input to the program is a text file that defines the environment, including:
- Grid dimensions (X, Y)
- Package information (pickup and delivery locations, time constraints)
- Blocked edges (permanent obstacles)
- Fragile edges (can be broken by saboteurs)
- Agents and saboteurs' initial positions

### Example Input

#X 4 <br />
#Y 4 <br />
#P 3 2 0  D 3 2 4 <br />
#P 0 2 0  D 0 2 500 <br />
#B 0 2 1 2 <br />
#A 1 0 <br />

## Usage

1. Clone the repository and navigate to the project directory.
2. Create input files in the specified format.
3. Run the Python script:

   ```ruby
   python mapd_a_star.py
   ```
## Requirements
This project uses Python 3.x and requires no external libraries beyond Python’s standard library.

## License
This project is licensed under the MIT License.
