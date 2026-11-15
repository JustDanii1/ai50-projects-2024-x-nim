# Nim AI with Q-Learning

A Python implementation of the classic mathematical strategy game **Nim** featuring an Artificial Intelligence agent trained via **Reinforcement Learning (Q-learning)** through self-play.

---

## Game Rules

1. The game starts with several piles of objects (default configuration: ``).
2. Two players take turns removing objects from the piles.
3. On each turn, a player must remove **at least 1 object**, and may remove **any number of objects**, provided they all come from the **same pile**.
4. **Misère play:** The player forced to take the **last remaining object loses** the game.

---

## How the AI Works

The AI is implemented in the `NimAI` class and learns an optimal policy using tabular **Q-learning**:

* **Q-Table (`self.q`)**: Maps `(state, action)` pairs to expected reward values, initialized to `0`.
* **Self-Play Training (`train(n)`)**: The agent plays `n` games against itself to update its Q-values:
  * Making a move that takes the last object results in a **`-1`** reward (loss).
  * Forcing the opponent to take the last object grants a **`+1`** reward (win).
  * Intermediate moves receive a **`0`** immediate reward and are updated using the maximum expected future reward (`best_future_reward`).
* **Exploration vs. Exploitation (ε-greedy)**: During training, the AI explores random moves with probability `epsilon` (default `0.1`) and exploits the highest-valued action otherwise. When playing against a human (`epsilon=False`), it strictly chooses optimal moves.

---

## Getting Started

### Prerequisites
* **Python 3.6+**
* No external dependencies required (uses only Python's built-in `random` and `time` modules).

### Installation & Usage

1. Clone the repository or save the script as `nim.py`:
   ```bash
   git clone https://github.com
   cd your-repo-name
   ```

2. Train the AI and start a match in Python (or append this snippet to the bottom of `nim.py`):
   ```python
   from nim import train, play

   # Train the AI over 10,000 games
   ai = train(10000)

   # Play against the trained AI (turn order is randomized by default)
   play(ai)
   ```

3. Run the game from your terminal:
   ```bash
   python nim.py
   ```

---

## Code Overview

* **`class Nim`**: Manages game state, valid actions (`available_actions`), turn switching, and win/loss conditions (`move`).
* **`class NimAI`**: Implements the Q-learning agent, Q-value updates (`update_q_value`), and ε-greedy action selection (`choose_action`).
* **`train(n)`**: Runs `n` simulated self-play games and returns a trained `NimAI` instance.
* **`play(ai, human_player=None)`**: Starts an interactive CLI match between a human and the AI. Pass `human_player=0` to go first or `human_player=1` to go second.

---

## License

This project is open-source and available under the [MIT License](LICENSE).
