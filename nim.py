
import random
import time


class Nim:
    """The Nim game and its rules."""

    def __init__(self, initial=[1, 3, 5, 7]):
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        """Return all possible moves."""
        actions = set()
        for pile, size in enumerate(piles):
            for count in range(1, size + 1):
                actions.add((pile, count))
        return actions

    @classmethod
    def other_player(cls, player):
        return 0 if player == 1 else 1

    def switch_player(self):
        self.player = Nim.other_player(self.player)

    def move(self, action):
        pile, count = action

        if self.winner is not None:
            raise Exception("Game already won")
        elif not 0 <= pile < len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        self.piles[pile] -= count
        self.switch_player()

        if all(p == 0 for p in self.piles):
            self.winner = self.player


class NimAI:
    """AI that learns to play Nim using Q-learning."""

    def __init__(self, alpha=0.5, epsilon=0.1):
        self.q = {}
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        """Update the Q-value after a move."""
        old_q = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old_q, reward, best_future)

    def get_q_value(self, state, action):
        """Return the Q-value for a state and action."""
        return self.q.get((tuple(state), tuple(action)), 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """Update the Q-value using the learning formula."""
        estimate = reward + future_rewards
        self.q[(tuple(state), tuple(action))] = (
            old_q + self.alpha * (estimate - old_q)
        )

    def best_future_reward(self, state):
        """Return the best possible future reward."""
        actions = Nim.available_actions(state)

        if not actions:
            return 0

        return max(self.get_q_value(state, action) for action in actions)

    def choose_action(self, state, epsilon=True):
        """Choose a random or the best available move."""
        actions = list(Nim.available_actions(state))

        if not actions:
            return None

        if epsilon and random.random() < self.epsilon:
            return random.choice(actions)

        best_value = max(self.get_q_value(state, action) for action in actions)

        best_actions = [
            action for action in actions
            if self.get_q_value(state, action) == best_value
        ]

        return random.choice(best_actions)


def train(n):
    """Train the AI by playing against itself."""
    ai = NimAI()

    for game_no in range(1, n + 1):
        print(f"Playing training game {game_no}")
        game = Nim()

        last_move = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        while True:
            state = game.piles.copy()
            action = ai.choose_action(game.piles)

            last_move[game.player]["state"] = state
            last_move[game.player]["action"] = action

            game.move(action)
            new_state = game.piles.copy()

            if game.winner is not None:
                ai.update(state, action, new_state, -1)
                ai.update(
                    last_move[game.player]["state"],
                    last_move[game.player]["action"],
                    new_state,
                    1
                )
                break

            elif last_move[game.player]["state"] is not None:
                ai.update(
                    last_move[game.player]["state"],
                    last_move[game.player]["action"],
                    new_state,
                    0
                )

    print("Done training")
    return ai


def play(ai, human_player=None):
    """Play against the trained AI."""
    if human_player is None:
        human_player = random.randint(0, 1)

    game = Nim()

    while True:
        print()
        print("Piles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")
        print()

        available_actions = Nim.available_actions(game.piles)
        time.sleep(1)

        if game.player == human_player:
            print("Your Turn")
            while True:
                pile = int(input("Choose Pile: "))
                count = int(input("Choose Count: "))
                if (pile, count) in available_actions:
                    break
                print("Invalid move, try again.")

        else:
            print("AI's Turn")
            pile, count = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chose to take {count} from pile {pile}.")

        game.move((pile, count))

        if game.winner is not None:
            print()
            print("GAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return
