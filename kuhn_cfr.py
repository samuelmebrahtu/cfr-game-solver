import numpy as np
from collections import defaultdict
import random

class KuhnCFR:
    """
    Counterfactual Regret Minimization for Kuhn Poker
    
    Kuhn Poker Rules:
    - 3 cards: J, Q, K
    - 2 players, each antes 1 chip
    - Each player gets 1 card
    - Actions: Pass (p) or Bet (b)
    - Player 1 acts first
    """
    
    def __init__(self):
        self.cards = ['J', 'Q', 'K']
        self.n_actions = 2  # Pass or Bet
        
        # Store cumulative regrets and strategy for each information set
        self.regret_sum = defaultdict(lambda: np.zeros(self.n_actions))
        self.strategy_sum = defaultdict(lambda: np.zeros(self.n_actions))
        self.n_info_sets = 0
        
    def get_strategy(self, info_set):
        """
        Get current strategy using regret matching
        """
        regrets = self.regret_sum[info_set]
        
        # Regret matching: strategy proportional to positive regrets
        strategy = np.maximum(regrets, 0)
        normalizing_sum = np.sum(strategy)
        
        if normalizing_sum > 0:
            strategy = strategy / normalizing_sum
        else:
            # Uniform random if no positive regrets
            strategy = np.ones(self.n_actions) / self.n_actions
            
        return strategy
    
    def get_action(self, strategy):
        """Sample an action according to strategy"""
        return np.random.choice(self.n_actions, p=strategy)
    
    def cfr(self, cards, history, player, reach_probs):
        """
        Counterfactual Regret Minimization recursive algorithm
        
        Args:
            cards: List of cards for [player0, player1]
            history: String of actions taken (e.g., "pb" = pass then bet)
            player: Current player (0 or 1)
            reach_probs: Reach probabilities [prob_player0, prob_player1]
        
        Returns:
            Expected utility for player 0
        """
        
        # Check if terminal state
        if self.is_terminal(history):
            return self.get_payoff(cards, history)
        
        # Create information set: player's card + history
        info_set = cards[player] + history
        
        # Get current strategy
        strategy = self.get_strategy(info_set)
        
        # Track this info set
        if info_set not in self.strategy_sum:
            self.n_info_sets += 1
        
        # Compute action utilities
        action_utils = np.zeros(self.n_actions)
        
        for action in range(self.n_actions):
            action_history = history + ('p' if action == 0 else 'b')
            
            # Update reach probabilities
            new_reach = reach_probs.copy()
            new_reach[player] *= strategy[action]
            
            # Recurse
            if player == 0:
                action_utils[action] = self.cfr(cards, action_history, 1, new_reach)
            else:
                action_utils[action] = self.cfr(cards, action_history, 0, new_reach)
        
        # Expected utility for this info set
        util = np.sum(strategy * action_utils)
        
        # Compute regrets
        regrets = action_utils - util
        
        # Update regret sum (weighted by opponent's reach probability)
        opponent = 1 - player
        self.regret_sum[info_set] += reach_probs[opponent] * regrets
        
        # Update strategy sum (weighted by player's reach probability)
        self.strategy_sum[info_set] += reach_probs[player] * strategy
        
        return util
    
    def is_terminal(self, history):
        """Check if game state is terminal"""
        # Terminal states:
        # "pp" = both pass
        # "pbb" = pass, bet, bet (call)
        # "pbp" = pass, bet, pass (fold)
        # "bp" = bet, pass (fold)
        # "bb" = bet, bet (call)
        
        if history in ["pp", "pbb", "pbp", "bp", "bb"]:
            return True
        return False
    
    def get_payoff(self, cards, history):
        """
        Get payoff for player 0 at terminal state
        Each player antes 1, bets are 1 chip
        """
        card_rank = {'J': 1, 'Q': 2, 'K': 3}
        
        # Determine winner
        if history == "pp":
            # Showdown, no bets
            if card_rank[cards[0]] > card_rank[cards[1]]:
                return 1  # Player 0 wins 1 chip
            else:
                return -1  # Player 0 loses 1 chip
                
        elif history == "pbp" or history == "bp":
            # Player folded
            if history == "pbp":
                return 1  # Player 1 folded, player 0 wins
            else:
                return -1  # Player 0 folded, player 1 wins
                
        elif history == "pbb" or history == "bb":
            # Showdown with bet
            if card_rank[cards[0]] > card_rank[cards[1]]:
                return 2  # Player 0 wins 2 chips (ante + bet)
            else:
                return -2  # Player 0 loses 2 chips
        
        return 0
    
    def train(self, iterations):
        """Train CFR for specified iterations"""
        util_sum = 0
        
        for i in range(iterations):
            # Shuffle and deal cards
            cards = self.cards.copy()
            random.shuffle(cards)
            cards = cards[:2]  # Deal 2 cards
            
            # Run CFR
            util = self.cfr(cards, "", 0, [1.0, 1.0])
            util_sum += util
            
            if (i + 1) % 10000 == 0:
                avg_util = util_sum / (i + 1)
                print(f"Iteration {i + 1}: Avg utility = {avg_util:.6f}")
        
        print(f"\nTraining complete! {self.n_info_sets} information sets found.")
    
    def get_average_strategy(self):
        """Get average strategy profile"""
        avg_strategy = {}
        
        for info_set in self.strategy_sum.keys():
            strategy_sum = self.strategy_sum[info_set]
            normalizing_sum = np.sum(strategy_sum)
            
            if normalizing_sum > 0:
                avg_strategy[info_set] = strategy_sum / normalizing_sum
            else:
                avg_strategy[info_set] = np.ones(self.n_actions) / self.n_actions
        
        return avg_strategy
    
    def print_strategy(self):
        """Print the computed strategy"""
        avg_strategy = self.get_average_strategy()
        
        print("\n=== Nash Equilibrium Strategy ===")
        print("Information Set | Pass | Bet")
        print("-" * 35)
        
        # Sort for readable output
        for info_set in sorted(avg_strategy.keys()):
            strategy = avg_strategy[info_set]
            print(f"{info_set:15} | {strategy[0]:.3f} | {strategy[1]:.3f}")
    
    def compute_exploitability(self):
        """
        Compute exploitability (approximation)
        Lower is better - measures distance from Nash equilibrium
        """
        exploitability = 0
        avg_strategy = self.get_average_strategy()
        
        for info_set in avg_strategy.keys():
            regrets = self.regret_sum[info_set]
            positive_regrets = np.maximum(regrets, 0)
            exploitability += np.sum(positive_regrets)
        
        # Normalize by number of information sets and iterations
        return exploitability / (self.n_info_sets * 1000)


def main():
    print("Training CFR on Kuhn Poker...\n")
    
    # Initialize and train
    cfr = KuhnCFR()
    cfr.train(iterations=100000)
    
    # Display results
    cfr.print_strategy()
    
    # Compute exploitability
    exploit = cfr.compute_exploitability()
    print(f"\nExploitability: {exploit:.6f}")
    print(f"Exploitability %: {exploit * 100:.4f}%")
    
    if exploit < 0.001:
        print("✓ Successfully converged to Nash equilibrium!")
    else:
        print("Run more iterations for better convergence")


if __name__ == "__main__":
    main()