from dataclasses import dataclass
import random

@dataclass
class ArmStats:
    wins: int = 0
    trials: int = 0

class DuelingBandit:
    """
    Minimal dueling bandit. Replace with RUCB/Relative-UCB for stronger guarantees.
    Tracks arm preference using wins/trials; selection based on posterior mean.
    """
    def __init__(self, arms):
        self.arms = list(arms)
        self.stats = {a: ArmStats() for a in self.arms}
        self.last_arm = None
        self.tokens_used = 0

    def select(self):
        if self.last_arm is None:
            self.last_arm = random.choice(self.arms)
            return self.last_arm
        challenger = random.choice([a for a in self.arms if a != self.last_arm])
        sA, sB = self.stats[self.last_arm], self.stats[challenger]
        scoreA = (sA.wins + 1) / (sA.trials + 2)
        scoreB = (sB.wins + 1) / (sB.trials + 2)
        self.last_arm = self.last_arm if scoreA >= scoreB else challenger
        return self.last_arm

    def update(self, arm, reward, tokens=0):
        st = self.stats[arm]
        st.trials += 1
        st.wins += 1 if reward > 0 else 0
        self.tokens_used += tokens