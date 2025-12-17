import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ActorNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.model(x)


class CriticNet(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        # return shape (batch,) for convenience
        return self.model(x).squeeze(-1)


class ActorCriticAgent:
    """
    Simple on-policy Actor–Critic agent with one-step TD target.

    - select_action(state) → int
    - update(state, action, reward, next_state, done)
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 3,
        gamma: float = 0.99,
        lr_actor: float = 1e-5,
        lr_critic: float = 1e-4,
        device: str = None,
    ):
        # Auto-select GPU when available unless explicitly overridden.
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.actor = ActorNet(state_dim, n_actions).to(self.device)
        self.critic = CriticNet(state_dim).to(self.device)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.n_actions = n_actions

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> int:
        """Samples an action from π(·|state)."""
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs=probs)
        a = dist.sample()
        return int(a.item())

    def update(self, state, action, reward, next_state, done):
        """
        Performs a single gradient update step using one-step TD target.

        Args:
            state, next_state: np.ndarray
            action: int
            reward: float
            done: bool
        """
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)

        r = torch.tensor(reward, dtype=torch.float32, device=self.device)
        d = torch.tensor(done, dtype=torch.float32, device=self.device)

        # Critic: V(s), V(s')
        v_s = self.critic(state)
        with torch.no_grad():
            v_ns = self.critic(next_state) * (1.0 - d)

        td_target = r + self.gamma * v_ns
        td_error = td_target - v_s

        # ---- Critic loss ----
        critic_loss = td_error.pow(2).mean()
        self.optim_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
        self.optim_critic.step()

        # ---- Actor loss ----
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs=probs)
        logp = dist.log_prob(torch.tensor(action, device=self.device))
        entropy = dist.entropy()
        entropy_coeff = 0.01  # you can tune this
        actor_loss = -(logp * td_error.detach() + entropy_coeff * entropy)
        self.optim_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
        self.optim_actor.step()


