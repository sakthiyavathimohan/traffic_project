  
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ─── SUMO PATH SETUP ───
tools = os.path.join(os.environ.get("SUMO_HOME", "C:/Program Files (x86)/Eclipse/Sumo"), "tools")
sys.path.append(tools)
import traci

# ─── FUZZY REWARD SHAPING ───
def build_fuzzy():
    queue_input = ctrl.Antecedent(np.arange(0, 31, 1), 'queue')
    delay_input = ctrl.Antecedent(np.arange(0, 201, 1), 'delay')
    reward_out  = ctrl.Consequent(np.arange(-1, 1.1, 0.1), 'reward')

    queue_input['low']    = fuzz.trimf(queue_input.universe, [0, 0, 10])
    queue_input['medium'] = fuzz.trimf(queue_input.universe, [5, 15, 25])
    queue_input['high']   = fuzz.trimf(queue_input.universe, [20, 30, 30])

    delay_input['low']    = fuzz.trimf(delay_input.universe, [0, 0, 60])
    delay_input['medium'] = fuzz.trimf(delay_input.universe, [40, 100, 160])
    delay_input['high']   = fuzz.trimf(delay_input.universe, [140, 200, 200])

    reward_out['very_low'] = fuzz.trimf(reward_out.universe, [-1, -1, -0.5])
    reward_out['low']      = fuzz.trimf(reward_out.universe, [-0.8, -0.4, 0])
    reward_out['medium']   = fuzz.trimf(reward_out.universe, [-0.2, 0, 0.2])
    reward_out['high']     = fuzz.trimf(reward_out.universe, [0, 0.4, 0.8])
    reward_out['very_high']= fuzz.trimf(reward_out.universe, [0.5, 1, 1])

    rules = [
        ctrl.Rule(queue_input['high']   & delay_input['high'],   reward_out['very_low']),
        ctrl.Rule(queue_input['high']   & delay_input['medium'], reward_out['low']),
        ctrl.Rule(queue_input['medium'] & delay_input['high'],   reward_out['low']),
        ctrl.Rule(queue_input['medium'] & delay_input['medium'], reward_out['medium']),
        ctrl.Rule(queue_input['low']    & delay_input['low'],    reward_out['very_high']),
        ctrl.Rule(queue_input['low']    & delay_input['medium'], reward_out['high']),
        ctrl.Rule(queue_input['medium'] & delay_input['low'],    reward_out['high']),
        ctrl.Rule(queue_input['high']   & delay_input['low'],    reward_out['medium']),
        ctrl.Rule(queue_input['low']    & delay_input['high'],   reward_out['low']),
    ]

    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)

fuzzy_sim = build_fuzzy()

def fuzzy_reward(queue, delay):
    try:
        fuzzy_sim.input['queue'] = min(float(queue), 30)
        fuzzy_sim.input['delay'] = min(float(delay), 200)
        fuzzy_sim.compute()
        return float(fuzzy_sim.output['reward'])
    except:
        return 0.0

# ─── DUELING DD-DQN NETWORK ───
class DuelingDDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.value_stream     = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value     = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# ─── AGENT ───
class DDQNAgent:
    def __init__(self, state_size=8, action_size=3):
        self.state_size   = state_size
        self.action_size  = action_size
        self.memory       = deque(maxlen=10000)
        self.gamma        = 0.95
        self.epsilon      = 1.0
        self.epsilon_min  = 0.01
        self.epsilon_decay= 0.995
        self.lr           = 0.0001
        self.batch_size   = 32
        self.device       = torch.device("cpu")

        self.online_net = DuelingDDQN(state_size, action_size).to(self.device)
        self.target_net = DuelingDDQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer  = optim.Adam(self.online_net.parameters(), lr=self.lr)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online_net(state_t).argmax().item()

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states)).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        curr_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_a = self.online_net(next_states).argmax(1)
        next_q = self.target_net(next_states).gather(1, next_a.unsqueeze(1)).squeeze(1)
        target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(curr_q, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

# ─── SUMO ENVIRONMENT ───
SUMO_CFG = "intersection.sumocfg"
PHASES = [
    "GGGggrrrrrGGGggrrrrr",
    "rrrrrGGGggrrrrrGGGgg",
    "GGGggrrrrrrrrrrrrrrr"
]
IN_LANES = ["north_in_0","north_in_1","south_in_0","south_in_1",
            "east_in_0","east_in_1","west_in_0","west_in_1"]

def get_state():
    return np.array([traci.lane.getLastStepHaltingNumber(l) for l in IN_LANES], dtype=np.float32)

def get_metrics():
    queues  = [traci.lane.getLastStepHaltingNumber(l) for l in IN_LANES]
    delays  = [traci.lane.getWaitingTime(l) for l in IN_LANES]
    through = sum(traci.lane.getLastStepVehicleNumber(l) for l in IN_LANES)
    return sum(queues), sum(delays), through

def apply_action(action, duration=10):
    traci.trafficlight.setRedYellowGreenState("center", PHASES[action])
    for _ in range(duration):
        traci.simulationStep()

# ─── TRAINING ───
def train(episodes=100):
    agent         = DDQNAgent()
    all_rewards   = []
    all_queues    = []
    all_delays    = []
    queue_history = []

    for ep in range(episodes):
        traci.start(["sumo", "-c", SUMO_CFG, "--no-warnings", "true"])
        state        = get_state()
        total_reward = 0
        ep_queue     = []
        step         = 0

        while traci.simulation.getMinExpectedNumber() > 0 and step < 500:
            action           = agent.act(state)
            apply_action(action, duration=10)
            next_state       = get_state()
            queue, delay, tp = get_metrics()

            base_reward  = -0.4 * queue - 0.4 * delay + 0.2 * tp
            shaped       = fuzzy_reward(queue / 30, delay / 200)
            reward       = base_reward + shaped * 10

            agent.remember(state, action, reward, next_state, False)
            agent.replay()

            state        = next_state
            total_reward += reward
            ep_queue.append(queue)
            step        += 1

        traci.close()

        if ep % 10 == 0:
            agent.update_target()

        avg_queue = np.mean(ep_queue) if ep_queue else 0
        all_rewards.append(total_reward)
        all_queues.append(avg_queue)
        all_delays.append(abs(total_reward))
        queue_history.extend(ep_queue[:50])

        winner = "N-S GREEN" if avg_queue > 5 else "E-W GREEN"
        print(f"Episode {ep+1:3d}/100 | Reward: {total_reward:10.1f} | Avg Queue: {avg_queue:.2f} | AI Decision: {winner}")

    return all_rewards, all_queues, all_delays, queue_history

# ─── PLOT RESULTS ───
def plot_results(rewards, queues, delays, queue_hist):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("DD-DQN + Fuzzy Reward Shaping — Traffic Signal Optimization", fontsize=13, fontweight='bold')

    axes[0,0].plot(queue_hist, color='steelblue', linewidth=0.8)
    axes[0,0].set_title("Queue Length over Simulation Steps")
    axes[0,0].set_xlabel("Step")
    axes[0,0].set_ylabel("Queue Length (vehicles)")

    axes[0,1].plot(rewards, color='tomato', linewidth=1.2)
    axes[0,1].set_title("Cumulative Reward per Episode")
    axes[0,1].set_xlabel("Episode")
    axes[0,1].set_ylabel("Total Reward")

    axes[1,0].plot(queues, color='seagreen', linewidth=1.2)
    axes[1,0].set_title("Average Queue Length per Episode")
    axes[1,0].set_xlabel("Episode")
    axes[1,0].set_ylabel("Avg Queue (vehicles)")

    axes[1,1].plot(delays, color='darkorange', linewidth=1.2)
    axes[1,1].set_title("Cumulative Delay per Episode")
    axes[1,1].set_xlabel("Episode")
    axes[1,1].set_ylabel("Delay")

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nResults saved as results.png")
    plt.show()

# ─── MAIN ───
if __name__ == "__main__":
    print("Starting DD-DQN Traffic Signal Training...")
    print("Episodes: 100 | This will take a few minutes\n")
    rewards, queues, delays, queue_hist = train(episodes=100)
    plot_results(rewards, queues, delays, queue_hist)
    print("\nTraining Complete!")
