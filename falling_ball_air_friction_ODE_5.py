import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 设置随机种子
def set_seed(seed=3074):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed()

# 物理参数
g = 9.81
rho = 1.225
Cd = 0.47
r = 0.033
A = np.pi * r ** 2
m = 0.057
dt = 0.01
steps = 1000  # 模拟10秒
restitution = 0.7  # 碰撞恢复系数

# 神经网络模型
class ODEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            # nn.Linear(20, 10),
            # nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

    def forward(self, x):
        return self.net(x)

# 真实物理导数函数
def true_derivatives(x, v):
    dxdt = v
    dvdt = -g - (0.5 * rho * Cd * A * v * torch.abs(v)) / m
    return torch.stack([dxdt, dvdt], dim=-1)

# 生成训练数据
def generate_physics_data(num_samples=3000):
    inputs, targets = [], []
    for _ in range(num_samples):
        x = torch.tensor([np.random.uniform(0, 20)], dtype=torch.float32)
        v = torch.tensor([np.random.uniform(-20, 20)], dtype=torch.float32)
        for _ in range(3):
            deriv = true_derivatives(x, v)
            inputs.append(torch.cat([x, v]))
            targets.append(deriv[0])
            x = x + deriv[0, 0] * dt
            v = v + deriv[0, 1] * dt
            if x <= 0 and v < 0:
                v = -v * restitution
                x = torch.tensor([0.0])
    return torch.stack(inputs), torch.stack(targets)

# RK4 积分（含事件）
def rk4_with_event(model, x0, v0, steps=1000, dt=0.01, restitution=0.7):
    traj = []
    state = torch.tensor([x0, v0], dtype=torch.float32).unsqueeze(0)
    for _ in range(steps):
        with torch.no_grad():
            k1 = model(state)
            k2 = model(state + 0.5 * dt * k1)
            k3 = model(state + 0.5 * dt * k2)
            k4 = model(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if state[0, 0] <= 0 and state[0, 1] < 0:
            state[0, 0] = 0.0
            state[0, 1] = -state[0, 1] * restitution
        traj.append(state.squeeze(0).clone())
    return torch.stack(traj)

# 真实轨迹（输出位置 + 速度）
def integrate_true_physics_full(x0, v0, steps=1000, dt=0.01, restitution=0.7):
    traj = []
    state = torch.tensor([x0, v0], dtype=torch.float32)
    for _ in range(steps):
        dxdt, dvdt = true_derivatives(state[0:1], state[1:2])[0]
        state[0] += dxdt * dt
        state[1] += dvdt * dt
        if state[0] <= 0 and state[1] < 0:
            state[0] = 0.0
            state[1] = -state[1] * restitution
        traj.append(state.clone())
    return torch.stack(traj)

# 训练函数
def train_model(model, inputs, targets, epochs=1000, lr=0.0005):
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            pred = model(batch_x)
            pointwise_loss = criterion(pred, batch_y)

            # 每20轮加入 trajectory-level loss
            if epoch % 20 == 0:
                state = batch_x[0]
                with torch.no_grad():
                    true_traj = integrate_true_physics_full(state[0].item(), state[1].item(), steps=200)
                pred_traj = rk4_with_event(model, state[0].item(), state[1].item(), steps=200)
                traj_loss = criterion(pred_traj, true_traj)
                loss = pointwise_loss + 0.5 * traj_loss
            else:
                loss = pointwise_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.6f}")

# 开始训练
model = ODEModel()
inputs, targets = generate_physics_data(3000)
train_model(model, inputs, targets, epochs=1000)



# 可视化
x0, v0 = 5.0, -4.0
t_vals = np.linspace(0, steps * dt, steps)
pred_traj = rk4_with_event(model, x0, v0, steps=steps, dt=dt, restitution=restitution).numpy()
true_traj = integrate_true_physics_full(x0, v0, steps=steps, dt=dt, restitution=restitution).numpy()



# 输入任意位置和速度进行预测
test_input = torch.tensor([[5, -4]], dtype=torch.float32)
with torch.no_grad():
    output = model(test_input)

print("preiction results：", output.numpy())

# # 可视化：位置对比
# plt.figure(figsize=(8, 5))
# plt.plot(t_vals, pred_traj[:, 0], label="Predicted x(t)", linestyle='--')
# plt.plot(t_vals, true_traj[:, 0], label="True x(t)", linestyle='-')
# plt.xlabel("Time (s)")
# plt.ylabel("Position (m)")
# plt.title("Trajectory Comparison: Position")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # 可视化：速度对比
# plt.figure(figsize=(8, 5))
# plt.plot(t_vals, pred_traj[:, 1], label="Predicted v(t)", linestyle='--')
# plt.plot(t_vals, true_traj[:, 1], label="True v(t)", linestyle='-')
# plt.xlabel("Time (s)")
# plt.ylabel("Velocity (m/s)")
# plt.title("Trajectory Comparison: Velocity")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error
# import numpy as np

# # position error
# position_rmse = np.sqrt(np.mean((pred_traj[:, 0] - true_traj[:, 0])**2))
# position_mae = np.mean(np.abs(pred_traj[:, 0] - true_traj[:, 0]))
#
# # velocity error
# velocity_rmse = np.sqrt(np.mean((pred_traj[:, 1] - true_traj[:, 1])**2))
# velocity_mae = np.mean(np.abs(pred_traj[:, 1] - true_traj[:, 1]))
#
# print(f"Position RMSE: {position_rmse:.4f} m")
# print(f"Position MAE: {position_mae:.4f} m")
# print(f"Velocity RMSE: {velocity_rmse:.4f} m/s")
# print(f"Velocity MAE: {velocity_mae:.4f} m/s")


import pandas as pd

# # Create a DataFrame
# data = {
#     "Step": np.arange(steps),
#     "Time (s)": np.round(t_vals, 4),
#     "Predicted Position (m)": np.round(pred_traj[:, 0], 6),
#     "True Position (m)": np.round(true_traj[:, 0], 6),
#     "Predicted Velocity (m/s)": np.round(pred_traj[:, 1], 6),
#     "True Velocity (m/s)": np.round(true_traj[:, 1], 6)
# }
#
# df = pd.DataFrame(data)
#
# # Save to CSV file
# df.to_csv("trajectory_comparison.csv", index=False)
#
# print("Saved to trajectory_comparison.csv")



# # Take one set of data, generalize through the model and plot, then show the data
# def visualize_random_training_sample(inputs, targets):
#     # 随机选一个样本
#     idx = np.random.randint(0, len(inputs))
#     sample_input = inputs[idx]
#     sample_target = targets[idx]
#     x0, v0 = sample_input[0].item(), sample_input[1].item()
#     dxdt_target, dvdt_target = sample_target[0].item(), sample_target[1].item()
#
#     # print the data set
#     print(f"choose a sample：")
#     print(f"initial position x0 = {x0:.4f} m")
#     print(f"initial velocity v0 = {v0:.4f} m/s")
#     print(f"corresponding derivative dx/dt = {dxdt_target:.4f} m/s, dv/dt = {dvdt_target:.4f} m/s²")
#
#     # real trajectory by integration
#     true_traj = integrate_true_physics_full(x0, v0, steps=steps, dt=dt, restitution=restitution).numpy()
#
#     # time axis
#     t_vals = np.linspace(0, steps * dt, steps)
#
#     # plot position vs time
#     plt.figure(figsize=(8, 5))
#     plt.plot(t_vals, true_traj[:, 0], label="True Position x(t)", color='b')
#     plt.xlabel("Time (s)")
#     plt.ylabel("Position (m)")
#     plt.title("True Trajectory: Position vs Time")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
#     # plot velocity vs time
#     plt.figure(figsize=(8, 5))
#     plt.plot(t_vals, true_traj[:, 1], label="True Velocity v(t)", color='r')
#     plt.xlabel("Time (s)")
#     plt.ylabel("Velocity (m/s)")
#     plt.title("True Trajectory: Velocity vs Time")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
# # plot
# visualize_random_training_sample(inputs, targets)

# # 保存所有权重和偏置到一个 txt 文件中
# with open("new_model_weights_biases.txt", "w") as f:
#      for name, param in model.named_parameters():
#          f.write(f"{name} (shape: {param.shape}):\n")
#          f.write(str(param.data.numpy()))
#          f.write("\n\n")

