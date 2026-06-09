import h5py
import numpy as np
import matplotlib.pyplot as plt

dataset_path = 'data/sim_transfer_cube_scripted/episode_0.hdf5'

with h5py.File(dataset_path, 'r') as f:
    action = f['action'][:]               # (400, 14)
    qpos   = f['observations/qpos'][:]    # (400, 14)

left_cmd    = action[:, 6]
right_cmd   = action[:, 13]
left_actual  = qpos[:, 6]
right_actual = qpos[:, 13]
timesteps = np.arange(len(action))

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

for ax, cmd, actual, title, color in zip(
    axes,
    [left_cmd, right_cmd],
    [left_actual, right_actual],
    ['Left Gripper', 'Right Gripper'],
    ['steelblue', 'tomato'],
):
    ax.plot(timesteps, cmd,    color=color, label='command (action)')
    ax.plot(timesteps, actual, color=color, linestyle='--', alpha=0.7, label='actual (qpos)')
    ax.set_title(title)
    ax.set_ylabel('[0=close, 1=open]')
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.axhline(1, color='gray', linewidth=0.5, linestyle=':')
    ax.legend()

axes[1].set_xlabel('Timestep')
plt.tight_layout()
plt.savefig('gripper_commands.png', dpi=150)
print('Saved to gripper_commands.png')
