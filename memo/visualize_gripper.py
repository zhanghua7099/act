import h5py
import numpy as np
import matplotlib.pyplot as plt

dataset_path = 'data/sim_transfer_cube_scripted/episode_0.hdf5'

with h5py.File(dataset_path, 'r') as f:
    action = f['action'][:]               # (400, 14)
    qpos   = f['observations/qpos'][:]    # (400, 14)

left_cmd     = action[:, 6]
right_cmd    = action[:, 13]
left_actual  = qpos[:, 6]
right_actual = qpos[:, 13]
timesteps = np.arange(len(action))

left_events = [
    (0,   'sleep'),
    (100, 'approach meet'),
    (260, 'move to meet'),
    (310, 'close gripper'),
    (360, 'move left'),
]
right_events = [
    (0,   'sleep'),
    (90,  'approach cube'),
    (130, 'go down'),
    (170, 'close gripper'),
    (200, 'approach meet'),
    (220, 'move to meet'),
    (310, 'open gripper'),
    (360, 'move right'),
]

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

for ax, cmd, actual, title, color, events in zip(
    axes,
    [left_cmd, right_cmd],
    [left_actual, right_actual],
    ['Left Gripper', 'Right Gripper'],
    ['steelblue', 'tomato'],
    [left_events, right_events],
):
    ax.plot(timesteps, cmd,    color=color, label='command (action)')
    ax.plot(timesteps, actual, color=color, linestyle='--', alpha=0.7, label='actual (qpos)')
    ax.set_title(title)
    ax.set_ylabel('[0=close, 1=open]')
    ax.set_ylim(-0.15, 1.15)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.axhline(1, color='gray', linewidth=0.5, linestyle=':')
    for t, label in events:
        ax.axvline(t, color='green', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.text(t + 2, 1.07, label, fontsize=7, color='green', rotation=30)
    ax.legend()

axes[1].set_xlabel('Timestep')
plt.tight_layout()
plt.savefig('gripper_commands.png', dpi=150)
print('Saved to gripper_commands.png')
