"""
This script demonstrates how to evaluate the ALOHA sim insertion human policy from HuggingFace Hub.

It requires the installation of the 'gym_aloha' simulation environment. Install it by running:
```bash
pip install -e ".[aloha]"
```
"""

from pathlib import Path

import gym_aloha  # noqa: F401
import gymnasium as gym
import imageio
import numpy
import torch

from lerobot.common.policies.act.modeling_act import ACTPolicy

# Create a directory to store the video of the evaluation
output_directory = Path("outputs/eval/example_aloha_sim_insertion")
output_directory.mkdir(parents=True, exist_ok=True)

# Select your device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Provide the HuggingFace repo id for ALOHA sim insertion human model
pretrained_policy_path = "lerobot/act_aloha_sim_insertion_human"
# OR a path to a local outputs/train folder if you have one
# pretrained_policy_path = Path("outputs/train/example_aloha_sim_insertion")

# Load the ACT policy (ALOHA uses ACT - Action Chunking with Transformers)
policy = ACTPolicy.from_pretrained(pretrained_policy_path)
policy.to(device)

# Initialize evaluation environment for ALOHA sim insertion task
# This environment simulates bimanual manipulation for insertion tasks
env = gym.make(
    "gym_aloha/AlohaInsertion-v0",
    obs_type="pixels_agent_pos",  # Use pixel observations WITH agent position (robot state)
    max_episode_steps=400,  # ALOHA tasks typically need more steps
)

# We can verify that the shapes of the features expected by the policy match the ones from the observations
# produced by the environment
print("Policy input features:")
print(policy.config.input_features)
print("\nEnvironment observation space:")
print(env.observation_space)

# Similarly, we can check that the actions produced by the policy will match the actions expected by the
# environment
print("\nPolicy output features:")
print(policy.config.output_features)
print("\nEnvironment action space:")
print(env.action_space)

# Reset the policy and environments to prepare for rollout
policy.reset()
numpy_observation, info = env.reset(seed=42)
print("numpy_observation keys: ", numpy_observation.keys())
print("info keys: ", info.keys())

# Prepare to collect every rewards and all the frames of the episode,
# from initial state to final state.
rewards = []
frames = []

# Render frame of the initial state
frames.append(env.render())

step = 0
done = False
print(f"Starting evaluation of the policy in the environment...")


while not done:
    # Prepare observation for the policy running in Pytorch
    state = torch.from_numpy(numpy_observation["agent_pos"])
    image = torch.from_numpy(numpy_observation["pixels"])

    # Convert to float32 with image from channel first in [0,255]
    # to channel last in [0,1]
    state = state.to(torch.float32)
    image = image.to(torch.float32) / 255
    image = image.permute(2, 0, 1)

    # Send data tensors from CPU to GPU
    state = state.to(device, non_blocking=True)
    image = image.to(device, non_blocking=True)

    # Add extra (empty) batch dimension, required to forward the policy
    state = state.unsqueeze(0)
    image = image.unsqueeze(0)

    # Create the policy input dictionary
    observation = {
        "observation.state": state,
        "observation.image": image,
    }

    # Predict the next action with respect to the current observation
    with torch.inference_mode():
        action = policy.select_action(observation)

    # Prepare the action for the environment
    numpy_action = action.squeeze(0).to("cpu").numpy()

    # Step through the environment and receive a new observation
    numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
    print(f"{step=} {reward=} {terminated=}")

    # Keep track of all the rewards and frames
    rewards.append(reward)
    frames.append(env.render())

    # The rollout is considered done when the success state is reached (i.e. terminated is True),
    # or the maximum number of iterations is reached (i.e. truncated is True)
    done = terminated | truncated | done
    step += 1

if terminated:
    print("Success!")
else:
    print("Failure!")

# Get the speed of environment (i.e. its number of frames per second).
fps = env.metadata["render_fps"]

# Encode all frames into a mp4 video.
video_path = output_directory / "rollout.mp4"
imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

print(f"Video of the evaluation is available in '{video_path}'.")
