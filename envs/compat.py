"""Gym/Gymnasium compatibility used without sibling-repository imports."""

try:  # Stable-Baselines3 >= 2
    import gymnasium as gym

    NEW_STEP_API = True
except ImportError:  # Stable-Baselines3 1.x baseline environment
    import gym

    NEW_STEP_API = False


def reset_result(observation, info=None):
    return (observation, info or {}) if NEW_STEP_API else observation


def step_result(observation, reward, terminated, truncated, info):
    if NEW_STEP_API:
        return observation, reward, terminated, truncated, info
    return observation, reward, bool(terminated or truncated), info

