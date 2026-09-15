#!/usr/bin/env python3
"""L4: train an AI dispatcher (conflict resolution) via bluesky-gym + stable-baselines3.
Reproducible pipeline. Run AFTER scripts/patch_numpy2_bluesky.py.

Usage:
  BLUESKY_HEADLESS=1 python scripts/train_l4_dispatcher.py [--env MergeEnv-v0] [--steps 50000] [--save /tmp/ppo]
"""
import argparse, gc
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import bluesky_gym
from stable_baselines3 import PPO

bluesky_gym.register_envs()

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="HorizontalCREnv-v0")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--save", default="/tmp/ppo_airdispatcher")
    return p.parse_args()

def make_env(name):
    e = gym.make(name, render_mode=None)
    return FlattenObservation(e)

def main():
    a = parse()
    env = make_env(a.env)
    print(f"Training PPO on {a.env} for {a.steps} steps -> {a.save}")
    model = PPO("MlpPolicy", env, verbose=0, n_steps=256, batch_size=64,
                policy_kwargs=dict(net_arch=[64,64]), seed=42)
    model.learn(total_timesteps=a.steps)
    model.save(a.save)
    print("saved", a.save)
    # quick deterministic eval
    o, _ = env.reset(); tot = 0.0; n = 0; d = False
    for _ in range(60):
        act, _ = model.predict(o, deterministic=True)
        o, r, d, tr, _ = env.step(act); tot += r; n += 1
        if d or tr: o, _ = env.reset()
    print("eval_mean_reward=%.3f" % (tot / max(n, 1)))
    env.close()

if __name__ == "__main__":
    main()
