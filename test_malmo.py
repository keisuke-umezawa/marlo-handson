#!/usr/bin/env python
# -*- coding: utf-8 -*-
import marlo


def make_env(env_seed=0):
    join_tokens = marlo.make(
        "MarLo-FindTheGoal-v0",
        params=dict(
            comp_all_commands=["move", "turn"],
            allowContinuousMovement=True,
            videoResolution=[336, 336],
        ))
    env = marlo.init(join_tokens[0])

    obs = env.reset()
    action = env.action_space.sample()
    obs, r, done, info = env.step(action)
    env.seed(int(env_seed))
    return env


env = make_env()
obs = env.reset()

print(env.action_space.n)

for i in range(100):
    action = env.action_space.sample()
    obs, r, done, info = env.step(action)
    print(action, r, done, info)
