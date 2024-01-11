from auto_policy_programming.wrappers.base import BaseWrapper

def low_timestep_obs_aug(obs):
    return f"Timestep left: low\n{obs}"


class LimitTimeStepWrapper(BaseWrapper):
    def __init__(self, env, max_episode_steps, low_episode_timestep):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.low_episode_timestep = low_episode_timestep
        self.timestep = 0

    def reset(self, *args, **kwargs):
        self.timestep = 0
        ret = super().reset(*args, **kwargs)
        obs = ret[0]
        obs = self.aug_observation(obs)
        return (obs, *ret[1:])

    def step(self, *args, **kwargs):
        self.timestep += 1
        obs, reward, done, info = super().step(*args, **kwargs)
        obs = self.aug_observation(obs)
        done = done or self.timestep >= self.max_episode_steps
        return (obs, reward, done, info)

    def aug_observation(self, obs):
        if self.max_episode_steps-self.timestep <= self.low_episode_timestep:
            return low_timestep_obs_aug(obs)
        return obs