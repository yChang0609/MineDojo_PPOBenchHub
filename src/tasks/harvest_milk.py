import minedojo
from minedojo.sim.inventory import InventoryItem
from src.core.reward.harvest_dense import HarvestDenseRewardWrapper
import numpy as np

class HarvestMilk(HarvestDenseRewardWrapper):
    def __init__(
        self,
        image_size,
        step_penalty: float | int,
        nav_reward_scale: float | int,
        success_reward: float | int,
        max_spawn_range = 10,
        target_quantities = 1,
        max_episode_len = 500,
    ):
        distance_to_axis = int(max_spawn_range / np.sqrt(2))
        spawn_range_low = (-distance_to_axis, 1, -distance_to_axis)
        spawn_range_high = (distance_to_axis, 1, distance_to_axis)
        env = minedojo.make(
            "harvest",
            target_names=["milk_bucket"],
            target_quantities=target_quantities,
            reward_weights=
            {
                "milk_bucket": success_reward,
            },
            # start_position=pos,
            initial_inventory=[
                InventoryItem(slot=0, name="bucket", variant=None, quantity=1)
            ],
            initial_mobs=["cow"],
            initial_mob_spawn_range_low=spawn_range_low,
            initial_mob_spawn_range_high=spawn_range_high,
            image_size=image_size,
            world_seed=123,
            specified_biome="sunflower_plains",
            fast_reset=True,
            use_voxel=True,
            use_lidar=True,
            lidar_rays=[
                (pitch, yaw, 9999)
                for pitch in [np.deg2rad(x) for x in np.linspace(0, 30, 3)]
                for yaw in [np.deg2rad(x) for x in np.linspace(-60, 60, 9)]
            ],
        )
        super().__init__(
            env=env,
            entity="cow",
            step_penalty=step_penalty,
            nav_reward_scale=nav_reward_scale,
        )

        # reset cmds, call before `env.reset()`
        self._reset_cmds = ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]

        self._episode_len = max_episode_len
        self._elapsed_steps = 0
        self._first_reset = True

    def reset(self, **kwargs):
        self._elapsed_steps = 0

        if not self._first_reset:
            for cmd in self._reset_cmds:
                self.env.unwrapped.execute_cmd(cmd)
            self.unwrapped.set_time(6000)
            self.unwrapped.set_weather("clear")
        self._first_reset = False

        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._episode_len:
            done = True
        return obs, reward, done, info