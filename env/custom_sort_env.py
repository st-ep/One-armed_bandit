import gymnasium as gym
import numpy as np
from panda_gym.envs.core import RobotTaskEnv, Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet


class CustomSortTask(Task):
    """Custom sorting task with red and blue cubes."""
    
    def __init__(
        self,
        sim: PyBullet,
        distance_threshold: float = 0.05,
        goal_range: float = 0.1,
        obj_range: float = 0.3,
    ) -> None:
        super().__init__(sim)
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        
        # Goal sampling range
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, 0])
        
        # Object sampling range  
        self.obj_range_low = np.array([-obj_range / 2, -obj_range / 2, 0])
        self.obj_range_high = np.array([obj_range / 2, obj_range / 2, 0])
        
        # Create the scene once
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        """Create the scene with table, cubes, and target."""
        # Create standard panda_gym scene
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        
        # Create cubes
        half_extents = np.ones(3) * self.object_size / 2
        
        self.sim.create_box(
            body_name="red_cube",
            half_extents=half_extents,
            mass=1.0,
            position=np.array([0.0, 0.1, self.object_size / 2]),
            rgba_color=np.array([1.0, 0.0, 0.0, 1.0])
        )
        
        self.sim.create_box(
            body_name="blue_cube",
            half_extents=half_extents,
            mass=1.0,
            position=np.array([0.0, -0.1, self.object_size / 2]),
            rgba_color=np.array([0.0, 0.0, 1.0, 1.0])
        )
        
        # Create target (ghost cube for visualization)
        self.sim.create_box(
            body_name="target",
            half_extents=half_extents,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.0, 1.0, 0.0, 0.3])
        )
        
        # Set realistic friction for grasping
        self.sim.set_lateral_friction("red_cube", 0, 0.5)  # link=0 for base link
        self.sim.set_lateral_friction("blue_cube", 0, 0.5)

    def reset(self) -> None:
        """Reset task by sampling new goal and object positions."""
        # Sample new positions
        self.goal = self._sample_goal()
        red_pos = self._sample_object()
        blue_pos = self._sample_object()
        
        # Reposition objects (don't recreate them)
        self.sim.set_base_pose("red_cube", red_pos, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("blue_cube", blue_pos, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Sample a random goal position."""
        goal = np.random.uniform(self.goal_range_low, self.goal_range_high)
        goal[2] = self.object_size / 2  # Place on table surface
        return goal

    def _sample_object(self) -> np.ndarray:
        """Sample a random object position."""
        pos = np.random.uniform(self.obj_range_low, self.obj_range_high)
        pos[2] = self.object_size / 2  # Place on table surface
        return pos

    def get_obs(self) -> np.ndarray:
        """Get task observations (object positions).
        
        Note: Goal position is not included here as it's provided separately
        by the environment framework in obs['desired_goal'].
        """
        # Red cube position (3 values)
        red_pos = self.sim.get_base_position("red_cube")
        # Blue cube position (3 values)  
        blue_pos = self.sim.get_base_position("blue_cube")
        
        # Concatenate all observations
        obs = np.concatenate([red_pos, blue_pos])
        return obs.astype(np.float32)

    def get_achieved_goal(self) -> np.ndarray:
        """Get achieved goal (red cube position for now)."""
        return np.array(self.sim.get_base_position("red_cube"))

    def get_goal(self) -> np.ndarray:
        """Return the current goal."""
        return super().get_goal()

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> float:
        """Compute reward based on distance to goal.
    
        Note: Using red_cube position as achieved_goal for this prototype.
        In a full sorting task, this could be extended to track multiple 
        objects or sorting criteria (e.g., red cubes to red zone).
        """
        distance = np.linalg.norm(achieved_goal - desired_goal)
        distance = np.clip(distance, 0.0, 1.0)  # Clip to bound rewards and prevent divergence
        
        # Dense reward: negative distance
        reward = -distance
        
        # Success bonus
        if distance < self.distance_threshold:
            reward += 10.0
        
        return reward

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        """Check if task is successfully completed."""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return distance < self.distance_threshold


class CustomSortEnv(RobotTaskEnv):
    """Custom sorting environment with Panda robot."""
    
    def __init__(self, render_mode: str = "rgb_array", **kwargs) -> None:
        sim = PyBullet(render_mode=render_mode)
        robot = Panda(sim, control_type="ee")
        task = CustomSortTask(sim, **kwargs)
        super().__init__(robot=robot, task=task)


# Test script
if __name__ == "__main__":
    # Create environment
    env = CustomSortEnv()
    
    # Test reset and basic functionality
    obs, info = env.reset()
    print("Environment created successfully!")
    print(f"Observation shape: {obs['observation'].shape}")
    print(f"Goal shape: {obs['desired_goal'].shape}")
    print(f"Achieved goal shape: {obs['achieved_goal'].shape}")
    
    # Debug: Print actual observation values
    print("\n=== Debug: Initial State ===")
    print(f"Task observation (red+blue pos): {obs['observation']}")
    print(f"Desired goal: {obs['desired_goal']}")
    print(f"Achieved goal (red pos): {obs['achieved_goal']}")
    print(f"Initial distance to goal: {np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']):.3f}")
    
    # Test a few steps
    print("\n=== Testing Steps ===")
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        print(f"Step {i+1}: reward={reward:.3f}, distance={distance:.3f}, success={info.get('is_success', False)}")
    
    env.close()
    print("\nTest completed successfully!")