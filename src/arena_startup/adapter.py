"""The pinned Arena OpenPI observation/action convention; no gripper inversion."""
import numpy as np
from openpi_client import image_tools
from vla_startup.evaluate import quat_to_axisangle


def policy_images(observation, size):
    def convert(key):
        # Both axes reverse: this is 180 degrees, not a single-axis flip.
        frame = np.ascontiguousarray(observation[key][::-1, ::-1])
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(frame, size, size))
    return convert('agentview_image'), convert('robot0_eye_in_hand_image')


def policy_state(observation):
    state = np.concatenate((observation['robot0_eef_pos'],
                            quat_to_axisangle(observation['robot0_eef_quat'], np),
                            observation['robot0_gripper_qpos']))
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError('Expected finite [eef_xyz, axis_angle_xyz, gripper_qpos_2] state')
    return state


def action_chunk(response, replan_steps):
    chunk = np.asarray(response['actions'])
    if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) < replan_steps or not np.isfinite(chunk).all():
        raise ValueError(f'Invalid policy action chunk: {chunk.shape}')
    return chunk
