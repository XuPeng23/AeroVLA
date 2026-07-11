from collections import OrderedDict
import copy
import random
import sys
import time
import numpy as np
import math
import os
import json
from pathlib import Path
import airsim
import random
from typing import Dict, List, Optional

import tqdm
from src.common.param import args
from utils.logger import logger
sys.path.append(str(Path(str(os.getcwd())).resolve()))
from airsim_plugin.AirVLNSimulatorClientTool_AeroVLA import AirVLNSimulatorClientTool
from utils.env_utils_uav import SimState
from utils.env_vector_uav import VectorEnvUtil
RGB_FOLDER = ['frontcamera', 'leftcamera', 'rightcamera', 'rearcamera', 'downcamera']
DEPTH_FOLDER = [name + '_depth' for name in RGB_FOLDER]

from scipy.spatial.transform import Rotation as R

def project_target_state2global_state_axis(this_target_state, target_state):  # local to global
    def to_eularian_angles(q):
        x,y,z,w = q
        ysqr = y * y
        t0 = +2.0 * (w*x + y*z)
        t1 = +1.0 - 2.0*(x*x + ysqr)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (w*y - z*x)
        if (t2 > 1.0):
            t2 = 1
        if (t2 < -1.0):
            t2 = -1.0
        pitch = math.asin(t2)
        t3 = +2.0 * (w*z + x*y)
        t4 = +1.0 - 2.0 * (ysqr + z*z)
        yaw = math.atan2(t3, t4)
        return (pitch, roll, yaw)
    def euler_to_rotation_matrix(e):
        rotation = R.from_euler('xyz', e, degrees=False)
        return rotation.as_matrix()
    start_pos = target_state['position']
    start_eular = to_eularian_angles(target_state['orientation'])
    this_pos = this_target_state['position']
    this_eular = to_eularian_angles(this_target_state['orientation'])
    rot = euler_to_rotation_matrix(start_eular) 
    this_global_pos = np.linalg.inv(rot).T @ np.array(this_pos) + np.array(start_pos)
    this_global_eular = np.array(this_eular) + np.array(start_eular)
    return {'position': this_global_pos.tolist(), 'orientation': this_global_eular.tolist()}

def prepare_object_map():
    with open(args.map_spawn_area_json_path, 'r') as f:
        map_dict = json.load(f)
    return map_dict

def find_closest_area(coord, areas):
    def euclidean_distance(coord1, coord2):
        return np.sqrt(sum((np.array(coord1) - np.array(coord2)) ** 2))
    min_distance = float('inf')
    closest_area = None
    closest_area_info = None
    for area in areas:
        if len(area) < 18:
            continue
        true_area = [area[0]+1, area[1]+1, area[2]+0.5]
        distance = euclidean_distance(coord, true_area)
        if distance < min_distance:
            min_distance = distance
            closest_area = true_area
            closest_area_info = area
    return closest_area, closest_area_info

class AirVLNENV:
    def __init__(self, batch_size=8, 
                 dataset_path=None,
                 save_path=None,
                 eval_json_path=None,
                 seed=1,
                 dataset_group_by_scene=True,
                 activate_maps=[]
                 ):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.eval_json_path = eval_json_path
        self.seed = seed
        self.collected_keys = set()
        self.dataset_group_by_scene = dataset_group_by_scene
        self.activate_maps = set(activate_maps)
        self.map_area_dict = prepare_object_map()
        self.exist_save_path = save_path
        load_data = self.load_my_datasets()
        self.ori_raw_data = load_data
        logger.info('Loaded dataset {}.'.format(len(self.eval_json_path)))
        self.index_data = 0
        self.data = self.ori_raw_data
        
        if dataset_group_by_scene:
            self.data = self._group_scenes()
            logger.warning('dataset grouped by scene, ')

        scenes = [item['map_name'] for item in self.data]
        self.scenes = set(scenes)
        self.sim_states: Optional[List[SimState]] = [None for _ in range(batch_size)]
        self.last_using_map_list = []
        self.one_scene_could_use_num = 5e3
        self.this_scene_used_cnt = 0
        self.init_VectorEnvUtil()

    def load_my_datasets(self):
        list_data_dict = json.load(open(self.eval_json_path, "r"))
        trajectorys_path = set()
        skipped_trajectory_set = set()
        data = []
        old_state = random.getstate()
        for item in list_data_dict:
            trajectorys_path.add(os.path.join(self.dataset_path, item['json']))
        for item in os.listdir(self.exist_save_path):
            item = item.replace('success_', '').replace('oracle_', '')
            skipped_trajectory_set.add(item)
        print('Loading dataset metainfo...')
        trajectorys_path = sorted(trajectorys_path)
        for merged_json in tqdm.tqdm(trajectorys_path):
            merged_json = merged_json.replace('data6', 'data5') # it is a fix since the mark.json saved on data5
            path_parts = merged_json.strip('/').split('/')
            map_name, seq_name = path_parts[-3], path_parts[-2]
            if (len(self.activate_maps) > 0 and map_name not in self.activate_maps) or seq_name in skipped_trajectory_set:
                continue
            mark_json = merged_json.replace('merged_data.json', 'mark.json')
            with open(mark_json, 'r') as f:
                mark_json = json.load(f)
                asset_name = mark_json['object_name']
                object_position = mark_json['target']['position']
                _, closest_area_info = find_closest_area(object_position, self.map_area_dict[map_name])
                object_position = [closest_area_info[9], closest_area_info[10], closest_area_info[11]]
                obj_pose = airsim.Pose(airsim.Vector3r(closest_area_info[9], closest_area_info[10], closest_area_info[11]), 
                                airsim.Quaternionr(closest_area_info[13], closest_area_info[14], closest_area_info[15], closest_area_info[12]))
                obj_scale = airsim.Vector3r(closest_area_info[17], closest_area_info[17], closest_area_info[17])
                asset_name = closest_area_info[16]
            traj_info = {}
            frames = []
            traj_dir = '/'.join(path_parts[:-1])
            traj_info['map_name'] = map_name
            traj_info['seq_name'] = seq_name
            traj_info['merged_json'] = merged_json
            with open(merged_json, 'r') as obj_f:
                merged_data = json.load(obj_f)
            frames = merged_data['trajectory_raw_detailed']
            traj_info['trajectory'] = frames
            traj_info['trajectory_dir'] = traj_dir
            traj_info['instruction'] = merged_data['conversations'][0]['value']
            traj_info['object'] = {'pose': obj_pose, 'scale': obj_scale, 'asset_name': asset_name}
            traj_info['object_position'] = object_position
            traj_info['length'] = len(frames)
            data.append(traj_info)
        random.setstate(old_state)      # Recover the state of the random generator
        return data
    
    def _group_scenes(self):
        assert self.dataset_group_by_scene, 'error args param'
        scene_sort_keys: OrderedDict[str, int] = {}
        for item in self.data:
            if str(item['map_name']) not in scene_sort_keys:
                scene_sort_keys[str(item['map_name'])] = len(scene_sort_keys)
        return sorted(self.data, key=lambda e: (scene_sort_keys[str(e['map_name'])], e['length']))

    def init_VectorEnvUtil(self):
        self.delete_VectorEnvUtil()
        self.VectorEnvUtil = VectorEnvUtil(self.scenes, self.batch_size)  # 

    def delete_VectorEnvUtil(self):
        if hasattr(self, 'VectorEnvUtil'):
            del self.VectorEnvUtil
        import gc
        gc.collect()

    def next_minibatch(self, skip_scenes=[], data_it=0):
        batch = []
        while True:
            if self.index_data >= len(self.data):
                random.shuffle(self.data)
                logger.warning('random shuffle data')
                if self.dataset_group_by_scene:
                    self.data = self._group_scenes()
                    logger.warning('dataset grouped by scene')

                if len(batch) == 0:
                    self.index_data = 0
                    self.batch = None
                    return

                self.index_data = self.batch_size - len(batch)
                batch += self.data[:self.index_data]
                break

            new_trajectory = self.data[self.index_data]

            if new_trajectory['map_name'] in skip_scenes:
                self.index_data += 1
                continue

            if args.run_type in ['collect', 'train'] and args.collect_type in ['dagger', 'SF']:
                
                _key = '{}_{}'.format(new_trajectory['seq_name'], data_it)
                if _key in self.collected_keys:
                    self.index_data += 1
                    continue
                else:
                    batch.append(new_trajectory)
                    self.index_data += 1
            else:
                batch.append(new_trajectory)
                self.index_data += 1

            if len(batch) == self.batch_size:
                break 

        self.batch = copy.deepcopy(batch)
        assert len(self.batch) == self.batch_size, 'next_minibatch error'
        self.VectorEnvUtil.set_batch(self.batch)
        return self.batch
        # return [b['trajectory_dir'] for b in self.batch]
    #
    def changeToNewTrajectorys(self):
        self._changeEnv(need_change=False)

        self._setTrajectorys()
        
        self._setObjects()

        self.update_measurements()

    def _setObjects(self, ):
        objects_info = [item['object'] for item in self.batch]
        return self.simulator_tool.setObjects(objects_info)
    
    def _changeEnv(self, need_change: bool = True):
        using_map_list = [item['map_name'] for item in self.batch]
        
        assert len(using_map_list) == self.batch_size, '错误'

        machines_info_template = copy.deepcopy(args.machines_info)
        total_max_scene_num = 0
        for item in machines_info_template:
            total_max_scene_num += item['MAX_SCENE_NUM']
        assert self.batch_size <= total_max_scene_num, 'error args param: batch_size'

        machines_info = []
        ix = 0
        for index, item in enumerate(machines_info_template):
            machines_info.append(item)
            delta = min(self.batch_size, item['MAX_SCENE_NUM'], len(using_map_list)-ix)
            machines_info[index]['open_scenes'] = using_map_list[ix : ix + delta]
            machines_info[index]['gpus'] = [args.gpu_id] * 8
            ix += delta

        cnt = 0
        for item in machines_info:
            cnt += len(item['open_scenes'])
        assert self.batch_size == cnt, 'error create machines_info'

        #
        if self.this_scene_used_cnt < self.one_scene_could_use_num and \
            len(set(using_map_list)) == 1 and len(set(self.last_using_map_list)) == 1 and \
            using_map_list[0] is not None and self.last_using_map_list[0] is not None and \
            using_map_list[0] == self.last_using_map_list[0] and \
            need_change == False:
            self.this_scene_used_cnt += 1
            logger.warning('no need to change env: {}'.format(using_map_list))
            # use the current environments
            return
        else:
            logger.warning('to change env: {}'.format(using_map_list))
 
        #
        while True:
            try:
                self.machines_info = copy.deepcopy(machines_info)
                print('machines_info:', self.machines_info)
                self.simulator_tool = AirVLNSimulatorClientTool(machines_info=self.machines_info)
                self.simulator_tool.run_call()
                break
            except Exception as e:
                logger.error("启动场景失败 {}".format(e))
                time.sleep(3)
            except:
                logger.error('启动场景失败')
                time.sleep(3)

        self.last_using_map_list = using_map_list.copy()
        self.this_scene_used_cnt = 1

    def _setTrajectorys(self):
        start_position_list = [item['trajectory'][0]['position'] for item in self.batch]
        start_rotation_list = [item['trajectory'][0]['orientation'] for item in self.batch]

        # setpose
        poses = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            poses.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                pose = airsim.Pose(
                    position_val=airsim.Vector3r(
                        x_val=start_position_list[cnt][0],
                        y_val=start_position_list[cnt][1],
                        z_val=start_position_list[cnt][2],
                    ),
                    orientation_val=airsim.Quaternionr(
                        x_val=start_rotation_list[cnt][0],
                        y_val=start_rotation_list[cnt][1],
                        z_val=start_rotation_list[cnt][2],
                        w_val=start_rotation_list[cnt][3],
                    ),
                )
                poses[index_1].append(pose)
                cnt += 1

        results = self.simulator_tool.setPoses(poses=poses)
        results = self.simulator_tool.setPoses(poses=poses)
        results = self.simulator_tool.setPoses(poses=poses)
        state_info_results = self.simulator_tool.getSensorInfo()
        
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            for index_2, _ in enumerate(item['open_scenes']):
                pose = airsim.Pose(
                    position_val=airsim.Vector3r(
                        x_val=start_position_list[cnt][0],
                        y_val=start_position_list[cnt][1],
                        z_val=start_position_list[cnt][2],
                    ),
                    orientation_val=airsim.Quaternionr(
                        x_val=start_rotation_list[cnt][0],
                        y_val=start_rotation_list[cnt][1],
                        z_val=start_rotation_list[cnt][2],
                        w_val=start_rotation_list[cnt][3],
                    ),
                )
                self.sim_states[cnt] = SimState(index=cnt, step=0, raw_trajectory_info=self.batch[cnt])
                self.sim_states[cnt].trajectory = [state_info_results[index_1][index_2]]
                cnt += 1


    def get_obs(self):
        obs_states = self._getStates()  # get from AirSim
        obs, states = self.VectorEnvUtil.get_obs(obs_states)  # process to get final obs and states
        self.sim_states = states  # update sim_states

        return obs

    def _getStates(self):

        responses = self.simulator_tool.getImageResponses()  # 5 rgb + 5 depth for inference
        responses_for_record = self.simulator_tool.getImageResponsesForRecord()  # 2 rgb + 2 depth for record

        cnt = 0
        for item in responses:
            cnt += len(item)
        assert len(responses) == len(self.machines_info), 'error'
        assert cnt == self.batch_size, 'error'

        states = [None for _ in range(self.batch_size)]
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            for index_2 in range(len(item['open_scenes'])):
                rgb_images = responses[index_1][index_2][0]
                depth_images = responses[index_1][index_2][1]
                rgb_records = responses_for_record[index_1][index_2][0]
                depth_records = responses_for_record[index_1][index_2][1]
                state = self.sim_states[cnt]  # get current simstate
                states[cnt] = (rgb_images, depth_images, state, rgb_records, depth_records)  # pack all info together
                cnt += 1

        return states
    
    def _get_current_state(self) -> list:
        states = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            states.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                s = self.sim_states[cnt].state
                state = airsim.KinematicsState()
                state.position = airsim.Vector3r(*s['position'])
                state.orientation = airsim.Quaternionr(*s['orientation'])
                state.linear_velocity = airsim.Vector3r(*s['linear_velocity'])
                state.angular_velocity = airsim.Vector3r(*s['angular_velocity'])
                states[index_1].append(state)
                cnt += 1
        return states

    def _get_current_pose(self) -> list:
        poses = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            poses.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                poses[index_1].append(
                    self.sim_states[cnt].pose
                )
                cnt += 1
        return poses

    def reset(self):
        self.changeToNewTrajectorys()
        return self.get_obs()

    def revert2frame(self, index):
        self.sim_states[index].revert2frames()
        
    def makeActions(self, actions_list):
        actions_args = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            actions_args.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                actions_args[index_1].append(actions_list[cnt])
                cnt += 1

        start_states = self._get_current_state()

        results = self.simulator_tool.move_path_by_actions(actions_list=actions_args, start_states=start_states)  # control AirSim to move

        batch_results = []
        batch_iscollision = []

        for index_1, item in enumerate(self.machines_info):
            for index_2, _ in enumerate(item['open_scenes']):
                batch_results.append(results[index_1][index_2]['states'])
                batch_iscollision.append(results[index_1][index_2]['collision'])
        # When the server returns less than 5 points (collision or environment blockage), fill it to a length of 5
        for batch_idx, batch_result in enumerate(batch_results):
            
            if 0 < len(batch_result) < 5:
                batch_result.extend([copy.deepcopy(batch_result[-1]) for i in range(5 - len(batch_result))])
                batch_iscollision[batch_idx] = True
            elif len(batch_result) == 0:
                batch_result.extend([copy.deepcopy(self.sim_states[batch_idx].trajectory[-1]) for i in range(5)])
                batch_iscollision[batch_idx] = True

        for index, action_dict in enumerate(actions_list):
            final_pos = batch_results[index][-1]['sensors']['state']['position']
            if np.linalg.norm(np.array(final_pos) - np.array(self.batch[index]['object_position'])) < self.sim_states[index].SUCCESS_DISTANCE:
                self.sim_states[index].oracle_success = True
            elif self.sim_states[index].step >= int(args.maxWaypoints):
                self.sim_states[index].is_end = True
            
            self.sim_states[index].step += 1
            self.sim_states[index].trajectory.extend(batch_results[index])  # [xyzxyzw]
            self.sim_states[index].pre_action = action_dict
            self.sim_states[index].is_collisioned = batch_iscollision[index]
        
        self.update_measurements()
        return batch_results

    def update_measurements(self):
        self._update_distance_to_target()
        
    def _update_distance_to_target(self):
        target_positions = [item['object_position'] for item in self.batch]
        for idx, target_position in enumerate(target_positions):
            current_position = self.sim_states[idx].pose[0:3]
            distance = np.linalg.norm(np.array(current_position) - np.array(target_position))
            self.sim_states[idx].distance_to_target = distance
            # print(f'batch[{idx}/{len(self.batch)}]| distance: {round(distance, 2)}, position: {current_position[0]}, {current_position[1]}, {current_position[2]}, target: {target_position[0]}, {target_position[1]}, {target_position[2]}')
            print(f'batch[{idx}/{len(self.batch)}]| distance: {distance:.2f}m | position: <{current_position[0]:.2f}, {current_position[1]:.2f}, {current_position[2]:.2f}> | target: <{target_position[0]:.2f}, {target_position[1]:.2f}, {target_position[2]:.2f}>')
