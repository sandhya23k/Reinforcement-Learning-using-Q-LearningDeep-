# """
#     This code communicates with the coppeliaSim software and simulates shaking a container to mix objects of different color 

#     Install dependencies:
#     https://www.coppeliarobotics.com/helpFiles/en/zmqRemoteApiOverview.htm
    
#     MacOS: coppeliaSim.app/Contents/MacOS/coppeliaSim -GzmqRemoteApi.rpcPort=23004 ~/path/to/file/mix_Intro_to_AI.ttt
#     Ubuntu: ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23004 ~/path/to/file/mix_Intro_to_AI.ttt
# """

# import sys
# # Change to the path of your ZMQ python API
# sys.path.append('\zmqRemoteApi')
# import numpy as np
# from zmqRemoteApi import RemoteAPIClient
# import tensorflow as tf
# from tensorflow import keras
# import time


# class Simulation():
#     def _init_(self, sim_port = 23000):
#         self.sim_port = sim_port
#         self.directions = ['Up','Down','Left','Right']
#         self.initializeSim()

#     def initializeSim(self):
#         self.client = RemoteAPIClient('localhost',port=self.sim_port)
#         self.client.setStepping(True)
#         self.sim = self.client.getObject('sim')
        
#         # When simulation is not running, ZMQ message handling could be a bit
#         # slow, since the idle loop runs at 8 Hz by default. So let's make
#         # sure that the idle loop runs at full speed for this program:
#         self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
#         self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)  
        
#         self.getObjectHandles()
#         self.sim.startSimulation()
#         self.dropObjects()
#         self.getObjectsInBoxHandles()
    
#     def getObjectHandles(self):
#         self.tableHandle=self.sim.getObject('/Table')
#         self.boxHandle=self.sim.getObject('/Table/Box')
    
#     def dropObjects(self):
#         self.blocks = 18
#         frictionCube=0.06
#         frictionCup=0.8
#         blockLength=0.016
#         massOfBlock=14.375e-03
        
#         self.scriptHandle = self.sim.getScript(self.sim.scripttype_childscript,self.tableHandle)
#         self.client.step()
#         retInts,retFloats,retStrings=self.sim.callScriptFunction('setNumberOfBlocks',self.scriptHandle,[self.blocks],[massOfBlock,blockLength,frictionCube,frictionCup],['cylinder'])
        
#         print('Wait until blocks finish dropping')
#         while True:
#             self.client.step()
#             signalValue=self.sim.getFloatSignal('toPython')
#             if signalValue == 99:
#                 loop = 20
#                 while loop > 0:
#                     self.client.step()
#                     loop -= 1
#                 break
    
#     def getObjectsInBoxHandles(self):
#         self.object_shapes_handles=[]
#         self.obj_type = "Cylinder"
#         for obj_idx in range(self.blocks):
#             obj_handle = self.sim.getObjectHandle(f'{self.obj_type}{obj_idx}')
#             self.object_shapes_handles.append(obj_handle)

#     def getObjectsPositions(self):
#         pos_step = []
#         box_position = self.sim.getObjectPosition(self.boxHandle,self.sim.handle_world)
#         for obj_handle in self.object_shapes_handles:
#             # get the starting position of source
#             obj_position = self.sim.getObjectPosition(obj_handle,self.sim.handle_world)
#             obj_position = np.array(obj_position) - np.array(box_position)
#             pos_step.append(list(obj_position[:2]))
#         return pos_step
    
#     def action(self,direction=None):
#         if direction not in self.directions:
#             print(f'Direction: {direction} invalid, please choose one from {self.directions}')
#             return
#         box_position = self.sim.getObjectPosition(self.boxHandle,self.sim.handle_world)
#         _box_position = box_position
#         span = 0.02
#         steps = 5
#         if direction == 'Up':
#             idx = 1
#             dirs = [1, -1]
#         elif direction == 'Down':
#             idx = 1
#             dirs = [-1, 1]
#         elif direction == 'Right':
#             idx = 0
#             dirs = [1, -1]
#         elif direction == 'Left':
#             idx = 0
#             dirs = [-1, 1]

#         for _dir in dirs:
#             for _ in range(steps):
#                 _box_position[idx] += _dir*span / steps
#                 self.sim.setObjectPosition(self.boxHandle, self.sim.handle_world, _box_position)
#                 self.stepSim()

#     def stepSim(self):
#         self.client.step()

#     def stopSim(self):
#         self.sim.stopSimulation()

#     def initialize_model(self,state,action,lr):
#         # init = tf.keras.initializers.HeUniform()
#         model = tf.keras.Sequential()
#         model.add(keras.layers.Dense(24, input_shape=(state,), activation='relu'))
#         model.add(keras.layers.Dense(12, activation='relu'))
#         model.add(keras.layers.Dense(action, activation='linear'))
#         model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=['accuracy'])
#         return model
    
#     def choose_action(self,state,action_size,epsilon,model):
#         if np.random.rand() <= epsilon:
#             return np.random.choice(action_size)
#         action_values = model.predict(state)
#         return np.argmax(action_values[0])

#     def calculate_next_state(self,positions, states_len):
#         flattened_state = np.array(positions).flatten().reshape((1, states_len))
#         return flattened_state

#     def calculateReward(self, blueObjs, redObjs):
#         threshold = 0.6
#         done = False
#         reward = 0
#         if self.checkMeanDistance(blueObjs, redObjs) < threshold:
#             reward = 1
#         return reward, done

#     def checkMeanDistance(self, blueObjs, redObjs):
#         blueMean = np.sum(blueObjs, axis=0) / len(blueObjs) 
#         redMean = np.sum(redObjs, axis=0) / len(redObjs)
#         overallMean = (blueMean + redMean) / 2  
#         return np.linalg.norm(overallMean)

#     def update_model(self, model, batch_sample, gamma, td_errors):
#         for value in batch_sample:
#             self._perform_update(model, value, gamma, td_errors)

#     def train_model(self, model, replay_memory, batch_size, gamma, td_errors):
#         np.random.shuffle(replay_memory)
#         batch_sample = replay_memory[0:batch_size]
#         self.update_model(model, batch_sample, gamma, td_errors)

#     def _perform_update(self, model, value, gamma, td_errors):
#         current_state_value = model.predict(value["current_state"])
#         target_value = value["reward"]
#         next_state_value = model.predict(value["next_state"])
#         current_action = value["action"]
#         env.action(direction=env.directions[current_action])
#         td_error = np.abs(current_state_value[0][current_action] - target_value)
#         if not value["done"]:
#             target_value = target_value + gamma * np.max(next_state_value[0])
#         td_error = np.abs(current_state_value[0][current_action] - target_value)
#         print(f"td_error: {td_error}")
#         td_errors.append(td_error)
#         current_state_value[0][current_action] = target_value
#         model.fit(value["current_state"], current_state_value, verbose=0)


#     def train_episode(self, episodes, steps, epsilon, decay, learning_rate, states_cnt, batch_size, min_epsilon):
#         replay_memory = []
#         steps_cnt = 0
#         td_errors = []
#         main_model = self.initialize_model(states_cnt, len(self.directions), learning_rate)

#         with open("log_details_tderrors.txt", "w") as log_file:
#             for episode in range(episodes):
#                 print(f'Running episode: {episode + 1}')
#                 total_episode_reward = 0
#                 state = env.getObjectsPositions()
#                 current_state = np.array(state).flatten().reshape((1, states_cnt))
#                 replay_memory = self._run_episode(main_model, epsilon, decay, batch_size, min_epsilon,
#                                                   current_state, steps, steps_cnt, replay_memory,
#                                                   total_episode_reward, log_file, episode, td_errors, states_cnt)
#         self.test_model(main_model)

#     def _run_episode(self, main_model, epsilon, decay, batch_size, min_epsilon,
#                      current_state, steps, steps_cnt, replay_memory,
#                      total_episode_reward, log_file, episode, td_errors, states_cnt):
#         # Initialize td_errors list outside the loop
#         td_errors_episode = []

#         for _ in range(steps):
#             steps_cnt += 1
#             action_taken = env.choose_action(current_state, len(env.directions), epsilon, main_model)
#             direction = env.directions[action_taken]
#             env.action(direction=direction)
#             positions = env.getObjectsPositions()
#             blue_objs = positions[:9]
#             red_objs = positions[9:]
#             current_reward, done = env.calculateReward(blue_objs, red_objs)
#             total_episode_reward += current_reward
#             next_state = env.calculate_next_state(positions, states_cnt)
#             replay_memory.append({
#                 "current_state": current_state,
#                 "action": action_taken,
#                 "reward": current_reward,
#                 "next_state": next_state,
#                 "done": done
#             })
#             current_state = next_state
#             if len(replay_memory) > (states_cnt * 100):
#                 replay_memory.pop(0)
#             if done or steps_cnt >= steps:  # Add condition to check if steps limit is reached
#                 epsilon = epsilon * np.exp(-decay * episode)
#                 if steps_cnt % batch_size == 0:
#                     self.train_model(main_model, replay_memory, batch_size, min_epsilon, td_errors_episode)

#                 # Accumulate the td_errors across the entire episode
#                 td_errors.extend(td_errors_episode)

#                 break  # Exit the loop if done or steps limit reached

#         # Log TD Errors at the end of each episode
#         log_file.write(f"TD Errors Episode {episode + 1}: {td_errors_episode}\n")

#         log_file.write(f"Episode {episode + 1}\n")
#         log_file.write(f"Total Episode Reward: {total_episode_reward}\n")

#         return replay_memory

#     def test_model(self, model):
#         success_count = 0
#         trials = 10
#         total_time = 0.0
#         for _ in range(trials):
#             start_time = time.time()
#             success_count += self.test_single_trial(model)
#             total_time += time.time() - start_time

#         success_percentage = (success_count / trials) * 100

#         result_file_path = "results.txt"
#         with open(result_file_path, "w") as result_file:
#             result_file.write(f"Success Rate: {success_percentage}%\n")
#             result_file.write(f"Total time taken:{total_time}\n")

#     def test_single_trial(self, model):
#         state = env.getObjectsPositions()
#         current_state = np.array(state).flatten().reshape((1, 36))
#         for _ in range(10):
#             action_taken = model.predict(current_state)
#             action_taken = np.argmax(action_taken)
#             direction = env.directions[action_taken]
#             env.action(direction=direction)
#             positions = env.getObjectsPositions()
#             blue_objs = positions[:9]
#             red_objs = positions[9:]
#             _, done = env.calculateReward(blue_objs, red_objs)
#             next_state = env.calculate_next_state(positions, 36)
#             current_state = next_state
#             if done:
#                 return 1
#         return 0


# if __name__ == '__main__':
#     env = Simulation()
#     env.train_episode(episodes=5, steps=10, epsilon=0.6, decay=0.01, learning_rate=0.001, states_cnt=2 * 18,
#                       batch_size=16, min_epsilon=0.08)
#     env.stopSim()



"""
    This code communicates with the coppeliaSim software and simulates shaking a container to mix objects of different color 

    Install dependencies:
    https://www.coppeliarobotics.com/helpFiles/en/zmqRemoteApiOverview.htm
    
    MacOS: coppeliaSim.app/Contents/MacOS/coppeliaSim -GzmqRemoteApi.rpcPort=23004 ~/path/to/file/mix_Intro_to_AI.ttt
    Ubuntu: ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23004 ~/path/to/file/mix_Intro_to_AI.ttt
"""

import sys
# Change to the path of your ZMQ python API
sys.path.append('\zmqRemoteApi')
import numpy as np
from zmqRemoteApi import RemoteAPIClient
import tensorflow as tf
from tensorflow import keras
import time


class Simulation:
    def __init__(self, sim_port=23000):
        self.sim_port = sim_port
        self.directions = ['Up', 'Down', 'Left', 'Right']
        self.initializeSim()

    def initializeSim(self):
        self.client = RemoteAPIClient('localhost', port=self.sim_port)
        self.client.setStepping(True)
        self.sim = self.client.getObject('sim')

        # When simulation is not running, ZMQ message handling could be a bit
        # slow, since the idle loop runs at 8 Hz by default. So let's make
        # sure that the idle loop runs at full speed for this program:
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)

        self.getObjectHandles()
        self.sim.startSimulation()
        self.dropObjects()
        self.getObjectsInBoxHandles()

    def getObjectHandles(self):
        self.tableHandle = self.sim.getObject('/Table')
        self.boxHandle = self.sim.getObject('/Table/Box')

    def dropObjects(self):
        self.blocks = 18
        frictionCube = 0.06
        frictionCup = 0.8
        blockLength = 0.016
        massOfBlock = 14.375e-03

        self.scriptHandle = self.sim.getScript(self.sim.scripttype_childscript, self.tableHandle)
        self.client.step()
        retInts, retFloats, retStrings = self.sim.callScriptFunction(
            'setNumberOfBlocks', self.scriptHandle, [self.blocks], [massOfBlock, blockLength, frictionCube, frictionCup],
            ['cylinder'])

        print('Wait until blocks finish dropping')
        while True:
            self.client.step()
            signalValue = self.sim.getFloatSignal('toPython')
            if signalValue == 99:
                loop = 20
                while loop > 0:
                    self.client.step()
                    loop -= 1
                break

    def getObjectsInBoxHandles(self):
        self.object_shapes_handles = []
        self.obj_type = "Cylinder"
        for obj_idx in range(self.blocks):
            obj_handle = self.sim.getObjectHandle(f'{self.obj_type}{obj_idx}')
            self.object_shapes_handles.append(obj_handle)

    def getObjectsPositions(self):
        pos_step = []
        box_position = self.sim.getObjectPosition(self.boxHandle, self.sim.handle_world)
        for obj_handle in self.object_shapes_handles:
            # get the starting position of source
            obj_position = self.sim.getObjectPosition(obj_handle, self.sim.handle_world)
            obj_position = np.array(obj_position) - np.array(box_position)
            pos_step.append(list(obj_position[:2]))
        return pos_step

    def action(self, direction=None):
        if direction not in self.directions:
            print(f'Direction: {direction} invalid, please choose one from {self.directions}')
            return
        box_position = self.sim.getObjectPosition(self.boxHandle, self.sim.handle_world)
        _box_position = box_position
        span = 0.02
        steps = 5
        if direction == 'Up':
            idx = 1
            dirs = [1, -1]
        elif direction == 'Down':
            idx = 1
            dirs = [-1, 1]
        elif direction == 'Right':
            idx = 0
            dirs = [1, -1]
        elif direction == 'Left':
            idx = 0
            dirs = [-1, 1]

        for _dir in dirs:
            for _ in range(steps):
                _box_position[idx] += _dir * span / steps
                self.sim.setObjectPosition(self.boxHandle, self.sim.handle_world, _box_position)
                self.stepSim()

    def stepSim(self):
        self.client.step()

    def stopSim(self):
        self.sim.stopSimulation()

    def initialize_model(self, state, action, lr):
        # init = tf.keras.initializers.HeUniform()
        model = tf.keras.Sequential()
        model.add(keras.layers.Dense(24, input_shape=(state,), activation='relu'))
        model.add(keras.layers.Dense(12, activation='relu'))
        model.add(keras.layers.Dense(action, activation='linear'))
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=['accuracy'])
        return model

    def choose_action(self, state, action_size, epsilon, model):
        if np.random.rand() <= epsilon:
            return np.random.choice(action_size)
        action_values = model.predict(state)
        return np.argmax(action_values[0])

    def calculate_next_state(self, positions, states_len):
        flattened_state = np.array(positions).flatten().reshape((1, states_len))
        return flattened_state

    def calculateReward(self, blueObjs, redObjs):
        threshold = 0.6
        done = False
        reward = 0
        if self.checkMeanDistance(blueObjs, redObjs) < threshold:
            reward = 1
        return reward, done

    def checkMeanDistance(self, blueObjs, redObjs):
        blueMean = np.sum(blueObjs, axis=0) / len(blueObjs)
        redMean = np.sum(redObjs, axis=0) / len(redObjs)
        overallMean = (blueMean + redMean) / 2
        return np.linalg.norm(overallMean)

    def update_model(self, model, batch_sample, gamma, td_errors):
        print("updating model")
        for value in batch_sample:
            self._perform_update(model, value, gamma, td_errors)

    def train_model(self, model, replay_memory, batch_size, gamma, td_errors):
        print("training model")
        np.random.shuffle(replay_memory)
        batch_sample = replay_memory[0:batch_size]
        self.update_model(model, batch_sample, gamma, td_errors)

    def _perform_update(self, model, value, gamma, td_errors):
        print("Nagendra")
        current_state_value = model.predict(value["current_state"])
        target_value = value["reward"]
        next_state_value = model.predict(value["next_state"])
        current_action = value["action"]
        env.action(direction=env.directions[current_action])
        td_error = np.abs(current_state_value[0][current_action] - target_value)
        if not value["done"]:
            target_value = target_value + gamma * np.max(next_state_value[0])
        td_error = np.abs(current_state_value[0][current_action] - target_value)
        print(f"td_error: {td_error}")
        td_errors.append(td_error)
        current_state_value[0][current_action] = target_value
        model.fit(value["current_state"], current_state_value, verbose=0)

    def train_episode(self, episodes, steps, epsilon, decay, learning_rate, states_cnt, batch_size, min_epsilon):
        replay_memory = []
        steps_cnt = 0
        td_errors = []
        main_model = self.initialize_model(states_cnt, len(self.directions), learning_rate)

        with open("log_details_tderrors.txt", "w") as log_file:
            for episode in range(episodes):
                print(f'Running episode: {episode + 1}')
                total_episode_reward = 0
                state = env.getObjectsPositions()
                current_state = np.array(state).flatten().reshape((1, states_cnt))
                replay_memory = self._run_episode(main_model, epsilon, decay, batch_size, min_epsilon,
                                                  current_state, steps, steps_cnt, replay_memory,
                                                  total_episode_reward, log_file, episode, td_errors, states_cnt)
        self.test_model(main_model)

    def _run_episode(self, main_model, epsilon, decay, batch_size, min_epsilon,
                     current_state, steps, steps_cnt, replay_memory,
                     total_episode_reward, log_file, episode, td_errors, states_cnt):
        # Initialize td_errors list outside the loop
        td_errors_episode = []

        for _ in range(steps):
            steps_cnt += 1
            action_taken = env.choose_action(current_state, len(env.directions), epsilon, main_model)
            direction = env.directions[action_taken]
            env.action(direction=direction)
            positions = env.getObjectsPositions()
            blue_objs = positions[:9]
            red_objs = positions[9:]
            current_reward, done = env.calculateReward(blue_objs, red_objs)
            total_episode_reward += current_reward
            next_state = env.calculate_next_state(positions, states_cnt)
            replay_memory.append({
                "current_state": current_state,
                "action": action_taken,
                "reward": current_reward,
                "next_state": next_state,
                "done": done
            })
            current_state = next_state
            if len(replay_memory) > (states_cnt * 100):
                replay_memory.pop(0)
            if done or steps_cnt >= steps:  # Add condition to check if steps limit is reached
                epsilon = epsilon * np.exp(-decay * episode)
                if steps_cnt % batch_size == 0:
                    self.train_model(main_model, replay_memory, batch_size, min_epsilon, td_errors_episode)

                # Accumulate the td_errors across the entire episode
                td_errors.extend(td_errors_episode)

                # Print TD Errors during training
                print(f"TD Errors Episode {episode + 1}: {td_errors_episode}")

                break  # Exit the loop if done or steps limit reached

        # Log TD Errors at the end of each episode
        log_file.write(f"TD Errors Episode {episode + 1}: {td_errors_episode}\n")

        log_file.write(f"Episode {episode + 1}\n")
        log_file.write(f"Total Episode Reward: {total_episode_reward}\n")

        return replay_memory

    def test_model(self, model):
        success_count = 0
        trials = 10
        total_time = 0.0
        for _ in range(trials):
            start_time = time.time()
            success_count += self.test_single_trial(model)
            total_time += time.time() - start_time

        success_percentage = (success_count / trials) * 100

        result_file_path = "results.txt"
        with open(result_file_path, "w") as result_file:
            result_file.write(f"Success Rate: {success_percentage}%\n")
            result_file.write(f"Total time taken:{total_time}\n")

    def test_single_trial(self, model):
        state = env.getObjectsPositions()
        current_state = np.array(state).flatten().reshape((1, 36))
        for _ in range(10):
            action_taken = model.predict(current_state)
            action_taken = np.argmax(action_taken)
            direction = env.directions[action_taken]
            env.action(direction=direction)
            positions = env.getObjectsPositions()
            blue_objs = positions[:9]
            red_objs = positions[9:]
            _, done = env.calculateReward(blue_objs, red_objs)
            next_state = env.calculate_next_state(positions, 36)
            current_state = next_state
            if done:
                return 1
        return 0


if __name__ == '__main__':
    env = Simulation()
    env.train_episode(episodes=5, steps=10, epsilon=0.6, decay=0.01, learning_rate=0.001, states_cnt=2 * 18,
                      batch_size=16, min_epsilon=0.08)
    env.stopSim()
