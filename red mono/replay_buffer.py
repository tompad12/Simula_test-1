from collections import deque
import random
import numpy as np

capacity_buffer=50000
size_batch=32

class replay_buffer:
    def __init__(self,cap=capacity_buffer,size=size_batch):
        self.buffer=deque(maxlen=cap)
        self.capacity=cap
    def __len__(self):
        return len(self.buffer)
    def push(self, frames, state, action, reward, next_frames, next_state, done):
        self.buffer.append((frames,state,action,reward,next_frames,next_state,done))
    def sample(self,batch_size):
        batch=random.sample(self.buffer,batch_size)
        frames,states,actions,rewards,next_frames,next_states,dones=zip(*batch)
        return (
        np.stack(frames,axis=0).astype(np.uint8),
        np.stack(states,axis=0).astype(np.float32),
        np.array(actions,dtype=np.int64),
        np.array(rewards,dtype=np.float32),
        np.stack(next_frames,axis=0).astype(np.uint8),
        np.stack(next_states,axis=0).astype(np.float32),
        np.array(dones,dtype=np.float32))


