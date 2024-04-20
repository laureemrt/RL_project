import numpy as np
import tensorflow as tf
from collections import deque
from sys import exit
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate


class ReplayBuffer:
  def __init__(self, obs_dim, act_dim, size):
    self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
    self.rews_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.float32)
    self.ptr, self.size, self.max_size = 0, 0, size

  def store(self, obs, act, rew, next_obs, done):
    self.obs1_buf[self.ptr] = obs
    self.obs2_buf[self.ptr] = next_obs
    self.acts_buf[self.ptr] = act
    self.rews_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done
    self.ptr = (self.ptr+1) % self.max_size
    self.size = min(self.size+1, self.max_size)

  def sample_batch(self, batch_size=32):
    idxs = np.random.randint(0, self.size, size=batch_size)
    return dict(s=self.obs1_buf[idxs],
                s2=self.obs2_buf[idxs],
                a=self.acts_buf[idxs],
                r=self.rews_buf[idxs],
                d=self.done_buf[idxs])


class BasicBuffer:
    
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)


def Critic_gen(state_size, action_size,hidden_layers):

    input_x = Input(shape=state_size)
    input_a = Input(shape=action_size)
    x = input_x
    for i,j in enumerate(hidden_layers[:-1]):
        if i==1:
            x = concatenate([x,input_a],axis=-1)
        x = Dense(j,activation='relu')(x)
    x = Dense(hidden_layers[-1])(x)
    
    return tf.keras.Model([input_x,input_a],x)

def Actor_gen(state_size, action_size, hidden_layers, action_mult=1):
    input_x = Input(shape=state_size)
    x = input_x
    for i in hidden_layers:
        x = Dense(i,activation='relu')(x)
    x = Dense(action_size,activation='tanh')(x)
    x = tf.math.multiply(x,action_mult)
    return tf.keras.Model(input_x,x)

class DDPGAgent:
    
    def __init__(self, env, gamma, tau, buffer_maxlen, critic_learning_rate, actor_learning_rate):
        
        self.env = env
        self.obs_dim = env.observation_space["observation"].shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_max = env.action_space.high[0]
        # self.action_max = 1
        
        # hyperparameters
        self.env = env
        self.gamma = gamma
        self.tau = tau
        
        # Main network outputs
        self.mu = Actor_gen((3),(1),[512,200,128],self.action_max)
        self.q_mu = Critic_gen((3),(1),[1024,512,300,1])

        # Target networks
        self.mu_target = Actor_gen((3),(1),[512,200,128],self.action_max)
        self.q_mu_target = Critic_gen((3),(1),[1024,512,300,1])
      
        # Copying weights in,
        self.mu_target.set_weights(self.mu.get_weights())
        self.q_mu_target.set_weights(self.q_mu.get_weights())
    
        # optimizers
        self.mu_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_learning_rate)
        self.q_mu_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_learning_rate)
  
        # self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.action_dim, size=buffer_maxlen)
        self.replay_buffer = BasicBuffer(buffer_maxlen)
        
        self.q_losses = []
        
        self.mu_losses = []
        
    def get_action(self, s, noise_scale):
        a =  self.mu.predict(s.reshape(1,-1))[0]
        a += noise_scale * np.random.randn(self.action_dim)
        return np.clip(a, -self.action_max, self.action_max)

    def update(self, batch_size):
        
        # batch = self.replay_buffer.sample_batch(batch_size)
        # X = batch['s']
        # X2 = batch['s2']
        # A = batch['a']
        # R = batch['r']
        # D = batch['d']
        X,A,R,X2,D = self.replay_buffer.sample(batch_size)
        X = np.asarray(X,dtype=np.float32)
        A = np.asarray(A,dtype=np.float32)
        R = np.asarray(R,dtype=np.float32)
        X2 = np.asarray(X2,dtype=np.float32)
        # print(R.shape)
        
        
        
        Xten=tf.convert_to_tensor(X)
        

        # Updating Ze Critic
        with tf.GradientTape() as tape:
          A2 =  self.mu_target(X2)
          # print(self.q_mu_target([X2,A2]))
          # exit(0)
          # print((self.gamma  * self.q_mu_target([X2,A2])).shape)
          # exit(0)
          q_target = R + self.gamma  * self.q_mu_target([X2,A2])
          # temp2 = np.concatenate((X,A),axis=1)
          qvals = self.q_mu([X,A]) 
          q_loss = tf.reduce_mean((qvals - q_target)**2)
          grads_q = tape.gradient(q_loss,self.q_mu.trainable_variables)
        self.q_mu_optimizer.apply_gradients(zip(grads_q, self.q_mu.trainable_variables))
        self.q_losses.append(q_loss)


        #Updating ZE Actor
        with tf.GradientTape() as tape2:
          A_mu =  self.mu(X)
          Q_mu = self.q_mu([X,A_mu])
          mu_loss =  -tf.reduce_mean(Q_mu)
          grads_mu = tape2.gradient(mu_loss,self.mu.trainable_variables)
          # print(grads_mu)
        self.mu_losses.append(mu_loss)
        self.mu_optimizer.apply_gradients(zip(grads_mu, self.mu.trainable_variables))



        # update target networks
              ## Updating both netwokrs
        # # updating q_mu network
        
        temp1 = np.array(self.q_mu_target.get_weights())
        temp2 = np.array(self.q_mu.get_weights())
        temp3 = self.tau*temp2 + (1-self.tau)*temp1
        self.q_mu_target.set_weights(temp3)
      

      # updating mu network
        temp1 = np.array(self.mu_target.get_weights())
        temp2 = np.array(self.mu.get_weights())
        temp3 = self.tau*temp2 + (1-self.tau)*temp1
        self.mu_target.set_weights(temp3)
        




def smooth(x):
  # last 100
  n = len(x)
  y = np.zeros(n)
  for i in range(n):
    start = max(0, i - 99)
    y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
  return y





def trainer2(env, agent, max_episodes, max_steps, batch_size, action_noise):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state, action_noise)
            next_state, reward, done, _ = env.step(action)
            d_store = False if step == max_steps-1 else done
            agent.replay_buffer.store(state, action, reward, next_state, d_store)
            episode_reward += reward

            if agent.replay_buffer.size > batch_size:
                # pass
                agent.update(batch_size)   


            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode_rewards
from sys import exit
def trainer(env, agent, max_episodes, max_steps, batch_size,action_noise):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state,action_noise)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward
            # print(len(agent.replay_buffer.buffer))
            # exit(0)
            if len(agent.replay_buffer.buffer) > batch_size:
                agent.update(batch_size)  
                # pass 

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode_rewards
