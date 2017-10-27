import os

import keras.backend as K
from DenseNet import DenseNet
import numba as nb
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def policy_loss(actual_value, predicted_value, old_prediction):
    advantage = actual_value - predicted_value

    # Maybe some halfbaked normalization would be nice
    # something like advantage = advantage + 0.1 * advantage/(K.std(advantage) + 1e-10)

    # Fullbaked norm seems very unstable
    # advantage /= (K.std(advantage) + 1e-10)

    def loss(y_true, y_pred):
        prob = K.sum(y_pred * y_true, axis=-1)
        old_prob = K.sum(old_prediction * y_true, axis=-1)
        log_prob = K.log(prob + 1e-10)

        r = prob / (old_prob + 1e-10)

        entropy = K.sum(y_pred * K.log(y_pred + 1e-10), axis=-1)
        return -log_prob * K.mean(K.minimum(r * advantage, K.clip(r, min_value=0.8, max_value=1.2) * advantage)) + 0.01 * entropy
    return loss



class Agent:
    def __init__(self, training_epochs=10, from_save=False, gamma=.95, batch_size=126, max_size_prime=16):
        self.training_epochs = training_epochs
        self.max_size_prime = max_size_prime

        self.training_data = [[], [], [], []]
        self.batch_size = batch_size

        # Bunch of placeholders values
        self.dummy_value = np.zeros((1, 1))
        self.dummy_predictions = np.zeros((1, self.max_size_prime))

        self.actor_critic= self._build_actor_critic()

        self.gammas = np.array([gamma**(i+1) for i in range(self.max_size_prime)]).astype(np.float32)

        if from_save is True:
            self.actor_critic.load_weights('actor_critic' + str(self.max_size_prime))

    def _build_actor_critic(self):

        state_input = Input(shape=(self.max_size_prime, 1))

        # Used for loss function
        actual_value = Input(shape=(1,))
        predicted_value = Input(shape=(1,))
        old_predictions = Input(shape=(self.max_size_prime, ))

        main_network = DenseNet(state_input, 5, 4, 24, 36)

        actor_bit_switch = Dense(self.max_size_prime, activation='softmax')(main_network)
        critic_value = Dense(1)(main_network)


        actor_critic_discriminator = Model(inputs=[state_input, actual_value, predicted_value, old_predictions], outputs=[actor_bit_switch, critic_value])
        actor_critic_discriminator.compile(optimizer='adam',
                      loss=[policy_loss(actual_value=actual_value,
                                        predicted_value=predicted_value,
                                        old_prediction=old_predictions
                                        ),
                            'mse'
                            ])

        actor_critic_discriminator.summary()

        return actor_critic_discriminator


    def train(self, epoch):

        value_list, policy_losses, critic_losses = [], [], []
        e = 0
        while e <= epoch:
            done = False
            print('Epoch :', e)
            batch_num = 0
            while done == False:

                fake_batch, actions, predicted_values, old_predictions = self.get_fake_batch()
                values = get_values(fake_batch, self.max_size_prime, self.batch_size, self.gammas)
                fake_batch = fake_batch[:self.batch_size]
                value_list.append(np.mean(values))


                tmp_loss = np.zeros(shape=(10, 2))
                for i in range(self.training_epochs):
                    tmp_loss[i] = (self.actor_critic.train_on_batch([fake_batch, values, predicted_values, old_predictions], [actions, values]))[1:]
                policy_losses.append(np.mean(tmp_loss[:,0]))
                critic_losses.append(np.mean(tmp_loss[:,1]))



                if batch_num % 100 == 0:
                    print()
                    self.actor_critic.save_weights('actor_critic_' + str(self.max_size_prime))
                    print('Batch number :', batch_num, '\tEpoch :', e, '\tAverage values :', np.mean(value_list))
                    print('Policy losses :', '%.5f' % np.mean(policy_losses),
                          '\tCritic losses :', '%.5f' % np.mean(critic_losses))
                    self.print_pred()
                    value_list, policy_losses, critic_losses = [], [], []

                batch_num += 1
            e += 1

    @nb.jit
    def get_fake_batch(self):
        seed = np.random.random_integers(low=0, high=1, size=(1, self.max_size_prime, 1))

        fake_batch = np.zeros((self.batch_size + self.max_size_prime, self.max_size_prime, 1))
        predicted_values = np.zeros((self.batch_size + self.max_size_prime, 1))
        actions = np.zeros((self.batch_size + self.max_size_prime, self.max_size_prime))
        old_predictions = np.zeros((self.batch_size + self.max_size_prime, self.max_size_prime))

        for i in range(self.batch_size + self.max_size_prime):
            predictions = self.actor_critic.predict([seed, self.dummy_value, self.dummy_value, self.dummy_predictions])
            numba_optimised_pred_rollover(old_predictions, predictions, self.max_size_prime, i, seed, predicted_values,
                                          actions, fake_batch)

        return fake_batch, actions[:self.batch_size], predicted_values[:self.batch_size], old_predictions[
                                                                                          :self.batch_size]

    @nb.jit
    def make_seed(self):
        seed = np.random.random_integers(low=0, high=1, size=(1, self.max_size_prime, 1))

        for i in range(self.max_size_prime):
            predictions = self.actor_critic.predict([seed, self.dummy_value, self.dummy_value, self.dummy_predictions])
            choice = np.random.choice(self.max_size_prime, 1, p=predictions[0][0])
            seed[:, choice] = 1 - seed[:, choice]

        return seed


    @nb.jit
    def print_pred(self):
        fake_state = self.make_seed()

        pred = ""
        for i in range(self.max_size_prime):
            pred += str(fake_state[i, 0][0])[0]
        print(pred, int(pred, 2), divisors(int(pred, 2)))



@nb.jit(nb.float32[:,:](nb.float32[:,:,:], nb.int64, nb.int64, nb.float32[:]))
def get_values(fake_batch, max_size_prime, batch_size, gammas):
    values = np.zeros(shape=(fake_batch.shape[0], 1))
    for i in range(fake_batch.shape[0]):
        number = ''
        for j in range(fake_batch.shape[1]):
            number += str(fake_batch[i,j][0])[0]
        number = int(number, 2)
        values[i] = (number / divisors(number)) / 2**(max_size_prime - 1)
    return numba_optimised_nstep_value_function(values, batch_size, max_size_prime, gammas)


# Some strong numba optimisation in bottlenecks
# N_Step reward function
@nb.jit(nb.float32[:,:](nb.float64[:,:], nb.int64, nb.int64, nb.float32[:]))
def numba_optimised_nstep_value_function(values, batch_size, n_step, gammas):
    for i in range(batch_size):
        for j in range(n_step):
            values[i] += values[i + j + 1] * gammas[j]
    return values[:batch_size]


@nb.jit(nb.void(nb.float32[:,:], nb.float32[:,:], nb.int64, nb.int64, nb.float32[:,:,:], nb.float32[:,:], nb.float32[:,:], nb.float32[:,:,:]))
def numba_optimised_pred_rollover(old_predictions, predictions, max_prime_size, index, seed, predicted_values, actions, fake_batch):
    old_predictions[index] = predictions[0][0]

    choice = np.random.choice(max_prime_size, 1, p=predictions[0][0])
    predicted_values[index][0] = predictions[1][0]
    actions[index][choice] = 1
    seed[:, choice] = 1 - seed[:, choice]
    fake_batch[index] = seed


# Kinda realising now that il just be fitting on prime numbers in an indirect way,
# maybe some generalisation will be had anyway since im trying 2 get as little divisors in a very indirect way
@nb.jit(nb.int64(nb.int64))
def divisors(x):
    limit = x
    number_of_divisors = 0

    if x == 1:
        return 1

    i = 1
    while i < limit:

        if x % i == 0:
            limit = x // i
            if limit != i:
                number_of_divisors += 1
            number_of_divisors += 1
        i += 1
    return number_of_divisors



if __name__ == '__main__':
    with tf.device('cpu/:0'):
        agent = Agent(max_size_prime=32)
        agent.train(epoch=5000)