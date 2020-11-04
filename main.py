import random

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import mlflow
import numpy as np
from KerasGA import GeneticAlgorithm

mlflow.set_tracking_uri('http://192.168.15.57:5000')
mlflow.set_experiment("PPT")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


class jogo:

    def __init__(self):
        self.pedra = 0
        self.papel = 1
        self.tesoura = 2        
        self.cpu_jogar()

    def cpu_jogar(self):
        self.cpu = random.choice([self.pedra, self.papel, self.tesoura])

    def jogar(self, jogada):
        self.pessoa = jogada
        if self.cpu == self.pessoa:
            return 0
        # casos CPU WIN
        elif (self.cpu == self.pedra and self.pessoa == self.tesoura) or (self.cpu == self.tesoura and self.pessoa == self.papel) or (self.cpu == self.papel and self.pessoa == self.pedra):
            return -1
        else:
            return 1

def transform(entrada):
    lista = [0,0,0]
    lista[entrada] = 1
    return lista


def build_keras_model():

    model = Sequential()

    # model.add(Dense(10, input_dim=3, activation='relu'))
    # model.add(Dense(3, activation='softmax'))

    model.add(Dense(13, input_dim=3, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    return model
    
def obtem_valores(i):   
    j = jogo()
    input_nn = transform(j.cpu)
    jogada =np.argmax(model.predict([input_nn]))
    return j.jogar(jogada)


def proxima_gen(population, scores):
    top_performers = GA.strongest_parents(population,scores)

    # Make pairs:
    # 'GA.pair' return a tuple of type: (chromosome, it's score)
    pairs = []
    while len(pairs) != GA.population_size:
        pairs.append( GA.pair(top_performers) )

    # Crossover:
    base_offsprings =  []
    for pair in pairs:
        offsprings = GA.crossover(pair[0][0], pair[1][0])
        # 'offsprings' contains two chromosomes
        base_offsprings.append(offsprings[-1])

    # Mutation:
    return GA.mutation(base_offsprings)



if __name__ == "__main__":
    import multiprocessing
    from tqdm import tqdm    
    
    model = build_keras_model()
    population_size =  100
    GA = GeneticAlgorithm(model, population_size = population_size, selection_rate = 0.05, mutation_rate = 0.1)

    population = GA.initial_population()

    scores_gen = []
    for geracao in range(20000):
        scores = []

        for indice, individuo in enumerate(tqdm(population)):
            
            model.set_weights(individuo)
            # starta os processos
            with multiprocessing.pool.ThreadPool(10) as p:
                acumulador = p.map(obtem_valores, [i for i in range(10)])
            scores.append(sum(acumulador))

        scores_gen.append(np.mean(scores))
        mlflow.log_metric('scores_geracao', np.mean(scores), geracao)
        population = proxima_gen(population, scores)