import random
import math
import matplotlib.pyplot as plt
import numpy as np

def bukin6(x):
    return 100 * math.sqrt(abs(x[1] - 0.01 * x[0]**2)) + 0.01 * abs(x[0] + 10)

def matyas(x):
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

def schaffer2(x):
    return 0.5 + ((math.sin(x[0]**2 - x[1]**2)**2 - 0.5) / (1 + 0.001 * (x[0]**2 + x[1]**2))**2)

def create_ind():
    return [random.uniform(-5, 5), random.uniform(-5, 5)]

def create_population(pop_len):
    population = []
    for _ in range(pop_len):
        population.append(create_ind())
    return population

def fitness_func(population, func):
    fitness = []
    for ind in population:
        fitness.append(func(ind))
    return fitness

def sigma_func(parent, population, sigma, mu):
    for i in range(mu):
        parent = random.choice(population)
        child = [parent[0] + random.gauss(0, sigma), parent[1] + random.gauss(0, sigma)]
        population.append(child)

def mutation(population, mut):
    for i in range(len(population)):
        candidate = population[i]
        mutation_candidate = [candidate[0] + random.gauss(0, mut), candidate[1] + random.gauss(0, mut)]
        population[i] = mutation_candidate

def choose_best(population, fitness, lamda_):
    new_gen = []
    for i in range(lamda_):
        new_gen.append(population[fitness.index(fitness[i])])
        print(fitness[i])
    return new_gen

def genetic_alg(func, num_gen=5, mut_prob=0.1, sigma=0.1, lambda_=5, mu=10):
    population = create_population(lambda_)
    best_par = None
    best_fit = None
    for i in range(num_gen):
        fitness_values = fitness_func(population, func)
        fitness_values.sort()
        print(fitness_values)
        best = fitness_values[0]
        if best_fit == None or best < best_fit:
            best_fit = best
            best_par = population[fitness_values.index(best)]
        print(best)
        sigma_func(best_par, population, sigma, mu)
        mutation(population, mut_prob)
        fitness_values = fitness_func(population, func)
        fitness_values.sort()
        print(fitness_values)
        #print(i+1, " ", fitness_values[0], " ", sum(fitness_values)/len(fitness_values))
        population = choose_best(population, fitness_values, lambda_)

genetic_alg(matyas)
