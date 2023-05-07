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
    return [random.uniform(-5,5), random.uniform(-5,5)]

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

def sigma(population):
    new_population = []
    for i in range(len(population)):
        parent = random.choice(population)
        child = [parent[0] + random.gauss(0, s1gma), parent[1] + random.gauss(0, s1gma)]
        new_population.append(child)

def mutation(population):
    for i in range(len(population)):
        candidate = population[i]
        mutation_candidate = [candidate[0] + random.gauss(0, mut), candidate[1] + random.gauss(0, mut)]
        population[i] = mutation_candidate



s1gma = 0.1
mut = 0.1