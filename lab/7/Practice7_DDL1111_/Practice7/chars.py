import functools
import random
# now refactor things into a *Problem* abstraction
# you can directly reuse what you have implemented above
from abc import ABC, abstractmethod

import numpy as np

# setting up the problem
from utils import plot_evolution


def init_population(pop_size, gene_pool, state_length):
    """
    Randomly initialize a population for genetic algorithm
        pop_size  :  Number of individuals in population
        gene_pool   :  List of possible values for individuals
        state_length:  The length of each individual
    """
    population = []
    for _ in range(pop_size):
        new_individual = "".join(random.choices(gene_pool, k=state_length))
        population.append(new_individual)

    return population


def select(r, population, fitness_fn):
    """
    TODO: select *r* samples from *population*
    the simplest choice is to sample from *population* with each individual weighted by its fitness
    """

    # fitnesses = []
    # for p in population:
    #     fitnesses.append(fitness_fn(p))
    # evaluations = [0]
    # for f in fitnesses:
    #     evaluations.append(evaluations[-1] + f)
    # newp = []
    # maximum = evaluations[-1]
    # for i in range(r):
    #     temp = random.randrange(0, maximum, 1)
    #     for j in range(len(evaluations)):
    #         if temp >= evaluations[j]:
    #             newp.append(population[j])
    #             break
    def compare(a, b):
        if fitness_fn(a) > fitness_fn(b):
            return -1
        elif fitness_fn(a) == fitness_fn(b):
            return 0
        else:
            return 1

    population.sort(key=functools.cmp_to_key(compare))


    return population[0:r]


def crossover(x, y):
    """
    TODO: combine two parents to produce an offspring
    """
    n = len(x)
    c = random.randrange(0, n)
    x1 = x[:c] + y[c:n]
    x2 = y[:c] + x[c:n]
    return x1, x2


def mutate(x, gene_pool):
    """
    apply mutation to *x* by randomly replacing one of its gene from *gene_pool*
    """
    n = len(x)
    g = len(gene_pool)
    c = random.randrange(0, n)
    r = random.randrange(0, g)

    return x[:c] + gene_pool[r] + x[c+1: n]


class GAProblem(ABC):
    @abstractmethod
    def init_population(self, pop_size): pass

    @abstractmethod
    def fitness(self, sample): pass

    @abstractmethod
    def reproduce(self, population): pass

    @abstractmethod
    def replacement(self, old, new): pass


class PhraseGeneration(GAProblem):
    def __init__(self, target, alphabet):
        self.target = target
        self.alphabet = alphabet

    def init_population(self, pop_size):
        # raise NotImplementedError()
        return init_population(pop_size, self.alphabet, len(self.target))

    def fitness(self, sample):
        # TODO: evaluate how close *sample* is to the target
        same = 0
        for i in range(len(sample)):
            if sample[i] == self.target[i]:
                same = same + 1
        return same

    def reproduce(self, population, mutation_rate):
        """
        TODO: generate the next generation of population

        hint: make a new individual with

        mutate(recombine(*select(2, population, fitness_fn)), gene_pool, pmut)

        """
        evaluations = []
        population.sort(key=self.fitness, reverse=True)
        for x in population:
            evaluations.append(self.fitness(x))
        p = select(12, population, fitness_fn=self.fitness)
        # for x in p:
        #     print(x)
        #     print(self.fitness(x))
        for i in range(3):
            for j in range(i+1, 3):
                x, y = crossover(p[i], p[j])
                p.append(x)
                p.append(y)
        for i in range(len(population) - len(p)):
            p.append(mutate(population[i], alphabet))

        return p

    def replacement(self, old, new):
        """
        You can use your own strategy, for example retain some solutions from the old population
        """
        return new


def genetic_algorithm(
        problem: GAProblem,
        ngen, n_init_size, mutation_rate,
        log_intervel=100
):
    population = problem.init_population(n_init_size)
    best = max(population, key=problem.fitness)
    history = [(0, list(map(problem.fitness, population)))]

    for gen in range(ngen):
        next_gen = problem.reproduce(population, mutation_rate)
        population = problem.replacement(population, next_gen)

        if gen % log_intervel == 0:
            current_best = max(population, key=problem.fitness)
            if problem.fitness(current_best) > problem.fitness(best): best = current_best
            print(f"Generation: {gen}/{ngen},\tBest: {best},\tFitness={problem.fitness(best)}")
            history.append((gen, list(map(problem.fitness, population))))

    history.append((ngen - 1, list(map(problem.fitness, population))))
    return best, history


# now set up the parameters
ngen = 1200
max_population = 120
mutation_rate = 0.08

sid = 12011327  # TODO:  replace this with your own sid
target = f"Genetic Algorithm by {sid}"
u_case = [chr(x) for x in range(65, 91)]
l_case = [chr(x) for x in range(97, 123)]
gene_pool = u_case + l_case + [' ']  # all English chracters and white space
alphabet = gene_pool + [str(x) for x in range(10)]  # TODO: fix this: what is the search space now?

problem = PhraseGeneration(target, alphabet)

# and run it
solution, history = genetic_algorithm(problem, ngen, max_population, mutation_rate)

# visualize the evolution of the polulation
# bins = np.linspace(0, problem.max_fitness, problem.max_fitness + 1)
# plot_evolution(history, bins)
