import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import time
import sys, os

# Permutation representation
euroCities = []
with open("european_cities.csv", "r") as infile:
	for line in infile:
		euroCities.append(line.split(';'))
cityNames = np.array(list(enumerate(euroCities[0][:])))
euroCities = euroCities[1:][:]
for i in xrange(len(euroCities)):
	euroCities[i][-1] = euroCities[i][-1].replace("\r\n", "")
euroCities = np.array(euroCities, dtype=float)

seed = 123
np.random.seed(seed)

subset = 7
permute = lambda a: a[np.random.choice(len(a), size=subset, replace=False)]
temp = xrange(0,subset)
cities = euroCities[temp]
cities = cities[:, temp]
sliceLength = np.ceil(subset * 0.50)

class Method:
	def __init__(self):
		pass

	def pairwise(self, iterable):
		a, b = it.tee(iterable)
		next(b, None)
		return list(it.izip(a, b))

	def distance(self, perm): #the objective function
		inds = np.array(perm[:,0], dtype=int)
		pairs = np.array(self.pairwise(inds), dtype=int)
		dists = euroCities[pairs[:,0], pairs[:,1]]
		return sum(dists) + euroCities[inds[0], inds[-1]], perm #ties up the route

# Exhaustive Search
class ExhaustiveSearch(Method):
	def __call__(self):
		t1 = time.time()
		SearchSpace = np.array(list(it.permutations(cityNames[temp])))
		candidates = []

		for i,p in enumerate(SearchSpace):
			candidates.append(self.distance(p)[0])
		t2 = time.time()
		candidates = np.array(candidates)
		print min(candidates)
		print SearchSpace[np.argmin(candidates)]
		print "CPU time: ", t2-t1
#ES = ExhaustiveSearch()()
# Hill Climbing
class HillClimbing(Method):
	def __call__(self, printInfo=True):
		# Random permutation/route
		current = permute(cityNames[temp])
		# "Nearby" if differs by a transposition
		t1 = time.time()
		while True:
			locality = []
			for i in xrange(len(current)-1):
				for j in xrange(i+1, len(current)):
					inds = range(len(current))
					inds[i], inds[j] = inds[j], inds[i]
					neigh = current[inds]
					locality.append(self.distance(neigh))
			current = min(locality)[1]
			if min(locality)[0] >= self.distance(current)[0]:
				if printInfo==True:
					print "Found an optimum!"
				break
			if time.time() - t1 > 60:
				if printInfo==True:
					print "Takes too long!"
				break
		if printInfo==True:
			print "CPU time: ", time.time() - t1
			print self.distance(current)[0]
			print current
		self.current = current

	def performanceTest(self):
		runs = []
		for i in xrange(20):
			self(0)
			runs.append(self.distance(self.current)[0])
		print "Mean: ", np.mean(runs)
		print "Best: ", np.min(runs)
		print "Worst: ", np.max(runs)
		print "Stdev: ", np.std(runs)

#HC = HillClimbing()
#HC.performanceTest()

# Genetic Algorithm
class GeneticAlgorithm(Method):
	def __call__(self):
		# initializing a random population
		self.popSize = 20
		self.pop = [np.random.permutation(cityNames[temp]) for i in xrange(self.popSize)]
		self.fhistory = []
		bestFit = self.fitness(self.pop)
		self.fhistory.append(bestFit)
		print "Initial fitness", bestFit
		iteration = 0
		t1 = time.time()
		while True:
			parents = self.parentSelection()
			pairs = list(it.combinations(parents, 2))
			children = self.testRecombine(pairs)
			self.pop = self.mutate(children)
			bestFit = self.fitness(self.pop)
			self.fhistory.append(bestFit)
			iteration += 1
			if iteration == 10000:
				print "Max iteration reached"
				break
		results = min(bestFit)
		print "CPU time: ", time.time() - t1
		print results
		self.visualize()

	def visualize(self):
		histvals = [np.mean(h[0]) for h in self.fhistory]
		plt.plot(histvals)
		plt.show()

	def fitness(self, p):
		# Calculates fitness values for the population based on distance()
		fvalues = []
		for indiv in p:
			fvalues.append(self.distance(indiv))
		return fvalues

	def mutate(self, x): # trying out Inversion Mutation
		r = np.random.randint(1,sliceLength)
		for c in x:
			c[r:r+sliceLength] = c[r:r+sliceLength][::-1] # = np.flip(x[r.min():r.max()]).fliplr()
		return x

	def parentSelection(self):
		# I'm trying a scheme that balances exploration/exploitation
		parents = []
		# 1. Choose best half of population
		parents = sorted([self.distance(p) for p in self.pop])[0:int(self.popSize/2.)]
		parents = [parent[1] for parent in parents]
		# 2. Let parents die with some probability
		parents = list(it.compress(parents, np.random.random(len(parents)) < 0.5))
		# 3. Fill in with totally random individuals
		parents += [np.random.permutation(cityNames[temp]) for i in xrange(int(self.popSize/2.) - len(parents))]
		# 4. Guarantee at least 2 parents
		return parents

	def testRecombine(self, parentPairs):
		# Just a quick and simple placeholder for testing of algorithm
		choice = []
		for i in xrange(10):
			ind = np.random.randint(10)
			choice.append(parentPairs[ind])
		children = []
		for i in xrange(10):
			child = choice[i][0]
			child[int(len(choice[i][1])/2.):] = choice[i][1][int(len(choice[i][1])/2.):]
			children.append(child)
		return children

	def recombine(self, x, y): # PMX, Partially Mapped Crossover
		# step 1
		child = np.zeros(subset)
		r = np.random.randint(1,sliceLength)
		child[r:r+sliceLength] = x[r:r+sliceLength]
		tmp = np.zeros(subset, dtype=int); tmp[r:r+sliceLength]+=1
		# step 2
		xTemp = np.array(list(enumerate(x)))
		yTemp = permute(xTemp)
		yTempSliced = yTemp[(xTemp != yTemp)[:,0] * tmp]
		child[yTempSliced[:,0]] = yTemp[yTempSliced[:,0]]
		print x
		print yTemp[:,1]
		return child

GA = GeneticAlgorithm()()
