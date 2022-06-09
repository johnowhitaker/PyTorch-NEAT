import neat, torch
from pytorch_neat.cppn import create_cppn
from matplotlib import pyplot as plt
import numpy as np
size=128
x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
xs = [t for t in torch.tensor(np.stack([x.flatten(), y.flatten()]))]
config_file = 'config_test_activation'
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      config_file)
p_new = neat.Population(config) # New, random population
keys = list(p_new.population.keys())
genome = p_new.population[keys[0]]
nodes = create_cppn(genome, config, ['x', 'y'], ['r','g','b']+list(range(len(genome.nodes)-3)))
[r_node, g_node, b_node] = nodes
out_im = torch.stack([r_node.activate(xs, xs[0].shape).reshape(size, size),
                      g_node.activate(xs, xs[0].shape).reshape(size, size),
                      b_node.activate(xs, xs[0].shape).reshape(size, size)])
plt.imshow(out_im.permute(1, 2, 0))
