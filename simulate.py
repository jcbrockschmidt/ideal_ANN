import genetic
import sys

output = sys.stdout

def setOutput(out):
    output = out
    genetic.setOutput(output)

# Test data for a XOR gate
testData = (
    (0.1, 0.1, 0.9),
    (0.1, 0.9, 0.9),
    (0.9, 0.1, 0.9),
    (0.9, 0.9, 0.1)
)

def simulate():
    sim = genetic.Simulation(2, 1, testData, 100)
    sim.simulate(100)
