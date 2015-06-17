#!/usr/bin/python3
"""
Uses a genetic algorithm to determine the optimal number of layers,
neurons per each layer, learning rate and training iterations for an ANN given
a set of training data.
When running this script via a command line, it can take one optional argument
for the name of a file to stream output into in place of stdout.
"""

import logging
import simulate
import sys

# Evaluate command line arguments.
if len(sys.argv) > 1:
    try:
        output = open(sys.argv[1], 'w')
    except IOError:
        output = sys.stdout
        output.write("Error: can't open {} for writing")
        output.write("Output will be pushed to stdout")
    else:
        simulate.setOutput(output)
else:
    output = sys.stdout

logging.basicConfig(stream=output, level=logging.DEBUG)

try:
    simulate.simulate()
except:
    logging.exception("Got exception on main handler")
    raise
