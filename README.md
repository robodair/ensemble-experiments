# ensemble-experiments
Investigating the effects of overtraining on simple Neural Network Ensembles

Experiment command only does experiment for a single error rate at a time - can run different error rate experiments at the same time

Experiment 1:
300 Data points
0.8 Learning Rate
20 Hidden nodes
3000 Validation points
2000 patience
ee ex2d experiment-1 -r {10,20,30,40}

Experiment 2:
120 Data points
0.2 Learning Rate
9 Hidden nodes
3000 Validation points
2000 patience
ee ex2d experiment-1 -d 120 -l 0.2 --hidden-nodes 9 -r {10,20,30,40}

Experiment 3:
300 Data points
0.2 Learning Rate
20 Hidden nodes
3000 Validation points
5000 patience
ee ex2d experiment-1 -l 0.2 -p 5000 -r {10,20,30,40}
