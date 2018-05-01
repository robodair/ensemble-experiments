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
~1 Day train time if all error rates run in parallel

Experiment 2:
120 Data points
0.2 Learning Rate
9 Hidden nodes
3000 Validation points
2000 patience
ee ex2d experiment-2 -d 120 -l 0.2 --hidden-nodes 9 -r {10,20,30,40}
~1 Day train time if all error rates run in parallel

Experiment 3:
300 Data points
0.2 Learning Rate
20 Hidden nodes
3000 Validation points
5000 patience
ee ex2d experiment-3 -l 0.2 -p 5000 -r {10,20,30,40}
~1.5-2 Days train time if all error rates run in parallel

Experiment 4:
Control Experiment (No Bagging)
300 Data points
0.2 Learning Rate
20 Hidden nodes
3000 Validation points
5000 patience
ee ex2d experiment-4 --control -l 0.2 -p 5000 -r {10,20,30,40}
~1.5-2 Days train time if all error rates run in parallel
