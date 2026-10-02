# Overfitting and Regularization

Overfitting happens when a model memorises training data instead of learning general patterns, producing low training error but high error on new data. It is most likely with high-capacity models, small datasets, or training run far past the point of best validation performance. The standard defence is a held-out validation set used to monitor generalization during training.

Regularization techniques penalise complexity. L2 regularization adds the squared weight magnitudes to the loss, shrinking weights toward zero. Dropout randomly deactivates a fraction of neurons on each training step, forcing the network to learn redundant representations that are robust to missing units; it is disabled at inference time. Early stopping halts training when validation loss stops improving, which is cheap and effective.

Data augmentation, which creates plausible variants of training examples, acts as another regularizer by expanding the effective dataset. In practice, combining several mild techniques beats any single aggressive one, and the validation curve is the ground truth for whether regularization is working.
