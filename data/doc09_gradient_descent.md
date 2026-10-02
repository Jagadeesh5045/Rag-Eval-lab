# Gradient Descent and Optimizers

Gradient descent trains neural networks by iteratively moving parameters in the direction that reduces the loss, with the step size controlled by the learning rate. A learning rate that is too large causes divergence, while one that is too small makes training painfully slow. Learning rate schedules such as warmup followed by cosine decay are standard practice for large models.

Stochastic gradient descent (SGD) estimates the gradient on small mini-batches, which adds noise that can help escape poor local minima. Adam maintains per-parameter adaptive learning rates using running averages of the gradient and its square, which makes it robust to the choice of initial learning rate and the default choice for Transformer training. AdamW decouples weight decay from the adaptive update and is preferred for regularized training.

Batch size interacts with optimization: larger batches give cleaner gradient estimates and better hardware utilisation but may generalize slightly worse. Gradient clipping caps the gradient norm to prevent explosive updates in recurrent and very deep networks.
