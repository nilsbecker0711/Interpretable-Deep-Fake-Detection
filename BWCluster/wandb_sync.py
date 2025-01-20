import wandb

# Initialize W&B run
wandb.init(project='Interpretable-Deep-Fake-Detection', entity='555kaireffert')

# Path to pre-existing TensorBoard logs
log_dir = '~/Interpretable-Deep-Fake-Detection/BWCluster/logs'

# Sync the TensorBoard logs with W&B
wandb.save(log_dir)

# Finish the W&B run
wandb.finish()
