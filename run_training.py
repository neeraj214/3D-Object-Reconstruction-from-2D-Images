
import sys
from pathlib import Path
import json

# Add project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

from src.training.enhanced_trainer import EnhancedTrainer, create_training_config
from src.datasets.unified_multiview import UnifiedMultiviewDataset

def main():
    """
    Main function to configure and run the training pipeline.
    """
    # --- Create Base Configuration ---
    config = create_training_config()

    # --- Customize for Our Project ---
    # Model Config
    config['model_config']['representation'] = 'pointcloud'
    config['model_config']['num_points'] = 1024
    config['model_config']['encoder_type'] = 'resnet50'
    config['model_config']['attention_layers'] = []
    config['model_config']['decoder_depth'] = 8
    config['model_config']['decoder_hidden_dim'] = 512

    # Training Config
    config['batch_size'] = 4
    config['device'] = 'cpu'
    config['use_wandb'] = False # Disable wandb for now
    config['num_workers'] = 0

    # Optimizer and Scheduler
    config['optimizer_config']['lr'] = 1e-4
    config['scheduler_config']['warmup_epochs'] = 5

    # Loss Config
    config['loss_config']['chamfer_loss_weight'] = 0.6
    config['loss_config']['emd_loss_weight'] = 0.25
    config['loss_config']['emd_gp_weight'] = 0.01
    config['loss_config']['silhouette_loss_weight'] = 0.05
    config['loss_config']['normal_loss_weight'] = 0.1

    # --- Create Datasets ---
    train_dataset = UnifiedMultiviewDataset(
        annotations_path='data/pix3d/pix3d.json',
        split='train',
        num_points=config['model_config']['num_points'],
        augment=True,
        image_size=224
    )

    val_dataset = UnifiedMultiviewDataset(
        annotations_path='data/pix3d/pix3d.json',
        split='val',
        num_points=config['model_config']['num_points'],
        augment=False,
        image_size=224
    )

    # --- Stage A: Sanity Run ---
    print("--- Starting Stage A: Sanity Run (10 epochs) ---")
    stage_a_config = config.copy()
    stage_a_config['epochs'] = 10
    stage_a_config['checkpoint_dir'] = 'checkpoints/stage_a'
    trainer_a = EnhancedTrainer(stage_a_config)
    trainer_a.train(train_dataset, val_dataset)
    print("--- Stage A Complete ---")

    # --- Stage B: Full Run ---
    print("--- Starting Stage B: Full Run (30 epochs) ---")
    stage_b_config = config.copy()
    stage_b_config['epochs'] = 30
    stage_b_config['checkpoint_dir'] = 'checkpoints/stage_b'
    # Load from Stage A checkpoint
    stage_b_config['load_checkpoint'] = 'checkpoints/stage_a/final_model.pth'

    trainer_b = EnhancedTrainer(stage_b_config)
    trainer_b.train(train_dataset, val_dataset)
    print("--- Stage B Complete ---")

    # --- Stage C: Fine-Tune ---
    print("--- Starting Stage C: Fine-Tune (20 epochs) ---")
    stage_c_config = config.copy()
    stage_c_config['epochs'] = 20
    stage_c_config['optimizer_config']['lr'] = 1e-5 # Lower learning rate for fine-tuning
    stage_c_config['checkpoint_dir'] = 'checkpoints/stage_c'
    # Load from Stage B checkpoint
    stage_c_config['load_checkpoint'] = 'checkpoints/stage_b/final_model.pth'

    trainer_c = EnhancedTrainer(stage_c_config)
    trainer_c.train(train_dataset, val_dataset)
    print("--- Stage C Complete ---")


if __name__ == "__main__":
    main()
