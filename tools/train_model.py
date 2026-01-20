import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Train 3D Reconstruction Model")
    parser.add_argument("--mode", type=str, choices=['fusion', 'pointnet', 'multiview'], default='fusion', help="Training mode")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()
    
    # Determine which script to run
    if args.mode == 'fusion':
        script = "src/training/train_fusion.py"
    elif args.mode == 'pointnet':
        script = "src/training/train_pointnet.py"
    elif args.mode == 'multiview':
        script = "src/training/train_multiview.py"
    else:
        print(f"Unknown mode: {args.mode}")
        return

    # Check if script exists
    if not os.path.exists(script):
        print(f"Error: Training script not found: {script}")
        # Fallback to enhanced_trainer if fusion/pointnet not found but enhanced is there
        if os.path.exists("src/training/enhanced_trainer.py"):
            print("Falling back to src/training/enhanced_trainer.py")
            script = "src/training/enhanced_trainer.py"
        else:
            return

    cmd = [sys.executable, script]
    if args.config:
        cmd.extend(["--config", args.config])
    
    print(f"Starting training with command: {' '.join(cmd)}")
    os.system(' '.join(cmd))

if __name__ == "__main__":
    main()
