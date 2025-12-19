from tensorboard.backend.event_processing import event_accumulator
import sys
import os

def main():
    log_dir = sys.argv[1]
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    # print(f"Available tags: {tags}")
    
    for tag in ['world_model/kl_loss', 'world_model/reconstruction_loss', 'world_model/total_loss']:
        if tag in tags:
            events = ea.Scalars(tag)
            if events:
                print(f"{tag}: {events[-1].value:.4f} (step {events[-1].step})")
        else:
            print(f"{tag} not found")

if __name__ == "__main__":
    main()
