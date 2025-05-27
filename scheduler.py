import math

class LRScheduler:
    """Simple exponential decay learning rate scheduler for all parameters."""
    
    def __init__(self, initial_lr, final_lr_factor=0.01):
        """
        Args:
            initial_lr: Starting learning rate
            final_lr_factor: Final LR as fraction of initial (e.g., 0.01 means final_lr = 0.01 * initial_lr)
        """
        self.initial_lr = initial_lr
        self.final_lr = initial_lr * final_lr_factor
    
    def get_lr(self, iteration, total_iterations):
        """Get learning rate for given iteration using exponential decay."""
        if total_iterations <= 1:
            return self.initial_lr
        
        # Exponential decay from initial_lr to final_lr
        progress = iteration / (total_iterations - 1)
        progress = min(progress, 1.0)  # Clamp to [0, 1]
        
        # Exponential interpolation: lr = initial * (final/initial)^progress
        lr_ratio = self.final_lr / self.initial_lr
        current_lr = self.initial_lr * (lr_ratio ** progress)
        
        return current_lr
