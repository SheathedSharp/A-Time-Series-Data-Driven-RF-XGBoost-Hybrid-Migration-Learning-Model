"""
Progress Display Utility Module
Provides elegant terminal progress display functionality with Rich and Halo library support
"""

import time
from typing import Optional, Any
from contextlib import contextmanager

try:
    from rich.console import Console
    from rich.status import Status
    from rich.progress import Progress, TaskID
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from halo import Halo
    HALO_AVAILABLE = True
except ImportError:
    HALO_AVAILABLE = False


class ProgressDisplay:
    """Elegant progress display utility class"""
    
    def __init__(self, use_rich: bool = True):
        """
        Initialize progress display
        
        Args:
            use_rich: Whether to prefer Rich library, fallback to Halo or plain print if unavailable
        """
        self.use_rich = use_rich and RICH_AVAILABLE
        self.use_halo = not self.use_rich and HALO_AVAILABLE
        self.console = Console() if self.use_rich else None
        
    @contextmanager
    def feature_selection_status(self):
        """
        Feature selection status manager
        
        Usage:
        with progress.feature_selection_status() as status:
            status.update("Loading data...")
            # Execute operations
            status.update("Training model...")
            # Execute operations
            status.complete("Feature selection completed, using 15 features")
        """
        if self.use_rich:
            manager = self._RichStatusManager(self.console)
        elif self.use_halo:
            manager = self._HaloStatusManager()
        else:
            manager = self._BasicStatusManager()
        
        # Manually call __enter__ and __exit__
        try:
            status = manager.__enter__()
            yield status
        finally:
            manager.__exit__(None, None, None)
    
    @contextmanager
    def model_training_progress(self, total_steps: int = 100):
        """
        Model training progress manager
        
        Args:
            total_steps: Total number of steps
            
        Usage:
        with progress.model_training_progress(100) as prog:
            for i in range(100):
                prog.update(1, f"Training step {i+1}")
                # Execute training
        """
        if self.use_rich:
            yield self._RichProgressManager(self.console, total_steps)
        else:
            yield self._BasicProgressManager(total_steps)
    
    def display_results_table(self, title: str, data: dict):
        """
        Display results table
        
        Args:
            title: Table title
            data: Data dictionary to display
        """
        if self.use_rich:
            self._display_rich_table(title, data)
        else:
            self._display_basic_table(title, data)
    
    def _display_rich_table(self, title: str, data: dict):
        """Display beautiful table using Rich"""
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        for key, value in data.items():
            table.add_row(str(key), str(value))
        
        panel = Panel(
            table,
            title=f"[bold blue]{title}[/bold blue]",
            border_style="green"
        )
        
        self.console.print(panel)
    
    def _display_basic_table(self, title: str, data: dict):
        """Display basic text table"""
        print(f"\n=== {title} ===")
        max_key_len = max(len(str(k)) for k in data.keys())
        for key, value in data.items():
            print(f"{str(key):<{max_key_len}} : {value}")
        print("=" * (max_key_len + 20))
    
    class _RichStatusManager:
        """Rich status manager"""
        
        def __init__(self, console: Console):
            self.console = console
            self.status = None
            self.fallback_mode = not console.is_terminal
            
        def __enter__(self):
            if self.fallback_mode:
                # If not in terminal environment, use simple print mode
                print("Processing...")
            else:
                try:
                    self.status = self.console.status("[bold green]Processing...", spinner="dots")
                    self.status.start()
                except Exception as e:
                    print(f"  [Error] Failed to create status: {e}")
                    self.fallback_mode = True
                    print("Processing...")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.status:
                self.status.stop()
        
        def update(self, message: str):
            """Update status message"""
            if self.fallback_mode:
                if self.console:
                    self.console.print(f"  - {message}", style="dim white")
                else:
                    print(f"  - {message}")
            elif self.status:
                # Also print each step in Rich terminal to ensure visibility
                self.console.print(f"  - {message}", style="dim white")
                self.status.update(f"[bold blue]{message}")
            else:
                if self.console:
                    self.console.print(f"  ! {message}", style="dim yellow")
                else:
                    print(f"  ! {message}")
        
        def complete(self, message: str):
            """Complete and display final message"""
            if self.status:
                self.status.stop()
            # Display completion message in all modes
            rprint(f"[bold green]+[/bold green] {message}")
    
    class _HaloStatusManager:
        """Halo status manager"""
        
        def __init__(self):
            self.spinner = None
            
        def __enter__(self):
            self.spinner = Halo(text='Processing random forest feature selection...', spinner='dots')
            self.spinner.start()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.spinner and self.spinner._spinner_id:
                if exc_type:
                    self.spinner.fail('Operation failed')
                else:
                    self.spinner.stop()
        
        def update(self, message: str):
            """Update status message"""
            if self.spinner:
                self.spinner.text = message
        
        def complete(self, message: str):
            """Complete and display final message"""
            if self.spinner:
                self.spinner.succeed(message)
    
    class _BasicStatusManager:
        """Basic status manager"""
        
        def __enter__(self):
            print("Processing random forest feature selection...")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        
        def update(self, message: str):
            """Update status message"""
            # Use ANSI color codes to set light gray
            gray_color = "\033[90m"  # Light gray
            reset_color = "\033[0m"  # Reset color
            print(f"{gray_color}  - {message}{reset_color}")
        
        def complete(self, message: str):
            """Complete and display final message"""
            green_color = "\033[92m"  # Green
            reset_color = "\033[0m"  # Reset color
            print(f"{green_color}+ {message}{reset_color}")
    
    class _RichProgressManager:
        """Rich progress manager"""
        
        def __init__(self, console: Console, total: int):
            self.console = console
            self.total = total
            self.progress = None
            self.task = None
            
        def __enter__(self):
            self.progress = Progress()
            self.progress.__enter__()
            self.task = self.progress.add_task("[green]Model training", total=self.total)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.progress:
                self.progress.__exit__(exc_type, exc_val, exc_tb)
        
        def update(self, advance: int = 1, description: str = None):
            """Update progress"""
            if self.progress and self.task is not None:
                self.progress.update(self.task, advance=advance)
                if description:
                    self.progress.update(self.task, description=f"[green]{description}")
    
    class _BasicProgressManager:
        """Basic progress manager"""
        
        def __init__(self, total: int):
            self.total = total
            self.current = 0
            
        def __enter__(self):
            print("Starting model training...")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            print("Model training completed")
        
        def update(self, advance: int = 1, description: str = None):
            """Update progress"""
            self.current += advance
            percent = (self.current / self.total) * 100
            if description:
                print(f"  {description} ({percent:.1f}%)")


# Convenience functions
def create_progress_display(use_rich: bool = True) -> ProgressDisplay:
    """
    Create progress display instance
    
    Args:
        use_rich: Whether to prefer Rich library
        
    Returns:
        ProgressDisplay instance
    """
    return ProgressDisplay(use_rich=use_rich)


# Global default instance
default_progress = create_progress_display()


# Convenience decorator
def with_progress(message: str = "Processing..."):
    """
    Progress display decorator
    
    Args:
        message: Message to display
        
    Usage:
    @with_progress("Training model...")
    def train_model():
        # Training logic
        pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with default_progress.feature_selection_status() as status:
                status.update(message)
                result = func(*args, **kwargs)
                status.complete(f"{message} completed")
                return result
        return wrapper
    return decorator 