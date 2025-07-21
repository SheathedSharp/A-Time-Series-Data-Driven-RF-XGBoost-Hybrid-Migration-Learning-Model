"""
Progress Display Utility Module
Provides elegant terminal progress display functionality with Rich and Halo library support
"""

import time
from typing import Optional, Any, Dict, List
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
    """Elegant progress display utility class for comprehensive project operations"""
    
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
    def operation_status(self, initial_message: str = "Processing..."):
        """
        Generic operation status manager for any long-running process
        
        Usage:
        with progress.operation_status("Loading data...") as status:
            status.update("Processing features...")
            # Execute operations
            status.update("Training model...")
            # Execute operations
            status.complete("Operation completed successfully")
        """
        if self.use_rich:
            manager = self._RichStatusManager(self.console, initial_message)
        elif self.use_halo:
            manager = self._HaloStatusManager(initial_message)
        else:
            manager = self._BasicStatusManager(initial_message)
        
        try:
            status = manager.__enter__()
            yield status
        finally:
            manager.__exit__(None, None, None)
    
    @contextmanager
    def feature_selection_status(self):
        """
        Feature selection status manager (legacy method for backward compatibility)
        
        Usage:
        with progress.feature_selection_status() as status:
            status.update("Loading data...")
            # Execute operations
            status.update("Training model...")
            # Execute operations
            status.complete("Feature selection completed, using 15 features")
        """
        with self.operation_status("Processing random forest feature selection...") as status:
            yield status
    
    @contextmanager
    def data_loading_status(self):
        """
        Data loading status manager
        
        Usage:
        with progress.data_loading_status() as status:
            status.update("Reading CSV file...")
            status.update("Processing temporal features...")
            status.complete("Data loaded successfully")
        """
        with self.operation_status("Loading and processing data...") as status:
            yield status
    
    @contextmanager
    def model_training_status(self):
        """
        Model training status manager
        
        Usage:
        with progress.model_training_status() as status:
            status.update("Initializing model...")
            status.update("Training on batch 1/10...")
            status.complete("Model training completed")
        """
        with self.operation_status("Training machine learning model...") as status:
            yield status
    
    @contextmanager
    def parameter_optimization_status(self):
        """
        Parameter optimization status manager
        
        Usage:
        with progress.parameter_optimization_status() as status:
            status.update("Iteration 1/10...")
            status.update("Evaluating parameters...")
            status.complete("Optimization completed")
        """
        with self.operation_status("Optimizing model parameters...") as status:
            yield status
    
    @contextmanager
    def temporal_processing_status(self):
        """
        Temporal feature processing status manager
        
        Usage:
        with progress.temporal_processing_status() as status:
            status.update("Calculating rolling statistics...")
            status.update("Generating lag features...")
            status.complete("Temporal features generated")
        """
        with self.operation_status("Processing temporal features...") as status:
            yield status
    
    @contextmanager
    def sampling_status(self):
        """
        Data sampling status manager
        
        Usage:
        with progress.sampling_status() as status:
            status.update("Analyzing fault patterns...")
            status.update("Balancing dataset...")
            status.complete("Dataset balanced successfully")
        """
        with self.operation_status("Balancing dataset with CBSS...") as status:
            yield status
    
    @contextmanager
    def model_evaluation_status(self):
        """
        Model evaluation status manager
        
        Usage:
        with progress.model_evaluation_status() as status:
            status.update("Calculating metrics...")
            status.update("Generating reports...")
            status.complete("Evaluation completed")
        """
        with self.operation_status("Evaluating model performance...") as status:
            yield status
    
    @contextmanager
    def progress_bar(self, total_steps: int = 100, description: str = "Processing"):
        """
        Progress bar manager for operations with known total steps
        
        Args:
            total_steps: Total number of steps
            description: Description of the operation
            
        Usage:
        with progress.progress_bar(100, "Training model") as prog:
            for i in range(100):
                prog.update(1, f"Training step {i+1}")
                # Execute training
        """
        if self.use_rich:
            yield self._RichProgressManager(self.console, total_steps, description)
        else:
            yield self._BasicProgressManager(total_steps, description)
    
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
    
    def display_data_summary(self, data, title: str = "Data Summary"):
        """
        Display data summary information
        
        Args:
            data: DataFrame to summarize
            title: Summary title
        """
        summary_data = {
            "Shape": str(data.shape),
            "Missing Values": str(data.isnull().sum().sum()),
        }
        
        if 'label' in data.columns:
            summary_data.update({
                "Positive Samples": str(data['label'].sum()),
                "Negative Samples": str(len(data) - data['label'].sum()),
                "Positive Ratio": f"{data['label'].mean():.4f}"
            })
        
        self.display_results_table(title, summary_data)
    
    def display_model_metrics(self, metrics: Dict[str, float], title: str = "Model Performance"):
        """
        Display model performance metrics
        
        Args:
            metrics: Dictionary of metric names and values
            title: Table title
        """
        formatted_metrics = {k: f"{v:.4f}" for k, v in metrics.items()}
        self.display_results_table(title, formatted_metrics)
    
    def display_error(self, message: str, details: str = None):
        """
        Display error message
        
        Args:
            message: Error message
            details: Additional error details
        """
        if self.use_rich:
            self.console.print(f"[bold red]Error:[/bold red] {message}")
            if details:
                self.console.print(f"[dim]{details}[/dim]")
        else:
            print(f"Error: {message}")
            if details:
                print(f"  Details: {details}")
    
    def display_warning(self, message: str):
        """
        Display warning message
        
        Args:
            message: Warning message
        """
        if self.use_rich:
            self.console.print(f"[bold yellow]Warning:[/bold yellow] {message}")
        else:
            print(f"Warning: {message}")
    
    def display_success(self, message: str):
        """
        Display success message
        
        Args:
            message: Success message
        """
        if self.use_rich:
            self.console.print(f"[bold green]Success:[/bold green] {message}")
        else:
            print(f"Success: {message}")
    
    def _display_rich_table(self, title: str, data: dict):
        """Display beautiful table using Rich"""
        table = Table(title=None, show_header=False)
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
        
        def __init__(self, console: Console, initial_message: str):
            self.console = console
            self.status = None
            self.fallback_mode = not console.is_terminal
            self.initial_message = initial_message
            
        def __enter__(self):
            if self.fallback_mode:
                print(self.initial_message)
            else:
                try:
                    self.status = self.console.status(f"[bold green]{self.initial_message}", spinner="dots")
                    self.status.start()
                except Exception as e:
                    print(f"  [Error] Failed to create status: {e}")
                    self.fallback_mode = True
                    print(self.initial_message)
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
            rprint(f"[bold green]+[/bold green] {message}")
    
    class _HaloStatusManager:
        """Halo status manager"""
        
        def __init__(self, initial_message: str):
            self.spinner = None
            self.initial_message = initial_message
            
        def __enter__(self):
            self.spinner = Halo(text=self.initial_message, spinner='dots')
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
        
        def __init__(self, initial_message: str):
            self.initial_message = initial_message
            
        def __enter__(self):
            print(self.initial_message)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        
        def update(self, message: str):
            """Update status message"""
            gray_color = "\033[90m"
            reset_color = "\033[0m"
            print(f"{gray_color}  - {message}{reset_color}")
        
        def complete(self, message: str):
            """Complete and display final message"""
            green_color = "\033[92m"
            reset_color = "\033[0m"
            print(f"{green_color}+ {message}{reset_color}")
    
    class _RichProgressManager:
        """Rich progress manager"""
        
        def __init__(self, console: Console, total: int, description: str):
            self.console = console
            self.total = total
            self.progress = None
            self.task = None
            self.description = description
            
        def __enter__(self):
            self.progress = Progress()
            self.progress.__enter__()
            self.task = self.progress.add_task(f"[green]{self.description}", total=self.total)
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
        
        def __init__(self, total: int, description: str):
            self.total = total
            self.current = 0
            self.description = description
            
        def __enter__(self):
            print(f"Starting {self.description}...")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            print(f"{self.description} completed")
        
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
            with default_progress.operation_status(message) as status:
                status.update(message)
                result = func(*args, **kwargs)
                status.complete(f"{message} completed")
                return result
        return wrapper
    return decorator 