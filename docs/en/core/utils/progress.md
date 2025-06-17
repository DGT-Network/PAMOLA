# Example 1: Simple Progress Tracking
from pamola.pamola_core.utils.progress import SimpleProgressBar
import time

def simple_progress_example():
    print("Example 1: Simple Progress Tracking")
    # Create a simple progress bar
    with SimpleProgressBar(total=100, description="Processing records", unit="records") as progress:
        for i in range(100):
            # Simulate work
            time.sleep(0.01)
            # Update progress
            progress.update(1)
    print("Simple progress example completed\n")


# Example 2: Hierarchical Progress Tracking
from pamola.pamola_core.utils.progress import HierarchicalProgressTracker

def hierarchical_progress_example():
    print("Example 2: Hierarchical Progress Tracking")
    # Create a parent progress tracker
    master = HierarchicalProgressTracker(
        total=3, 
        description="Master process",
        unit="stages",
        track_memory=True
    )

    # Process with subtasks
    for stage in range(3):
        # Create subtask
        subtask = master.create_subtask(
            total=100,
            description=f"Stage {stage+1} processing",
            unit="items"
        )
        
        # Process subtask
        for i in range(100):
            # Simulate work - different stages take different time
            time.sleep(0.005 * (stage + 1))
            subtask.update(1)
        
        # Subtask will automatically update master when completed
        # No need to call master.update()

    # Close master tracker
    master.close()
    print("Hierarchical progress example completed\n")


# Example 3: Processing Large DataFrames
import pandas as pd
import numpy as np
from pamola.pamola_core.utils.progress import process_dataframe_in_chunks_enhanced

def dataframe_processing_example():
    print("Example 3: Processing Large DataFrames")
    # Create a sample large DataFrame
    df = pd.DataFrame({'value': range(10000)})

    # Define a processing function
    def process_function(chunk):
        # Simulate processing with some computation
        time.sleep(0.02)
        return chunk['value'].sum()

    # Process the DataFrame in chunks
    results = process_dataframe_in_chunks_enhanced(
        df=df,
        process_func=process_function,
        description="Processing large DataFrame",
        chunk_size=1000,
        error_handling="log"  # Options: "fail", "ignore", "log"
    )

    # Results contains output from each chunk
    print(f"Processed {len(results)} chunks with sum: {sum(results)}")
    print("DataFrame processing example completed\n")


# Example 4: Multi-stage Operations
from pamola.pamola_core.utils.progress import multi_stage_process

def multi_stage_example():
    print("Example 4: Multi-stage Operations")
    # Define stages for a multi-step process
    stages = ["Data loading", "Preprocessing", "Feature engineering", "Model training"]

    # Create a multi-stage tracker with custom weights
    tracker = multi_stage_process(
        total_stages=len(stages),
        stage_descriptions=stages,
        stage_weights=[0.1, 0.2, 0.3, 0.4]  # Weights reflecting expected duration
    )

    # Use the tracker for each stage
    for i, stage in enumerate(stages):
        subtask = tracker.create_subtask(
            total=100,
            description=f"Stage {i+1}: {stage}",
            unit="steps"
        )
        
        # Perform stage processing with progress updates
        for step in range(100):
            # Simulate work - different stages take different time
            time.sleep(0.001 * (i + 1))
            subtask.update(1)
            
    # Close the tracker
    tracker.close()
    print("Multi-stage example completed\n")


# Example 5: Parallel Processing
from pamola.pamola_core.utils.progress import process_dataframe_in_parallel_enhanced

def parallel_processing_example():
    print("Example 5: Parallel Processing")
    # Create a sample DataFrame
    df = pd.DataFrame({'value': np.random.rand(100000)})

    # Define processing function
    def heavy_processing(chunk):
        # Simulate computationally intensive work
        time.sleep(0.1)  # Simulate heavy computation
        return np.percentile(chunk['value'], [25, 50, 75, 90, 95, 99])

    # Process in parallel with 4 workers (or fewer if less cores available)
    results = process_dataframe_in_parallel_enhanced(
        df=df,
        process_func=heavy_processing,
        description="Parallel percentile calculation",
        chunk_size=10000,
        n_jobs=4,  # Use 4 workers
        track_memory=True
    )

    # Results contains the percentiles for each chunk
    print(f"Processed {len(results)} chunks in parallel")
    print("First chunk percentiles:", results[0])
    print("Parallel processing example completed\n")


# Example 6: Error Handling
from pamola.pamola_core.utils.progress import track_operation_safely

def error_handling_example():
    print("Example 6: Error Handling")
    
    # Custom error handler
    def on_error_callback(exception):
        print(f"Custom error handler received: {type(exception).__name__}: {exception}")
        print("Performing cleanup or logging...")
    
    try:
        # Use a tracker with error handling
        with track_operation_safely(
            description="Operation with potential errors", 
            total=10, 
            unit="steps",
            on_error=on_error_callback
        ) as tracker:
            
            for i in range(10):
                # Simulate work
                time.sleep(0.1)
                
                # Inject an error in the middle
                if i == 5:
                    raise ValueError("Simulated error occurred")
                    
                tracker.update(1)
                
    except ValueError:
        print("Error was caught at higher level after callback")
        
    print("Error handling example completed\n")


# Run all examples
if __name__ == "__main__":
    # Uncomment the examples you want to run
    simple_progress_example()
    hierarchical_progress_example()
    dataframe_processing_example()
    multi_stage_example()
    parallel_processing_example()
    error_handling_example()