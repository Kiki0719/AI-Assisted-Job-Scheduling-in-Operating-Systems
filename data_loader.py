import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class Job:
    """Job data class representing a scheduling task"""
    job_id: int
    arrive_time: int
    model_size: float
    batch_size: float
    dataset_size: float
    epochs: int
    uses_gpu: int
    true_runtime: float
    # Additional attributes for scheduling
    predicted_runtime: float = 0.0
    start_time: int = -1
    finish_time: int = -1

class JobLoader:
    """Job loader responsible for loading and sorting jobs from CSV files"""
    
    def __init__(self):
        self.jobs = []
    
    def load_jobs_from_csv(self, file_path: str) -> List[Job]:
        """
        Load job data from CSV file
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of Job objects
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            print(f"Loaded file: {file_path}, total jobs: {len(df)}")
            
            # Convert DataFrame rows to Job objects
            self.jobs = []
            for _, row in df.iterrows():
                job = Job(
                    job_id=int(row['job_id']),
                    arrive_time=int(row['arrive_time']),
                    model_size=float(row['model_size']),
                    batch_size=float(row['batch_size']),
                    dataset_size=float(row['dataset_size']),
                    epochs=int(row['epochs']),
                    uses_gpu=int(row['uses_gpu']),
                    true_runtime=float(row['true_runtime'])
                )
                self.jobs.append(job)
            
            # Sort jobs by arrival time
            self.jobs.sort(key=lambda job: job.arrive_time)
            print(f"Jobs sorted by arrival time, earliest: {self.jobs[0].arrive_time}")
            
            return self.jobs
            
        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
            return []
        except Exception as e:
            print(f"Error loading file: {e}")
            return []
    
    def get_jobs(self) -> List[Job]:
        """Get the loaded job list"""
        return self.jobs
    
    def display_jobs_info(self, max_display: int = 5):
        """Display job information for debugging"""
        print(f"\nJob information (showing first {max_display}):")
        print("JobID | ArriveTime | ModelSize | BatchSize | DatasetSize | Epochs | GPU | TrueRuntime")
        print("-" * 80)
        
        for i, job in enumerate(self.jobs[:max_display]):
            print(f"{job.job_id:4} | {job.arrive_time:10} | {job.model_size:9.2f} | "
                  f"{job.batch_size:9.2f} | {job.dataset_size:11.2f} | {job.epochs:6} | "
                  f"{job.uses_gpu:3} | {job.true_runtime:12.2f}")

if __name__ == "__main__":
    loader = JobLoader()

    train_jobs = loader.load_jobs_from_csv("training_jobs.csv")
    loader.display_jobs_info()

    test_jobs = loader.load_jobs_from_csv("testing_jobs.csv")
    loader.display_jobs_info()