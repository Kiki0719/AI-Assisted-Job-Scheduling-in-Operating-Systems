
import heapq
from typing import List
from data_loader import Job, JobLoader
from predictor import RuntimePredictor 


class PredictedSJF:
    def __init__(self, predictor: RuntimePredictor):
        self.predictor = predictor  # The runtime predictor
        self.time = 0  # Current time
        self.queue = []  # Queue for pending jobs
        self.completed_jobs = []  # List of completed jobs

    # Schedule the jobs based on predicted runtime
    def schedule(self, jobs: List[Job], model_type: str):
        waiting_times = []  # List to store waiting times
        turnaround_times = []  # List to store turnaround times
        max_waiting_time = float('-inf')  # Initialize to a very small number
        min_waiting_time = float('inf')  # Initialize to a very large number
        total_jobs = len(jobs)  # Total number of jobs
        throughput = 0  # Throughput (number of jobs completed)

        # Sort jobs by their arrival time
        jobs = sorted(jobs, key=lambda job: job.arrive_time)

        # Process jobs
        while jobs or self.queue:
            # Add all jobs that have arrived by the current time to the queue
            while jobs and jobs[0].arrive_time <= self.time:
                job = jobs.pop(0)  # Pop the first job that has arrived
                predicted_runtime = self.predictor.predict_runtime(job)[model_type]  # Predict runtime using the selected model
                job.predicted_runtime = predicted_runtime  # Set the predicted runtime
                heapq.heappush(self.queue, (predicted_runtime, job))  # Add job to queue based on predicted runtime

            # If there are jobs in the queue, schedule the one with the shortest predicted runtime
            if self.queue:
                _, job = heapq.heappop(self.queue) 

                # Calculate waiting time = current time - job arrival time
                wait_time = self.time - job.arrive_time
                waiting_times.append(wait_time)

                # Calculate turnaround time = waiting time + actual runtime
                turnaround_time = wait_time + job.true_runtime
                turnaround_times.append(turnaround_time)

                # Update the current time by adding the job's true runtime
                self.time += job.true_runtime

                # Mark the job as completed
                self.completed_jobs.append(job)

                # Track the max and min waiting times
                max_waiting_time = max(max_waiting_time, wait_time)
                min_waiting_time = min(min_waiting_time, wait_time)


            else:
                # If no jobs are in the queue, jump to the next job arrival time
                if jobs:
                    self.time = jobs[0].arrive_time

        # Calculate average waiting time and turnaround time
        avg_wait_time = sum(waiting_times) / len(waiting_times) if waiting_times else 0
        avg_turnaround_time = sum(turnaround_times) / len(turnaround_times) if turnaround_times else 0


        # Print scheduling metrics
        print(f"\n=== SFJ- {model_type} Scheduling Metrics ===")
        print("==================================================")
        print(f"avg_waiting_time         : {avg_wait_time:.2f}")
        print(f"avg_turnaround_time      : {avg_turnaround_time:.2f}")
        print(f"max_waiting_time         : {max_waiting_time:.2f}")
        print(f"min_waiting_time         : {min_waiting_time:.2f}")
        print("==================================================")
        print()


if __name__ == "__main__":
    loader = JobLoader()
    test_jobs = loader.load_jobs_from_csv("testing_jobs.csv")  
    loader.display_jobs_info() 
    
    # Load trained runtime predictor (both random forest and XGBoost models)
    predictor = RuntimePredictor(model_types=["random_forest", "xgboost"])
    predictor.load_model("random_forest_model.pkl", "xgboost_model.pkl", "scaler.pkl")  # Load models and scaler
    
    # Create and run Predicted-SJF scheduler using random forest model
    scheduler_rf = PredictedSJF(predictor)
    print("\n=== Using Random Forest Model ===")
    scheduler_rf.schedule(test_jobs, model_type="random_forest") 
    
    # Create and run Predicted-SJF scheduler using XGBoost model
    scheduler_xgb = PredictedSJF(predictor)
    print("\n=== Using XGBoost Model ===")
    scheduler_xgb.schedule(test_jobs, model_type="xgboost")  
