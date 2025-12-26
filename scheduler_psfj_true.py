import heapq
from typing import List
from data_loader import Job, JobLoader

class PredictedSJF:
    def __init__(self):
        self.time = 0  # Current time
        self.queue = []  # Queue for pending jobs
        self.completed_jobs = []  # List of completed jobs

    # Schedule the jobs based on true runtime (instead of predicted)
    def schedule(self, jobs: List[Job]):
        waiting_times = []  # List to store waiting times
        turnaround_times = []  # List to store turnaround times
        max_waiting_time = float('-inf')  # Initialize to a very small number
        min_waiting_time = float('inf')  # Initialize to a very large number
        total_jobs = len(jobs)  # Total number of jobs

        # Sort jobs by their arrival time
        jobs = sorted(jobs, key=lambda job: job.arrive_time)

        # Process jobs
        while jobs or self.queue:
            # Add all jobs that have arrived by the current time to the queue
            while jobs and jobs[0].arrive_time <= self.time:
                job = jobs.pop(0)  # Pop the first job that has arrived
                job.predicted_runtime = job.true_runtime  # Set predicted runtime to true runtime
                heapq.heappush(self.queue, (job.predicted_runtime, job))  # Add job to queue based on true runtime

            # If there are jobs in the queue, schedule the one with the shortest true runtime
            if self.queue:
                _, job = heapq.heappop(self.queue)  # Get the job with the shortest true runtime

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
                if jobs:
                    self.time = jobs[0].arrive_time

        # Calculate average waiting time and turnaround time
        avg_wait_time = sum(waiting_times) / len(waiting_times) if waiting_times else 0
        avg_turnaround_time = sum(turnaround_times) / len(turnaround_times) if turnaround_times else 0


        # Print scheduling metrics
        print(f"\n=== Scheduling Metrics ===")
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
    
    scheduler = PredictedSJF()
    print("\n=== Using True Runtime Scheduling ===")
    scheduler.schedule(test_jobs) 
