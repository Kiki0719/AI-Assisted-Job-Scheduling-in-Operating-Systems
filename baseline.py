import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from data_loader import JobLoader, Job


class Scheduler:
    def __init__(self):
        self.jobs = []
        self.results_df = pd.DataFrame()


    def load_jobs(self, jobs):
        """
        Receive jobs from JobLoader
        """
        self.jobs = jobs
        print(f"Loaded {len(self.jobs)} jobs into scheduler")
        return self.jobs

    def sort_jobs_by_arrival(self):
        self.jobs.sort(key=lambda x: x.arrive_time)
        print("Jobs sorted by arrival time")
        return self.jobs

    def display_first_n_jobs(self, n=5):
        print(f"\nFirst {n} jobs:")
        print("-" * 100)
        print(f"{'Job ID':<8} {'Arrive':<10} {'Model':<10} {'Batch':<10} "
              f"{'Dataset':<10} {'Epochs':<8} {'GPU':<5} {'Runtime':<10}")
        print("-" * 100)

        for job in self.jobs[:n]:
            print(f"{job.job_id:<8} {job.arrive_time:<10.2f} "
                  f"{job.model_size:<10.2f} {job.batch_size:<10.2f} "
                  f"{job.dataset_size:<10.2f} {job.epochs:<8} "
                  f"{job.uses_gpu:<5} {job.true_runtime:<10.2f}")


    def calculate_metrics(self, results_df):
        return {
            "avg_waiting_time": results_df["waiting_time"].mean(),
            "avg_turnaround_time": results_df["turnaround_time"].mean(),
            "max_waiting_time": results_df["waiting_time"].max(),
            "min_waiting_time": results_df["waiting_time"].min(),
        }

    def print_metrics(self, metrics, name):
        print(f"\n{name} Metrics")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"{k:<25}: {v:.2f}")
        print("=" * 50)

    # =========================
    # FCFS
    # =========================
    def fcfs_schedule(self):
        print("\n=== FCFS Scheduling ===")
        self.sort_jobs_by_arrival()

        current_time = 0
        results = []

        for job in self.jobs:
            job.start_time = max(current_time, job.arrive_time)
            job.finish_time = job.start_time + job.true_runtime

            waiting_time = job.start_time - job.arrive_time
            turnaround_time = job.finish_time - job.arrive_time

            current_time = job.finish_time

            results.append({
                "job_id": job.job_id,
                "arrive_time": job.arrive_time,
                "true_runtime": job.true_runtime,
                "start_time": job.start_time,
                "end_time": job.finish_time,
                "waiting_time": waiting_time,
                "turnaround_time": turnaround_time
            })

        df = pd.DataFrame(results)
        self.print_metrics(self.calculate_metrics(df), "FCFS")
        return df

    # =========================
    # Round Robin
    # =========================
    def rr_schedule(self, time_quantum=50):
        print(f"\n=== RR Scheduling (Q={time_quantum}) ===")
        self.sort_jobs_by_arrival()

        for job in self.jobs:
            job.remaining_time = job.true_runtime
            job.start_time = None

        current_time = 0
        queue = []
        completed = []
        results = []
        idx = 0

        while len(completed) < len(self.jobs):
            while idx < len(self.jobs) and self.jobs[idx].arrive_time <= current_time:
                queue.append(self.jobs[idx])
                idx += 1

            if not queue:
                current_time = self.jobs[idx].arrive_time
                continue

            job = queue.pop(0)

            if job.start_time is None:
                job.start_time = current_time

            exec_time = min(time_quantum, job.remaining_time)
            current_time += exec_time
            job.remaining_time -= exec_time

            while idx < len(self.jobs) and self.jobs[idx].arrive_time <= current_time:
                queue.append(self.jobs[idx])
                idx += 1

            if job.remaining_time > 0:
                queue.append(job)
            else:
                job.finish_time = current_time
                completed.append(job)

                results.append({
                    "job_id": job.job_id,
                    "arrive_time": job.arrive_time,
                    "true_runtime": job.true_runtime,
                    "start_time": job.start_time,
                    "end_time": job.finish_time,
                    "waiting_time": job.start_time - job.arrive_time,
                    "turnaround_time": job.finish_time - job.arrive_time
                })

        df = pd.DataFrame(results).sort_values("job_id").reset_index(drop=True)
        self.print_metrics(self.calculate_metrics(df), "RR")
        return df


def main():
    loader = JobLoader()
    scheduler = Scheduler()

    test_jobs = loader.load_jobs_from_csv("testing_jobs.csv")
    scheduler.load_jobs(test_jobs)

    scheduler.display_first_n_jobs()

    fcfs_results = scheduler.fcfs_schedule()

    # Reset job states
    scheduler.load_jobs(
        loader.load_jobs_from_csv("testing_jobs.csv")
    )

    rr_results = scheduler.rr_schedule(time_quantum=50)

    return fcfs_results, rr_results


if __name__ == "__main__":
    main()
