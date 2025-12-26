from tqdm import tqdm
import heapq
from typing import List, Tuple, Dict

from data_loader import Job, JobLoader
from predictor import RuntimePredictor


class PredictedSRTF:
    """
    Predicted SRTF (preemptive, with AI):
    - Uses PREDICTED remaining time to make preemption decisions.
    - Advances time using TRUE runtime (ground truth) for simulation.
    """

    EPS = 1e-9

    def __init__(self, predictor: RuntimePredictor):
        self.predictor = predictor
        self.time = 0.0
        # heap key uses predicted remaining time
        self.ready_heap: List[Tuple[float, float, int, Job]] = []  # (pred_remain, arrive, seq, job)
        self.completed_jobs: List[Job] = []
        self._seq = 0

    def _push_ready(self, job: Job, pred_remain: float):
        self._seq += 1
        heapq.heappush(self.ready_heap, (pred_remain, float(job.arrive_time), self._seq, job))

    def schedule(self, jobs: List[Job], model_type: str):
        # reset
        self.time = 0.0
        self.ready_heap = []
        self.completed_jobs = []
        self._seq = 0

        jobs = sorted(jobs, key=lambda j: j.arrive_time)
        n = len(jobs)
        if n == 0:
            print("No jobs.")
            return

        # ---- true remaining time (for advancing time + completion) ----
        true_remain: Dict[int, float] = {job.job_id: float(job.true_runtime) for job in jobs}
        completion_time: Dict[int, float] = {}

        # ---- predicted remaining time (for decisions) ----
        # initialize predicted runtime for each job ONCE at arrival, then decrement as it runs
        pred_remain: Dict[int, float] = {}

        i = 0
        current: Job | None = None
        preemptions = 0

        pbar = tqdm(
            total=n,
            desc=f"Predicted-SRTF ({model_type})",
            ncols=90,
            mininterval=0.2,
            maxinterval=1.0,
            smoothing=0.0
        )

        # jump to first arrival
        self.time = float(jobs[0].arrive_time)

        def add_arrivals_at_time(t: float):
            """Add all jobs with arrive_time <= t. Predict runtime once when job arrives."""
            nonlocal i
            while i < n and float(jobs[i].arrive_time) <= t + self.EPS:
                job = jobs[i]
                i += 1

                # predict runtime ONCE at arrival
                pred = float(self.predictor.predict_runtime(job)[model_type])
                pred = max(self.EPS, pred)  # avoid 0
                job.predicted_runtime = pred

                pred_remain[job.job_id] = pred
                self._push_ready(job, pred_remain[job.job_id])

        add_arrivals_at_time(self.time)

        while i < n or self.ready_heap or current is not None:
            # dispatch if CPU idle
            if current is None:
                if self.ready_heap:
                    _, _, _, current = heapq.heappop(self.ready_heap)
                else:
                    # nothing ready -> jump to next arrival
                    next_t = float(jobs[i].arrive_time)
                    self.time = next_t
                    add_arrivals_at_time(self.time)
                    continue

            # complete if true remaining ~0
            if true_remain[current.job_id] <= self.EPS:
                completion_time[current.job_id] = self.time
                self.completed_jobs.append(current)
                pbar.update(1)
                current = None
                continue

            next_arrival = float(jobs[i].arrive_time) if i < n else float("inf")
            finish_time = self.time + true_remain[current.job_id]

            # next event time
            t_next = min(next_arrival, finish_time)

            # advance time
            delta = t_next - self.time
            if delta < 0:
                delta = 0.0
            self.time = t_next

            # decrement BOTH true remaining and predicted remaining for the running job
            true_remain[current.job_id] -= delta
            pred_remain[current.job_id] = max(0.0, pred_remain[current.job_id] - delta)

            # if arrival event, add arrivals and consider preemption using predicted remaining
            if i < n and abs(self.time - float(jobs[i].arrive_time)) <= self.EPS:
                add_arrivals_at_time(self.time)

                if self.ready_heap:
                    best_pred_rem, _, _, best_job = self.ready_heap[0]
                    cur_pred_rem = pred_remain[current.job_id]

                    # preempt if someone has smaller predicted remaining
                    if best_pred_rem + self.EPS < cur_pred_rem:
                        self._push_ready(current, cur_pred_rem)
                        heapq.heappop(self.ready_heap)
                        current = best_job
                        preemptions += 1

            # finish handled at top of loop

        pbar.close()
        print()

        # metrics
        waiting_times = []
        turnaround_times = []
        max_waiting_time = float("-inf")
        min_waiting_time = float("inf")

        for job in self.completed_jobs:
            ct = completion_time[job.job_id]
            tat = ct - float(job.arrive_time)
            wt = tat - float(job.true_runtime)
            wt = max(0.0, wt)  # clamp floating noise
            turnaround_times.append(tat)
            waiting_times.append(wt)
            max_waiting_time = max(max_waiting_time, wt)
            min_waiting_time = min(min_waiting_time, wt)

        avg_wait = sum(waiting_times) / len(waiting_times) if waiting_times else 0.0
        avg_tat = sum(turnaround_times) / len(turnaround_times) if turnaround_times else 0.0

        print(f"\n=== Predicted-SRTF Scheduling Metrics ({model_type}) ===")
        print("==================================================")
        print(f"avg_waiting_time         : {avg_wait:.2f}")
        print(f"avg_turnaround_time      : {avg_tat:.2f}")
        print(f"max_waiting_time         : {max_waiting_time:.2f}")
        print(f"min_waiting_time         : {min_waiting_time:.2f}")
        print(f"preemptions              : {preemptions}")
        print("==================================================\n")


if __name__ == "__main__":
    loader = JobLoader()
    test_jobs = loader.load_jobs_from_csv("testing_jobs.csv")
    loader.display_jobs_info()

    predictor = RuntimePredictor(model_types=["random_forest", "xgboost"])
    predictor.load_model("random_forest_model.pkl", "xgboost_model.pkl", "scaler.pkl")

    scheduler = PredictedSRTF(predictor)

    print("\n=== Using Random Forest Model (Predicted-SRTF) ===")
    scheduler.schedule(test_jobs, model_type="random_forest")

    print("\n=== Using XGBoost Model (Predicted-SRTF) ===")
    scheduler.schedule(test_jobs, model_type="xgboost")
