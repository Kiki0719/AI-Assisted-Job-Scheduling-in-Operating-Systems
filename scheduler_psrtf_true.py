from tqdm import tqdm
import heapq
from typing import List, Tuple, Dict
from data_loader import Job, JobLoader


class SRTFTrue:
    """
    True SRTF (preemptive) baseline:
    - Uses TRUE remaining time to make preemption decisions.
    - Advances time in an event-driven way (arrival / completion).
    """

    EPS = 1e-9  # slightly larger to avoid float traps

    def __init__(self):
        self.time = 0.0
        self.ready_heap: List[Tuple[float, float, int, Job]] = []  # (remain, arrive, seq, job)
        self.completed_jobs: List[Job] = []
        self._seq = 0

    def _push_ready(self, job: Job, remain: float):
        self._seq += 1
        heapq.heappush(self.ready_heap, (remain, float(job.arrive_time), self._seq, job))

    def schedule(self, jobs: List[Job]):
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

        # remaining time
        remain: Dict[int, float] = {job.job_id: float(job.true_runtime) for job in jobs}
        completion_time: Dict[int, float] = {}

        i = 0
        current: Job | None = None

        pbar = tqdm(
            total=n,
            desc="SRTF-TRUE Scheduling",
            ncols=80,
            mininterval=0.2,
            maxinterval=1.0,
            smoothing=0.0
        )
        preemptions = 0

        # jump to first arrival
        self.time = float(jobs[0].arrive_time)

        def add_arrivals_at_time(t: float):
            """Add all jobs with arrive_time <= t (with EPS tolerance)."""
            nonlocal i
            while i < n and float(jobs[i].arrive_time) <= t + self.EPS:
                job = jobs[i]
                i += 1
                job.predicted_runtime = float(job.true_runtime)  # symmetry only
                self._push_ready(job, remain[job.job_id])

        # start by adding first-time arrivals
        add_arrivals_at_time(self.time)

        while i < n or self.ready_heap or current is not None:
            # If no running job, dispatch next from ready; if none, jump to next arrival
            if current is None:
                if self.ready_heap:
                    _, _, _, current = heapq.heappop(self.ready_heap)
                else:
                    # nothing ready -> jump to next arrival
                    next_t = float(jobs[i].arrive_time)
                    self.time = next_t
                    add_arrivals_at_time(self.time)
                    continue

            # If current is (almost) finished, complete it
            if remain[current.job_id] <= self.EPS:
                completion_time[current.job_id] = self.time
                self.completed_jobs.append(current)
                pbar.update(1)
                current = None
                continue

            next_arrival = float(jobs[i].arrive_time) if i < n else float("inf")
            finish_time = self.time + remain[current.job_id]

            # Next event time: arrival or finish
            t_next = min(next_arrival, finish_time)

            # Advance time to next event
            delta = t_next - self.time
            if delta < 0:
                delta = 0.0
            self.time = t_next
            remain[current.job_id] -= delta

            # If we reached an arrival event, add arrivals and consider preemption
            if i < n and abs(self.time - float(jobs[i].arrive_time)) <= self.EPS:
                add_arrivals_at_time(self.time)

                # preempt if someone in ready has smaller remaining time
                if self.ready_heap:
                    best_remain, _, _, best_job = self.ready_heap[0]
                    cur_remain = remain[current.job_id]
                    if best_remain + self.EPS < cur_remain:
                        self._push_ready(current, cur_remain)
                        heapq.heappop(self.ready_heap)
                        current = best_job
                        preemptions += 1


            # If we reached finish_time, completion will be handled at top of loop

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
            turnaround_times.append(tat)
            waiting_times.append(wt)
            max_waiting_time = max(max_waiting_time, wt)
            min_waiting_time = min(min_waiting_time, wt)

        avg_wait = sum(waiting_times) / len(waiting_times) if waiting_times else 0.0
        avg_tat = sum(turnaround_times) / len(turnaround_times) if turnaround_times else 0.0

        print("\n=== SRTF-TRUE Scheduling Metrics (Baseline, No AI) ===")
        print("==================================================")
        print(f"avg_waiting_time         : {avg_wait:.2f}")
        print(f"avg_turnaround_time      : {avg_tat:.2f}")
        print(f"max_waiting_time         : {max_waiting_time:.2f}")
        print(f"min_waiting_time         : {min_waiting_time:.2f}")
        print(f"preemptions             : {preemptions}")
        print("==================================================\n")


if __name__ == "__main__":
    loader = JobLoader()
    test_jobs = loader.load_jobs_from_csv("testing_jobs.csv")
    loader.display_jobs_info()

    scheduler = SRTFTrue()
    print("\n=== Using True Runtime Scheduling (SRTF-TRUE Baseline) ===")
    scheduler.schedule(test_jobs)

    total_runtime = sum(j.true_runtime for j in test_jobs)
    print("Total true runtime:", total_runtime)
    print("Max true runtime:", max(j.true_runtime for j in test_jobs))
    print("Max arrive time:", max(j.arrive_time for j in test_jobs))
