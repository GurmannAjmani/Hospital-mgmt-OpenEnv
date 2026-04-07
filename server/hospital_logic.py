import numpy as np

class Patient:
    def __init__(self, pid, severity):
        self.pid = pid
        self.severity = severity  # 1: Critical, 5: Low
        self.health = 100.0
        self.wait_time = 0
        self.status = "waiting"

class PatientManager:
    def __init__(self):
        self.max_icu = 5
        self.max_ward = 15
        self.icu_occupied = 0
        self.ward_occupied = 0
        self.patients = []
        self.all_patients = []
        self.deaths = 0
        self.collapse_threshold = 3
        self.decay_multiplier = 1.5

    def initialize(self, task_id):
        self.icu_occupied = 0
        self.ward_occupied = 0
        self.deaths = 0

        if task_id == "hard":
            # HARD — target avg: 15-20
            # 12 patients, sev 1-3, extremely tight beds, very fast decay
            self.max_icu = 3
            self.max_ward = 9
            self.collapse_threshold = 2
            self.decay_multiplier = 2.4
            count = 12
            self.patients = [Patient(i, np.random.randint(1, 4)) for i in range(count)]

        elif task_id == "medium":
            # MEDIUM — target avg: 35-40
            # 12 patients, sev 1-4, exact capacity, faster decay
            self.max_icu = 4
            self.max_ward = 8
            self.collapse_threshold = 2
            self.decay_multiplier = 1.8
            count = 12
            self.patients = [Patient(i, np.random.randint(1, 5)) for i in range(count)]

        else:  # easy
            # EASY — target avg: 70+
            # 10 patients, sev 2-5, plenty of beds, slightly faster decay
            self.max_icu = 10
            self.max_ward = 20
            self.collapse_threshold = 4
            self.decay_multiplier = 1.0
            count = 10
            self.patients = [Patient(i, np.random.randint(2, 6)) for i in range(count)]

        self.all_patients = list(self.patients)

    def update_health(self):
        for p in self.patients:
            if p.status == "waiting":
                p.wait_time += 1
                decay = (6 - p.severity) * self.decay_multiplier
                p.health -= decay
                if p.health <= 0:
                    p.status = "deceased"
                    self.deaths += 1

    def apply_action(self, action_idx):
        reward = 0.05
        if not self.patients:
            return 0.05

        target = self.patients[0]

        if action_idx == 0:  # Wait
            # Only penalise when there are critical patients AND beds available
            critical_waiting = [p for p in self.patients
                                if p.status == "waiting" and p.severity <= 2]
            beds_free = (self.icu_occupied < self.max_icu or
                         self.ward_occupied < self.max_ward)
            if critical_waiting and beds_free:
                reward = 0.05
            else:
                reward = 0.4

        elif action_idx == 1:  # ICU
            if self.icu_occupied < self.max_icu:
                self.icu_occupied += 1
                target.status = "treated"
                # Correct placement: critical patient in ICU
                # Wrong placement: mild patient in ICU (wastes bed but saves patient)
                reward = 0.95 if target.severity <= 2 else 0.05
            else:
                reward = 0.3  # ICU full

        elif action_idx == 2:  # Ward
            if self.ward_occupied < self.max_ward:
                self.ward_occupied += 1
                target.status = "treated"
                # Correct placement: mild patient in ward
                # Suboptimal: critical patient in ward (saved but should be ICU)
                reward = 0.8 if target.severity > 2 else 0.3
            else:
                reward = 0.3  # Ward full

        self.patients = [p for p in self.patients if p.status == "waiting"]
        return reward

    def is_collapsed(self):
        return self.deaths >= self.collapse_threshold

    def get_stats(self):
        waiting = [p for p in self.patients if p.status == "waiting"]
        next_sev = waiting[0].severity if waiting else 0
        next_hp  = waiting[0].health   if waiting else 100.0
        most_urgent = min((p.health for p in waiting), default=100.0)

        return {
            "er_queue_size": len(waiting),
            "icu_available": self.max_icu - self.icu_occupied,
            "ward_available": self.max_ward - self.ward_occupied,
            "avg_patient_health": float(np.mean([p.health for p in waiting])) if waiting else 100.0,
            "critical_count": len([p for p in waiting if p.severity <= 2]),
            "warning_signal": "OVERLOAD" if len(waiting) > 8 else None,
            "next_patient_severity": next_sev,
            "next_patient_health": next_hp,
            "deaths": self.deaths,
            "most_urgent_health": most_urgent,
        }