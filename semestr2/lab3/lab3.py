import time
import random
import pg8000
import matplotlib.pyplot as plt

DB_CONFIG = {
    "database": "",
    "user": "postgres",
    "password": "",
    "host": "127.0.0.1",
    "port": 5432
}


class ParkingOntology:
    def __init__(self):
        self.nodes = {"Parking": {"capacity": 100, "current_cars": 10}}
        self.params = self.load_params_from_db()

    def load_params_from_db(self):
        conn = None
        try:
            conn = pg8000.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT low_limit, med_limit, high_limit FROM fuzzy_parameters WHERE param_name = 'occupancy'")
            row = cursor.fetchone()
            cursor.close()
            if row:
                return {"low": row[0], "med": row[1], "high": row[2]}
        except Exception as e:
            print(f"Ошибка БД (загрузка): {e}. Используем стандарты.")
        finally:
            if conn:
                conn.close()
        return {"low": 40, "med": 20, "high": 60}


class FuzzyController:
    @staticmethod
    def fuzzify_occupancy(percent, params):
        low = max(0, min(1, (params['low'] - percent) / params['low']))
        medium = max(0, min((percent - params['med']) / 30, (80 - percent) / 30))
        high = max(0, min(1, (percent - params['high']) / 40))
        return {"low": low, "medium": medium, "high": high}

    @staticmethod
    def defuzzify(scores):
        return (scores['low'] * 1.0 + scores['medium'] * 0.4 + scores['high'] * 0.0)


class ParkingSimulator:
    def __init__(self):
        self.ontology = ParkingOntology()
        self.ctrl = FuzzyController()
        self.capacity = self.ontology.nodes["Parking"]["capacity"]
        self.hour = 7

        
        self.hours = []
        self.occupancy_history = []
        self.arrived_history = []
        self.departed_history = []
        self.access_index_history = []

    def log_to_db(self, time_str, occ, arrived, departed, idx):
        conn = None
        try:
            conn = pg8000.connect(**DB_CONFIG)
            cursor = conn.cursor()
            query = "INSERT INTO simulation_history (sim_time, occupancy_pct, arrived, departed, access_index) VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(query, (time_str, occ, arrived, departed, idx))
            conn.commit()
            cursor.close()
        except Exception as e:
            print(f"Ошибка БД (запись): {e}")
        finally:
            if conn:
                conn.close()

    def get_traffic_intensity(self):
        if (8 <= self.hour <= 10) or (17 <= self.hour <= 19):
            return "RUSH_HOUR"
        return "NORMAL"

    def run_step(self):
        current_cars = self.ontology.nodes["Parking"]["current_cars"]
        occupancy_pct = (current_cars / self.capacity) * 100
        traffic = self.get_traffic_intensity()

        fuzzy_vals = self.ctrl.fuzzify_occupancy(occupancy_pct, self.ontology.params)
        access_index = self.ctrl.defuzzify(fuzzy_vals)

       
        base_demand = random.randint(5, 10)
        if traffic == "RUSH_HOUR":
            base_demand = random.randint(15, 30)

        arrived = int(base_demand * access_index)

        if 8 <= self.hour <= 11:
            departed = random.randint(0, 2)
        elif 17 <= self.hour <= 20:
            departed = random.randint(7, 12)
        else:
            departed = random.randint(1, 5)

        new_total = current_cars + arrived - departed
        self.ontology.nodes["Parking"]["current_cars"] = max(0, min(self.capacity, new_total))

        time_str = f"{self.hour:02d}:00"

        
        self.log_to_db(time_str, occupancy_pct, arrived, departed, access_index)

       
        self.hours.append(self.hour)
        self.occupancy_history.append(occupancy_pct)
        self.arrived_history.append(arrived)
        self.departed_history.append(departed)
        self.access_index_history.append(access_index)

        status_icon = "повышенный" if traffic == "RUSH_HOUR" else "обычный"
        print(
            f"[{time_str}] {status_icon} Загрузка: {occupancy_pct:5.1f}% | Въехало: {arrived:2d} | Уехало: {departed:2d} | Индекс: {access_index:.2f}")

        if access_index < 0.2:
            print("СИСТЕМА: Въезд ограничен (Парковка заполнена).")

        self.hour = (self.hour + 1) % 24

    def plot_results(self):
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

       
        ax1.plot(self.hours, self.occupancy_history, 'b-', marker='o')
        ax1.set_ylabel('Загруженность (%)')
        ax1.set_title('Динамика загруженности парковки')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)

        
        ax2.bar([h - 0.15 for h in self.hours], self.arrived_history, width=0.3, label='Въехало', color='green')
        ax2.bar([h + 0.15 for h in self.hours], self.departed_history, width=0.3, label='Уехало', color='red')
        ax2.set_ylabel('Количество машин')
        ax2.set_title('Поток автомобилей')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

       
        ax3.plot(self.hours, self.access_index_history, 'm-', marker='s')
        ax3.set_ylabel('Индекс доступа')
        ax3.set_xlabel('Час дня')
        ax3.set_title('Нечёткий индекс доступа к парковке')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.05)

        plt.xticks(range(24))  
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    sim = ParkingSimulator()

    for _ in range(18):
        sim.run_step()
        time.sleep(0.2)

    sim.plot_results()
