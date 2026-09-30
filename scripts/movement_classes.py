from datetime import datetime
import graphcheck as gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


date_str = datetime.today().strftime('%Y-%m-%d')
date_time_str = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')


# Common output directory
path_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + "/"
path_movement_classes = path_local + "results/movement_classes/"
os.makedirs(path_movement_classes, exist_ok=True)


data = []
sessions = ["071102", "071107", "071109", "071213", 
            "071214", "071218", "071221", "071227", 
            "080213", "071228", "080206", "080208", 
            "080227", "080220", "080222"]


# === IMPORTANT ===
# This analysis needs the raw data files
path_data_root = "/path/to/data/root/"  # replace with the actual path to your data root


for session in sessions:
    for start in [100, 3600]:
        if start == 100:
            stage = "Early"
        else:
            stage = "Late"

        start = start * 1000
        end = 20 * 60 * 1000 + start
        freq = 20000

        lever = gc.open_file(path_data_root+session, "Lever.dat", start, end, freq, np.int16)
        event = gc.open_file(path_data_root+session, "Event.dat", start, end, freq, np.int16)

        lever[1] = gc.norm_min_max(lever[1])
        event[1] = gc.norm_max_abs(event[1])

        times_success, times_failure = gc.calculate_success_failure(event)
        avg_iri, avg_iti = gc.calculate_intertimes(times_success, times_failure)
        event_count = len(times_success) + len(times_failure)

        row = {
            "Session": str(int(session)),
            "Stage": stage,
            "avg_iri": avg_iri,
            "avg_iti": avg_iti,
            "event_count": event_count + 1,
            "rate_success": len(times_success) / event_count,
            "rate_failure": len(times_failure) / event_count
            }
        
        data.append(row)


# Includes all sessions and stages
df = pd.DataFrame(data)


mean_avg_iti = df['avg_iti'].mean()
median_avg_iti = df['avg_iti'].median()
std_avg_iti = df['avg_iti'].std()
print("Mean value of average inter-trial intervals: ", mean_avg_iti)
print("Median value of average inter-trial intervals: ", median_avg_iti)
print("Standard deviation of average inter-trial intervals: ", std_avg_iti)

median_rate_success = df['rate_success'].median()
print("Median value of rate of success: ", median_rate_success)

iti_cut = mean_avg_iti
reward_cut = 0.6

# Motivated stage: average inter-trial interval < mean_avg_iti (~3.5s)
# Unmovitated stage: average inter-trial interval >= mean_avg_iti (~3.5s)
df['Motivation'] = df['avg_iti'].apply(lambda x: "Motivated" if x < iti_cut else "Unmotivated")

# Rewarded stage: rate of success > 0.6
# Unrewarded stage: rate of success <= 0.6
df['Reward'] = df['rate_success'].apply(lambda x: "Rewarded" if x > reward_cut else "Unrewarded")

df['Category'] = df['Motivation'].str[0] + df['Reward'].str[0]


with open(path_movement_classes + "statistics.txt", "w") as file:
    file.write(f"Statistics for all sessions and stages\n")
    file.write(f"Mean value of average inter-trial intervals: {mean_avg_iti}\n")
    file.write(f"Median value of average inter-trial intervals: {median_avg_iti}\n")
    file.write(f"Standard deviation of average inter-trial intervals: {std_avg_iti}\n")
    file.write(f"Median value of rate of success: {median_rate_success}\n")
    file.write("\n")

    file.write(f"Motivated stage: average inter-trial interval < {iti_cut}\n")
    file.write(f"Rewarded stage: rate of success > {reward_cut}\n")
    file.write("\n")
    file.write(f"Mean value of average inter-trial intervals: {mean_avg_iti}\n")
    file.write(f"Median value of average inter-trial intervals: {median_avg_iti}\n")
    file.write(f"Standard deviation of average inter-trial intervals: {std_avg_iti}\n")
    file.write(f"Median value of rate of success: {median_rate_success}\n")


# Filter the data to keep only the motivated and rewarded trials
df_analysis = df[(df['Motivation'] == 'Motivated') & (df['Reward'] == 'Rewarded')].copy().reset_index(drop=True)


fig, ax = plt.subplots()
df.groupby(['Motivation', 'Reward']).size().plot(kind='bar', ax=ax)
ax.set_ylabel("Count")
ax.set_xlabel("Motivation, Reward")
fig.savefig(path_movement_classes + "motivation_reward.jpg", dpi=300, bbox_inches='tight')


df.to_csv(path_movement_classes + "allsessions_success_failure_clean.csv")
df_analysis.to_csv(path_movement_classes + "allsessions_success_failure_motivated_rewarded.csv")