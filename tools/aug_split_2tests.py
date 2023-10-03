import random

log_file_path = "./data/split/BGL_test20.log"
#log_file_path = "./data/raw/Spirit1G.log"
#log_file_path = "./data/raw/Thunderbird10M.log"

test1_path  = "./data/split/BGL_test10_1.log"
test2_path  = "./data/split/BGL_test10_2.log"

# Load your logs from the log file
with open(log_file_path, mode="r", encoding='utf8',errors='ignore') as file:
    logs = file.readlines()

total_samples = len(logs)
Split_ratio = 0.5
test1_samples = int(total_samples * Split_ratio)

test1_logs = logs[:test1_samples]
test2_logs = logs[test1_samples:]

# Save the train logs to a text file
with open(test1_path, 'w') as file:
    file.writelines(test1_logs)

# Save the test logs to a text file
with open(test2_path, 'w') as file:
    file.writelines(test2_logs)