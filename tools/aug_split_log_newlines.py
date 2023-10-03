import random

log_file_path = "./data/raw/BGL.log"
#log_file_path = "./data/raw/Spirit1G.log"
#log_file_path = "./data/raw/Thunderbird10M.log"

train_path = "./data/split/BGL_train80.log"
test_path  = "./data/split/BGL_test20.log"
#train_path = "./data/split/Spirit1G_train80.log"
#test_path  = "./data/split/Spirit1G_test20.log"
#train_path = "./data/split/Thunderbird10M_train80.log"
#test_path  = "./data/split/Thunderbird10M_test20.log"

# Load your logs from the log file
with open(log_file_path, mode="r", encoding='utf8',errors='ignore') as file:
    logs = file.readlines()

total_samples = len(logs)
train_ratio = 0.8
train_samples = int(total_samples * train_ratio)

train_logs = logs[:train_samples]
test_logs = logs[train_samples:]

# Save the train logs to a text file
with open(train_path, 'w') as file:
    file.writelines(train_logs)

# Save the test logs to a text file
with open(test_path, 'w') as file:
    file.writelines(test_logs)
