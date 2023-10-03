import sys
import re
import json

def convert_to_json(input_file, output_file):
    output_data = []

    with open(input_file, mode="r", encoding='utf8',errors='ignore') as file:
        for line in file:
            if line.startswith('-'):
                label = 0
            else:
                label = 1

            dec_line = line[line.find(' ') + 1:]              # remove label from log messages
            inc_line = ""

            entry = {
                "Dec": [dec_line],
                "Inc": [inc_line],
                "labels": label
            }

            output_data.append(entry)

    with open(output_file, "w") as fo:
        json.dump(output_data, fo, indent=2)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_to_json.py <input_log_file> <output_json_file>")
        sys.exit(1)

    input_log_file = sys.argv[1]
    output_json_file = sys.argv[2]

    convert_to_json(input_log_file, output_json_file)
    print("Conversion complete!")
