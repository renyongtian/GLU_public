import sys

if len(sys.argv) != 3:
    print("Usage: python css2csr.py input.mtx out.mtx")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

comments = []
data_lines = []
header_line = None
reading_data = False

with open(input_file, 'r') as f:
    for line in f:
        if line.startswith('%'):
            comments.append(line)
        elif not reading_data:
            header_line = line  # 第一条非注释的行（维度行）
            reading_data = True
        else:
            parts = line.strip().split()
            if len(parts) == 3:
                i, j, val = parts
                data_lines.append((int(i), int(j), float(val)))

data_lines.sort(key=lambda x: x[0])

with open(output_file, 'w') as f:
    for line in comments:
        f.write(line)
    f.write(header_line)
    for i, j, val in data_lines:
        f.write(f"{i} {j} {val}\n")
