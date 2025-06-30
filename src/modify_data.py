def clean_large_mtx(input_path, output_path):
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if line.startswith('%'):
                # f_out.write(line)
                continue
            
            cleaned = line.lstrip()
            cleaned = ' '.join(cleaned.split())  # 替代正则表达式
            f_out.write(cleaned + '\n')

clean_large_mtx("solverchallenge25_09_A.mtx", "solverchallenge25_09_A_clean.mtx")