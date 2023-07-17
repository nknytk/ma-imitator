from multiprocessing import Pool
import os
import sys


def count_chars(data_dir):
    num_workers = max(1, int(os.cpu_count() / 2))
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    with Pool(num_workers) as p:
        counteds = p.map(count, files)

    merged = {}
    for cs in counteds:
        for c in cs:
            merged[c] = merged.get(c, 0) + cs[c]

    merged_list = list(merged.items())
    merged_list.sort(key=lambda x: x[1], reverse=True)
    total_chars = sum(c[1] for c in merged_list)
    covered = 0
    with open('data/chars.csv', mode='w') as fp:
        for i, (c, cnt) in enumerate(merged_list):
            covered += cnt
            fp.write(f'{i + 1},{c},{cnt},{covered/total_chars}\n')



def count(file_path):
    cs = {}
    with open(file_path) as fp:
        for row in fp:
            for c in row.strip():
                cs[c] = cs.get(c, 0) + 1
    return cs


if __name__ == '__main__':
    count_chars(sys.argv[1])
