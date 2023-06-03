import os
import sys
import csv
from shuyi_gen import main as pen_main
from haoyu_gen import main as card_main
from cheng_gen import main as index_main

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: generate_properties.py <random seed>")
        exit(1)

    if not os.path.exists('vnnlib'):
        os.makedirs('vnnlib')
    seed = sys.argv[1]
    csv_data = []
    print("Generating Video Stream specifications, this may take around several minutes...")
    csv_data.extend(pen_main(seed))
    print("Generating Index specifications, this may take several minutes...")
    csv_data.extend(index_main(seed))
    print("Generating cardinality specifications, this may take around ten minutes...")
    csv_data.extend(card_main(seed))
    print(f"Successfully generate {len(csv_data)} files!")
    print(f"Total timeout is {sum([int(i[-1]) for i in csv_data])}")
    with open('instances.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)