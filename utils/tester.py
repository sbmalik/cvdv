import time
from tqdm import tqdm

for i in tqdm(range(10)):
    tqdm.write('Hello')
    time.sleep(0.1)
