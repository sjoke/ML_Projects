import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
import logging as lg
import queue
import time
import threading
from pathlib import Path
import argparse

lg.basicConfig(filename='no_image_items.log', level=lg.INFO, format='%(message)s')

q = queue.Queue(maxsize=1e4)

def writer(i):
    print('writer ', i, 'start')
    while True:
        item = q.get()
        if item is None:
            print('no item')
            break
        fp, content = item
        # print(i, 'receive', fp)
        with open(fp, 'wb') as f:
            f.write(content)
        q.task_done()
        # print(i, 'task done')


def download(item_id, fp, url, sess):
    r = sess.get(url, stream=True)
    if r.status_code == 200:
        # print('put', item_id)
        q.put((fp, r.content))
    else:
        lg.info(item_id)


def main(item_f, workers, writers, save_dir, out=False):
    rows = pd.read_csv(item_f)
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers, 
                                            pool_maxsize=workers)
    threads = []
    for i in range(writers):
        t = threading.Thread(target=writer, args=(i,))
        t.start()
        threads.append(t)

    with ThreadPoolExecutor(workers) as p, requests.Session() as sess:
        sess.mount('http://*******.com', adapter)
        # sess.mount('image.momoso.com', HTTP20Adapter())
        
        for (i, row) in rows.iterrows():
            if i % 100000 == 0:
                print('cnt:', i)
            item_id, url, cate1 = row[0], row[1], row[2]
            fp = save_dir / str(cate1)
            if not fp.exists():
                # make dir
                os.mkdir(fp)

            fp = fp / str(item_id)
            if fp.exists() and fp.stat().st_size > 0:
                continue
            p.submit(download, item_id, fp, url, sess)
        print('wait downloads finish')
        p.shutdown()
    
    print('wait task finish')
    q.join()
    for i in range(writers):
        q.put(None)
    for t in threads:
        t.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('params')
    parser.add_argument('--items', default='items.csv', type=str)
    parser.add_argument('--workers', default=32, type=int)
    parser.add_argument('--writers', default=8, type=int)
    parser.add_argument('--dir', default='image_set', type=str)
    parser.add_argument('--out', default=False, action='store_true')
    args = parser.parse_args()

    save_dir = Path(args.dir)
    if not save_dir.exists():
        os.makedirs(save_dir)
    
    t0 = time.time()
    main(args.items, args.workers, args.writers, save_dir, args.out)
    t1 = time.time()
    print("time: ", t1-t0)

