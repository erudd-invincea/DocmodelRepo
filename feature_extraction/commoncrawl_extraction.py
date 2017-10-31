import sys, time
from multiprocessing import cpu_count
sys.path.append("/home/ubuntu/mount/workspace/datascience")
import zlib
import numpy as np
import gzip
import json
# from feature_extraction import blockify, kb_string_features, ef_kb
# from feature_extraction import ef_kb_unzip_xmlsplit
import lmdb
from datascience.utils.data_access.s3 import list_s3_keys, list_s3_sha256_keys_parallel
from datascience.utils.data_access.s3 import download_sample
from datascience.personal.rharang.generators.generators_new import *
import os
import time 
import zlib
import numpy as np
from multiprocessing import cpu_count
import multiprocessing

from traitlets.config.loader import Config
from IPython.terminal.embed import InteractiveShellEmbed
ipshell = InteractiveShellEmbed(config=Config(),
                       banner1 = 'Dropping into IPython',
                       exit_msg = 'Leaving Interpreter, back to program.')


class bulk_extractor(queued_unordered_generator):
    def run(self):
        ct = 0
        tot = len(self.keys)
        t0 = time.time()
        while True:
            try:
                v = None
                v = self.next()
                ct+=1
                t1 = time.time()
                dt = t1-t0
                if ct%100 == 0:
                    sys.stdout.write("\r{} / {}  ({} s; {} items/s ) {}  {}".format(ct, tot, dt, ct/dt, self.oq.empty(), self.oq.qsize()))
                    sys.stdout.flush()
            except StopIteration:
                print "Caught StopIteration"
                break
        sys.stdout.write("\r{} / {}  ({} s; {} items/s ) {}  {}".format(ct, tot, dt, ct/dt, self.oq.empty(), self.oq.qsize()))
        sys.stdout.flush()
        print type(v)

def get_keys_from_s3(bucket='rharang',
                     prefix='web-office-files',
                     use_keycache=True,
                     keycache="/home/ubuntu/mount/workspace/datascience/datascience/personal/erudd/docmodel/feature_extraction/keycache.txt", # where original keys were stored
                     verbose=False,
                     limit=1000):
    if use_keycache:
        if os.path.isfile(keycache):
            with open(keycache,"r") as f:
                keys = f.read().splitlines()
        else:
            keys = list_s3_sha256_keys_parallel(bucket='rharang', prefix='', cpus=cpu_count())
            with open(keycache,"w") as f:
                f.write("\n".join(keys))
    else: 
        keys = list_s3_keys('rharang','web-office-files/',retries=3)
    return keys



def main():
    keylist = get_keys_from_s3()
    print keylist[:10]
    dbloc = "/home/ubuntu/mount/common_crawl_lmdb_fixed"
    ftdb = "kb_string"
    os.system("mkdir -p {}".format(dbloc))
    func = ef_kb_unzip_xmlsplit
    lmdbgetter=LMDBGetOrExtract(func, dbloc, ftdb, max_readers=512)
    extractor = bulk_extractor(keylist, lmdbgetter, repeat=False, shuffle=False, nthreads=2*cpu_count() , oqmax=1, iqmax=512)
    extractor.run()
    
if __name__=="__main__":
    main()
            
