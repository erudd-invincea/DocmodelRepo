from __future__ import print_function
import sys
sys.path.append('/home/ubuntu/mount/workspace/kb_branch/datascience/')
sys.path = ["/home/ubuntu/mount/workspace/archive_branch/datascience/datascience/projects/archive/data_acquisition"] + sys.path
from lmdb_utils import *
from datascience.personal.rharang.generators.generators_new import *
import zlib
import config
import numpy as np
import zipfile
from io import BytesIO
from lmdb_utils import *
from string import printable
from sklearn.utils import murmurhash3_32
from file_selector import file_selector
from multiprocessing import cpu_count
import time
import re
import argparse
import sqlite3
from feature_extraction import ef_kb_unzip_xmlsplit
import time

@singleton
class SFileLMDBGetter(FileLMDBGetter):
    def __init__(self,dbloc,ftdb,decompress=True):
        FileLMDBGetter.__init__(self,dbloc,ftdb,decompress=True)

file_lmdb_loc = "/home/ubuntu/mount/doc_learn_6/file_lmdb"
file_ftdb = None

#instantiate singleton
SFileLMDBGetter(file_lmdb_loc,file_ftdb,decompress=True)

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
                sys.stdout.write("\r{} / {}  ({} s; {} items/s ) {}  {}".format(ct, tot, dt, ct/dt, self.oq.empty(), self.oq.qsize()))
                sys.stdout.flush()
            except StopIteration:
                print("Caught StopIteration")
                break
        sys.stdout.write("\r{} / {}  ({} s; {} items/s ) {}  {}".format(ct, tot, dt, ct/dt, self.oq.empty(), self.oq.qsize()))
        sys.stdout.flush()

def get_shas(limit):
    # file_types = {
    #     'xml':{u'Office Open XML Spreadsheet', u'Office Open XML Document', u'Office Open XML Presentation'},
    #     'ole':{u'MS PowerPoint Presentation',u'MS Excel Spreadsheet',u'MS Word Document'},
    #     'all':{u'Office Open XML Spreadsheet', u'Office Open XML Document', u'Office Open XML Presentation',
    #            u'MS PowerPoint Presentation',u'MS Excel Spreadsheet',u'MS Word Document'}
    # }
    # TODO: incorporate fetchall_fast
    md_db_loc = "/home/ubuntu/mount/doc_learn_6/scans_meta.db"
    conn = sqlite3.connect(md_db_loc)
    c = conn.cursor()
    if limit != None:
        shas = c.execute("""select sha256 from scans where (file_type={} or file_type={} or file_type={}) limit {}""".format(u'\"Office Open XML Spreadsheet\"',u'\"Office Open XML Document\"',u'\"Office Open XML Presentation\"',limit)).fetchall()
    else:
        shas = c.execute("""select sha256 from scans where (file_type={} or file_type={} or file_type={})""".format(u'\"Office Open XML Spreadsheet\"',u'\"Office Open XML Document\"',u'\"Office Open XML Presentation\"')).fetchall()
    conn.close()
    return map(lambda x: x[0],shas)

def npserializer(nparray, dt):
    x = np.asarray(nparray, dtype=dt)
    zx = zlib.compress(x.tobytes())
    return zx

def npf32serializer(x):return npserializer(x, dt=np.float32)

def extract_feats(key):
    blob = SFileLMDBGetter().get_one(key)
    return npf32serializer(ef_kb_unzip_xmlsplit(blob))

def main(args):
    print("...getting keylist")
    print("...debug mode: {}".format(args.debug))
    if args.debug:
        keylist = get_shas(1000)
    else:
        keylist = get_shas(None)
    print('...spawning lmdb getter')
    lmdbgetter=LMDBGetOrExtract(extract_feats,args.dbloc,args.ftdb, max_readers=512)
    # for k in keylist:
    #     feat = lmdbgetter.get(k)
    #     print(feat)
    #     raw_input('press key')
    time.sleep(5)
    extractor = bulk_extractor(keylist, lmdbgetter, repeat=False, shuffle=False, nthreads=2*cpu_count(), oqmax=1, iqmax=256)
    extractor.run()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbloc',required=True,type=str,help='location for the output lmdb of features')
    parser.add_argument('--ftdb',required=True,type=str,help='name of the sub database w/in the new lmdb')
    parser.add_argument('--debug',default=False,type=bool,help='if enabled only do a dry run of 1K samples')
    parser.add_argument('--file_type',default='xml',type=str,choices=['xml','ole','all'],help='file formats to extract')
    args = parser.parse_args()
    print('args.debug: {}'.format(args.debug))
    main(args)


