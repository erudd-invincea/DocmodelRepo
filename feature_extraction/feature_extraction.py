import numpy as np
from sklearn.utils import murmurhash3_32
from sklearn.preprocessing import Normalizer
import sys
import zlib
import math, re
#from sophos.s3.load_key import download_sample
sys.path.append("/home/ubuntu/mount/workspace/datascience")
from datascience.utils.data_access.s3 import download_sample
import zipfile
from StringIO import StringIO

def blockify(blob, nblocks=8):
    l = len(blob)
    blocksize = l//(nblocks)
    indexes = [blocknumber*blocksize for blocknumber in range(nblocks)] + [l]
    blocks = [blob[start:stop] for start,stop in zip(indexes, indexes[1:])]
    return blocks


def lentokens(data, hashbins):
        feats = re.findall(r"([^\x00-\x7F]+|\w+)", data)
        final_feats = []
        rv = np.zeros(8*hashbins)
        for feat in feats:
            loglength = int(min(8,max(1,math.log(len(feat),1.4))))-1 # 0-7
            shash = murmurhash3_32(feat) % (hashbins)
            rv[loglength*(hashbins)+shash]+=1

        return rv

def ef(key):
    blob = download_sample('file/{}'.format(key), bucket='invincea-binary-feed')
    blocks = blockify(blob, 16)
    rfv = np.asarray(map(lambda x:lentokens(x, 128), blocks), dtype=np.float32)
    nfv = Normalizer().transform(rfv)
    return zlib.compress(nfv.tobytes())

def deserialize(blob):
    return np.fromstring(blob, dtype=np.float32).reshape((16, -1))

# def unzip(blob):
# def get_xml_contents(blob):
#     """
#     unzip and concatenate xml contents in memory. 
#     sometimes fails. call should be surrounded w/ try catch
#     """
#     fp = StringIO(blob)
#     zfp = zipfile.ZipFile(fp,"r")
#     contents = "\x00".join([zfp.read(fname) for fname in zfp.namelist()]) + "\x00" + "\x00".join(zfp.namelist())
#     zfp.close()
#     fp.close()
#     return contents,err_message
    

str_length_dim = 16
str_string_dim = 64
str_log_switch = 10
str_max_line = 2 ** 7
str_min_line = 5
# get min, max value
str_min_value = np.log(str_log_switch - str_log_switch + 3)
str_max_value = np.log(str_max_line - str_log_switch + 3)
str_xedges = np.linspace(str_min_value, str_max_value + 1e-8, str_length_dim - str_log_switch + str_min_line + 1,
                         endpoint=True) + str_log_switch
str_yedges = np.arange(0, str_string_dim + 1)
str_xedges = np.concatenate((range(str_min_line, str_log_switch), str_xedges))

from string import printable
import re

hasher = murmurhash3_32

def splitter_factory(split_chars):
    splitter_re = re.compile("["+"".join(split_chars)+"]")
    def splitter(s, minlength):
        return filter(lambda x:len(x)>minlength, splitter_re.split(s))
    return splitter

nonprintable = [chr(x) for x in range(256) if chr(x) not in printable]
strings = splitter_factory(nonprintable)
nonprintable_strings = splitter_factory([x for x in printable])

strings_xml_split =  splitter_factory(nonprintable + ['>']+ ['<'] + [" "])


def kb_string_features(val):
    lines = strings(val, str_min_line)
    xs = []
    ys = []
    for line in lines:
        line = line.strip()
        line = line[:str_max_line]
        l = len(line)
        if l < str_log_switch:
            length = l
        else:
            length = np.log(l - str_log_switch + 3) + str_log_switch
        xs.append(length)
        h = (hasher(line) % str_string_dim)
        ys.append(h)
    bins = np.histogram2d(xs, ys, bins=(str_xedges, str_yedges))[0]
    v = bins
    v = np.log(v + 1.0)
    return v.ravel()

def ef_kb_xmlsplit(val,splitfunc=strings_xml_split):
    lines = splitfunc(val, str_min_line)
    xs = []
    ys = []
    for line in lines:
        line = line.strip()
        line = line[:str_max_line]
        l = len(line)
        if l < str_log_switch:
            length = l
        else:
            length = np.log(l - str_log_switch + 3) + str_log_switch
        xs.append(length)
        h = (hasher(line) % str_string_dim)
        ys.append(h)
    bins = np.histogram2d(xs, ys, bins=(str_xedges, str_yedges))[0]
    v = bins
    v = np.log(v + 1.0)
    return v.ravel()

def ef_kb_unzip_xmlsplit(val):
    """
    unzip and concatenate the contents of a .docx file in memory
    """
    fp = StringIO(val) 
    if not zipfile.is_zipfile(fp):
        print "...not zipfile"
        contents = val
        fp.close()
    else:
        #print "...zipfile"
        zfp = zipfile.ZipFile(fp,"r")
        contents = "\x00".join([zfp.read(fname) for fname in zfp.namelist()]) + "\x00" + "\x00".join(zfp.namelist())
        zfp.close()
        fp.close()
    return ef_kb_xmlsplit(contents)

def ef_kb(key):
    blob = download_sample('file/{}'.format(key), bucket='invincea-binary-feed')
    blocks = blockify(blob, 16)
    rfv = np.asarray(map(lambda x:kb_string_features(x), blocks), dtype=np.float32)
    return zlib.compress(rfv.tobytes())


