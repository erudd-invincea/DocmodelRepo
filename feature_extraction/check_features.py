import sys
from traitlets.config.loader import Config
from IPython.terminal.embed import InteractiveShellEmbed
ipshell = InteractiveShellEmbed(config=Config(),
                       banner1 = 'Dropping into IPython',
                       exit_msg = 'Leaving Interpreter, back to program.')

sys.path = ["/home/ubuntu/mount/workspace/archive_branch/datascience/datascience/projects/archive/data_acquisition"] + sys.path
from lmdb_utils import *

import sqlite3

# select XML files only
# file_types = {
#     'xml':{ u'Office Open XML Spreadsheet', u'Office Open XML Document', u'Office Open XML Presentation'},
#     'ole':{u'MS PowerPoint Presentation',u'MS Excel Spreadsheet',u'MS Word Document'}
# }

md_db_loc = "/home/ubuntu/mount/doc_learn_6/scans_meta.db"
conn = sqlite3.connect(md_db_loc)
c = conn.cursor()
files = c.execute("""select sha256,file_type from scans where (file_type={} or file_type={} or file_type={}) limit 10""".format(u'\"Office Open XML Spreadsheet\"',u'\"Office Open XML Document\"',u'\"Office Open XML Presentation\"')).fetchall()
conn.close()

print files[:10]

feat_lmdb_loc = "/home/ubuntu/mount/doc6_features_kb_split_lmdb"
feat_ftdb = "kb_string_features_xml_decompress_split"
file_lmdb_loc = "/home/ubuntu/mount/doc_learn_6/file_lmdb"
file_ftdb = None

file_getter = FileLMDBGetter(file_lmdb_loc,file_ftdb)
feature_getter = FeatureLMDBGetter(feat_lmdb_loc,feat_ftdb)

files = file_getter.get(map(lambda x: x[0],files))
features =  feature_getter.get(map(lambda x: x[0],files))

assert len(files) == len(features), "File and feature lengths differ"

#feature extraction code
from feature_extraction import ef_kb_unzip_xmlsplit
featx = ef_kb_unzip_xmlsplit
N = len(files)
for i in xrange(len(files)):
    print "{} / {}".format(i,N)
    #print all(features[i][1] == featx(files[i][1]))
    print features[i][1]
    feats = featx(files[i][1])
    close =  np.isclose(features[i][1],featx(files[i][1]))
    non_zero = np.where(feats != 0)
    print len(non_zero[0])
    print all(close)
    raw_input('press key')
ipshell()

