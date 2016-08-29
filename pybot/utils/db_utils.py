# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import tables as tb
import numpy as np
import pickle
import time, logging, cPickle, shelve
from collections import defaultdict
from itertools import izip

import os.path
from pybot.utils.misc import progressbar
from pybot.utils.io_utils import create_path_if_not_exists
from pybot.utils.itertools_recipes import grouper

# =============================================================================
# Pytables helpers
# =============================================================================

# Avoids printing natural name warning when writing 
# dictionaries with ints as keys
import warnings
warnings.filterwarnings('ignore', category=tb.NaturalNameWarning)

def load_pickled_dict(fn): 
    return pickle.load(open(os.path.expanduser(fn), 'rb'))

def save_pickled_dict(fn, d): 
    pickle.dump(d, open(os.path.expanduser(fn), 'w'))

def load_json_dict(fn): 
    import json
    return json.load(open(os.path.expanduser(fn), 'r'))

def save_json_dict(fn, d, pretty=False): 
    import json
    with open(os.path.expanduser(fn), 'w') as fp:
        if pretty: 
            json.dump(d, fp, sort_keys=True, indent=4, separators=(',', ':'))
        else: 
            json.dump(d, fp)

def load_yaml_dict(fn): 
    import yaml
    return yaml.load(open(os.path.expanduser(fn), 'r'))

def save_yaml_dict(fn, d): 
    import yaml
    with open(os.path.expanduser(fn), 'w') as fp:
        fp.write(yaml.dump(d, default_flow_style=False))

def load_mat(fn): 
    import scipy.io as io
    return io.loadmat(os.path.expanduser(fn))

def save_mat(fn, d): 
    import scipy.io as io
    io.savemat(os.path.expanduser(fn), d)
    
def read_pytable(h5f, group=None): 
    if group is None: group = h5f.root

    data = AttrDict()
    for child in h5f.list_nodes(group): 
        item = None
        try: 
            if isinstance(child, tb.group.Group): 
                item = read_pytable(h5f, child)
            else: 
                item = child.read()
                if isinstance(item, str) and item.startswith('OBJ_'): 
                    item = cPickle.loads(item[4:])
            data[child._v_name] = item
        except tb.NoSuchNodeError:
            warnings.warn('No such node: "%s", skipping...' % repr(child))
            pass

    return data

def load_pytable(fn): 
    try: 
        h5f = tb.open_file(os.path.expanduser(fn), mode='r', title='Title: %s' % fn)
        data = read_pytable(h5f, group=h5f.root)
        h5f.close()
    except Exception as e: 
        raise RuntimeError('{}'.format(e))
        data = AttrDict()
    return data

def get_node(g, k): 
    if g._v_pathname.endswith('/'):
        return ''.join([g._v_pathname,k])
    else: 
        return ''.join([g._v_pathname,'/',k])

def flush_pytable(h5f, data=None, group=None, table=None, force=True): 
    # if data is None: data = self.data
    # if table is None: table = self.tables
    # if group is None: group = self.groups

    # print 'Keys: ', data.keys()
    for k,v in data.iteritems(): 
        # print 'key,val', k,v, type(v)

        try: 
            k = str(k)
        except: 
            print 'Cannot save to DB, key is not string %s ' % k
            continue

        # if not isinstance(k, str): 
        #     continue

        # Clean up before writing 
        if force: 
            try:
                h5f.remove_node(get_node(group._gp,k), recursive=True) 
            except tb.NoSuchNodeError:
                pass

        # print 'In Group: ', group, k, v                
        if isinstance(v, dict):
            # self.log.debug('Attempting to save dict type')
            # assert(k not in table);
            table[k] = AttrDict()
            group[k] = AttrDict();
            group[k]._gp = h5f.create_group(group._gp, k)
            h5f.flush()
            # self.log.debug('Out Group: %s' % group[k])
            flush_pytable(h5f, data=v, group=group[k], table=table[k])
        elif isinstance(v, np.ndarray): 
            # self.log.debug('Attempting to save ndarray %s' % type(v))
            table[k] = h5f.create_array(group._gp, k, v)
            # self.log.debug('Out Table: %s' % table[k])
        else: 
            # print 'Attempting to save arbitrary type %s' % type(v), k, group._gp
            try: 
                assert v is not None
                table[k] = h5f.create_carray(group._gp, k, obj=v)
            except (TypeError, ValueError, AssertionError): 
                v = 'OBJ_' + cPickle.dumps(v, -1)
                table[k] = h5f.create_array(group._gp, k, v)
                # print 'TypeError', v
            finally: 
                h5f.flush()
    return 

def save_pytable(fn, d): 
    create_path_if_not_exists(fn)
    h5f = tb.open_file(os.path.expanduser(fn), mode='w', title='%s' % fn)

    tables = AttrDict()
    groups = AttrDict()
    groups._gp = h5f.root

    flush_pytable(h5f, data=d, group=groups, table=tables)
    h5f.close()

# =============================================================================
# AttrDict
# =============================================================================

class AttrDict(dict): 
    def __init__(self, *args, **kwargs): 
        super(AttrDict, self).__init__(*args, **kwargs)
        
    def __getitem__(self, attr): 
        return super(AttrDict, self).__getitem__(attr)

    def __setitem__(self, attr, value): 
        super(AttrDict, self).__setitem__(attr, value)

    def __getattr__(self, attr): 
        return super(AttrDict, self).__getitem__(attr)

    def __setattr__(self, attr, value): 
        super(AttrDict, self).__setitem__(attr, value)

    def __getstate__(self): 
        pass

    def __setstate__(self): 
        pass

    def to_dict(self): 
        return dict(self)

    @staticmethod
    def load_dict(fn): 
        return AttrDict(load_pickled_dict(fn))

    def save_dict(self, fn): 
        save_pickled_dict(fn, self)
        print 'saving ', self

    @staticmethod
    def load_json(fn): 
        return AttrDict(load_json_dict(fn))

    def save_json(self, fn): 
        save_json_dict(fn, self)

    @staticmethod
    def load_yaml(fn): 
        return AttrDict(load_yaml_dict(fn))

    def save_yaml(self, fn): 
        print self.to_dict()
        save_yaml_dict(fn, self.to_dict())

    @staticmethod
    def load_mat(fn): 
        return AttrDict(load_mat(fn))

    def save_mat(self, fn): 
        save_mat(fn, self)

    @staticmethod
    def load(fn): 
        return load_pytable(fn)

    def save(self, fn): 
        fn = os.path.expanduser(fn)
        print 'Saving ', fn
        create_path_if_not_exists(fn)
        return save_pytable(fn, self)

class IterDBChunk(object): 
    def __init__(self, filename, mode, fields=[]): 
        pass

class IterDB(object): 
    def __init__(self, filename, mode, fields=[], batch_size=5): 
        """
        An iterable database that should theoretically allow 
        scalable reading/writing of datasets. 
           batch_size: length of list

        Notes: 
           meta_file should contain all the related meta data 
        including keys, their corresponding value lengths, 
        overall file size etc
        """

        # Setup filenames, meta filenames, and chunk idx
        self.setup(filename)

        if mode == 'w': 
            # Keep data_.keys(), and keys_ consistent
            self.data_ = AttrDict()
            self.keys_ = set()
            for field in fields: 
                self.data_[field] = []
                self.keys_.add(field)
            self.meta_file_ = AttrDict(chunks=[], keys=[])
            self.batch_size_ = batch_size
            self.lengths_ = defaultdict(lambda: 0)
        elif mode == 'r': 
            # Load first chunk, and keep keys_ consistent
            self.meta_file_ = AttrDict.load(self.meta_filename_)
            self.keys_ = self.meta_file_.keynames
            self.lengths_ = {}
            for key in self.keys_: 
                l = 0
                for chunk in self.meta_file_.chunks: l += chunk[key]
                self.lengths_[key] = l
            print 'IterDB::[LOADED], keys: {}, metafiles: {}'.format(self.keys_, len(self.meta_file_))
        
            # For the time-being, dynamically disattach append, extend functionality

            self.append, self.extend, self.add_fields = None, None, None
            self.flush, self.save = None, None

        else: 
            raise RuntimeError('Unknown mode %s' % mode)

    def setup(self, filename): 
        self.chunk_idx_ = 0
        self.folder_, self.filename_ = os.path.split(filename)
        self.meta_filename_ = os.path.join(self.folder_, self.filename_, 'meta_%s' % self.filename_)
        self.get_chunk_filename = lambda idx: os.path.join(self.folder_, self.filename_, 'chunks', 'chunk_%04i_%s' % (idx, self.filename_))

    # def __del__(self): 
    #     if len(self.data_): 
    #         print 'Seems like you have data stored away, but has not been flushed!'
    #         self.flush()

    def __getitem__(self, key): 
        return self.meta_file_[key]

    def __setitem__(self, key, value): 
        self.meta_file_[key] = value

    def add_fields(self, fields): 
        for field in fields: 
            if field in self.data_: 
                raise RuntimeError('Field %s is already in data, cannot update/replace fields!' % field)
            self.data_[field] = []
            self.keys_.add(field)

    def append(self, key, item): 
        if len(self.data_[key]) >= self.batch_size_: 
            self.flush()
        self.data_[key].append(item)
        # print [(k, len(v)) for k,v in self.data_.iteritems()]

    def extend(self, key, items): 
        self.data_[key].extend(item)

    def length(self, key): 
        return self.lengths_[key]

    def itervalues(self, key, inds=None, verbose=False): 
        if key not in self.keys_: 
            raise RuntimeError('Key %s not found in dataset. keys: %s' % (key, self.keys_))

        idx, ii = 0, 0
        total_chunks = len(self.meta_file_.chunks)
        inds = np.sort(inds) if inds is not None else None

        for chunk_idx, chunk in enumerate(progressbar(self.meta_file_.chunks, size=total_chunks, verbose=verbose)): 
            data = AttrDict.load(self.get_chunk_filename(chunk_idx))
            if inds is None: 
                for item in data[key]: 
                    yield item
            else:
                for i, item in enumerate(data[key]): 
                    if inds[ii] == idx + i: 
                        yield item
                        ii += 1
                        if ii >= len(inds): break
                idx += len(data[key])

    def keys(self): 
        return self.keys_

    def iter_keys_values(self, keys, inds=None, verbose=False): 
        for key in keys: 
            if key not in self.keys_: 
                raise RuntimeError('Key %s not found in dataset. keys: %s' % (key, self.keys_))

        idx, ii = 0, 0
        total_chunks = len(self.meta_file_.chunks)
        inds = np.sort(inds) if inds is not None else None

        for chunk_idx, chunk in enumerate(progressbar(self.meta_file_.chunks, size=total_chunks, verbose=verbose)): 
            data = AttrDict.load(self.get_chunk_filename(chunk_idx))
            
            # if inds is None: 
            items = (data[key] for key in keys)
            for item in izip(*items): 
                yield item
            # else:
            #     for i, item in enumerate(data[key]): 
            #         if inds[ii] == idx + i: 
            #             yield item
            #             ii += 1
            #             if ii >= len(inds): break
            #     idx += len(data[key])
        

    def iterchunks(self, key, batch_size=10, verbose=False): 
        if key not in self.keys_: 
            raise RuntimeError('Key %s not found in dataset. keys: %s' % (key, self.keys_))

        idx, ii = 0, 0
        total_chunks = len(self.meta_file_.chunks)
        batch_chunks = grouper(range(len(self.meta_file_.chunks)), batch_size)

        for chunk_group in progressbar(batch_chunks, size=total_chunks / batch_size, verbose=verbose): 
            items = []
            # print key, chunk_group
            for chunk_idx in chunk_group: 
                # grouper will fill chunks with default none values
                if chunk_idx is None: continue
                # Load chunk
                data = AttrDict.load(self.get_chunk_filename(chunk_idx))
                for item in data[key]: 
                    items.append(item)
            yield items
 
    def iterchunks_keys(self, keys, batch_size=10, verbose=False): 
        """
        Iterate in chunks and izip specific keys
        """
        for key in keys: 
            if key not in self.keys_: 
                raise RuntimeError('Key %s not found in dataset. keys: %s' % (key, self.keys_))
            
        iterables = [self.iterchunks(key, batch_size=batch_size, verbose=verbose) for key in keys]
        return izip(*iterables)

    def flush(self): 
        """
        Save dictionary with metadata, move to new chunk, 
        and prepare it for writing.       
        """
        self.meta_file_.keynames = list(self.data_.keys())
        self.meta_file_.chunks.append(AttrDict((k, len(v)) for k,v in self.data_.iteritems()))
        self.meta_file_.save(self.meta_filename_)
        for k,v in self.data_.iteritems(): 
            self.lengths_[k] += len(v)

        self.data_.save(self.get_chunk_filename(self.chunk_idx_))
        self._next_chunk_to_write()
        print 'Flushing', self.meta_file_.chunks[-1]

    def finalize(self): 
        self.flush()

    def _next_chunk_to_write(self): 
        for k, v in self.data_.iteritems(): 
            self.data_[k] = []
        self.chunk_idx_ += 1

class AttrDictDB(object):
    # Set up tables first and then flush
    def __init__(self, filename='', data=AttrDict(), mode='r', 
                 force=False, recursive=True, maxlen=100, ext='.h5'):
        self.log = logging.getLogger(self.__class__.__name__)

        # Set up output file
        if ext in filename: 
            output_filename = "%s" % filename
        else: 
            output_filename = "%s%s" % (filename, ext)
        self.log.info('DictDB: Opening file %s' % output_filename)

        # Open the db, no append support just yet
        self.h5f = tb.open_file(output_filename, mode=mode, title='%s' % filename)

        # Init the dict
        self.data = data
        self.tables = AttrDict()
        self.groups = AttrDict()
        self.groups._gp = self.h5f.root
        self.maxlen = maxlen

        # Read the db based on the mode
        if mode == 'r' or mode == 'a': 
            self.log.info('Reading DB to DictDB')
            self.data = self.read(group=self.h5f.root)

            self.__getattr__ = self.data.__getitem__
            self.__getitem__ = self.data.__getitem__

        self.__setattr__ = self.data.__setitem__
        self.__setitem__ = self.data.__setitem__

    def __del__(self): 
        self.h5f.close()

    def close(self): 
        try: 
            self.h5f.flush()
            self.h5f.close()
        except tb.exceptions.ClosedFileError as e: 
            print 'AttrDictDB already closed'

    def read(self, group=None): 
        if group is None:  group = self.h5f.root

        data = AttrDict()
        for child in self.h5f.list_nodes(group): 
            item = None
            try: 
                if isinstance(child, tb.group.Group): 
                    item = self.read(child)
                else: 
                    item = child.read()
                    if isinstance(item, str) and item.startswith('OBJ_'): 
                        item = cPickle.loads(item[4:])
                data[child._v_name] = item
            except tb.NoSuchNodeError:
                warnings.warn('No such node: "%s", skipping...' %repr(child))
                pass
        return data
                

    def get_node(self, g, k): 
        if g._v_pathname.endswith('/'):
            return ''.join([g._v_pathname,k])
        else: 
            return ''.join([g._v_pathname,'/',k])

    def flush(self, data=None, group=None, table=None, force=True): 
        if data is None: data = self.data
        if table is None: table = self.tables
        if group is None: group = self.groups
            
        # print 'Keys: ', data.keys()
        for k,v in data.iteritems(): 
            # print 'key,val', k,v, type(v)
            
            if not isinstance(k, str): 
                self.log.debug('Cannot save to DB, key is not string %s ' % k)
                continue

            # Clean up before writing 
            if force: 
                try:
                    self.h5f.remove_node(self.get_node(group._gp,k), recursive=True) 
                except tb.NoSuchNodeError:
                    pass

            # print 'In Group: ', group, k, v                
            if isinstance(v, dict):
                self.log.debug('Attempting to save dict type')
                # assert(k not in table);
                table[k] = AttrDict()
                group[k] = AttrDict();
                group[k]._gp = self.h5f.create_group(group._gp, k)
                self.h5f.flush()
                self.log.debug('Out Group: %s' % group[k])
                self.flush(data=v, group=group[k], table=table[k])
            elif isinstance(v, np.ndarray): 
                self.log.debug('Attempting to save ndarray %s' % type(v))
                table[k] = self.h5f.create_array(group._gp, k, v)
                self.log.debug('Out Table: %s' % table[k])
            # elif isinstance(v,io_utils.TableWriter):
            #     self.log.debug('Attempting to save with custom writer')
            #     table[k] = self.h5f.createTable(group._gp, name=k, 
            #                                     description=v.description, 
            #                                     title='%s-data' % (k) )
            #     v.write(table[k])
            #     # print 'Creating table with group:%s name:%s desc:%s' % (group._gp, k, writer.description)
            #     # print 'Out Table: ', table[k]
            else: 
                self.log.debug('Attempting to save arbitrary type %s' % type(v))
                try: 
                    assert v is not None
                    table[k] = self.h5f.create_carray(group._gp, k, obj=v)
                except (TypeError, ValueError, AssertionError): 
                    v = 'OBJ_' + cPickle.dumps(v, -1)
                    table[k] = self.h5f.create_array(group._gp, k, v)
                    # print 'TypeError', v
                finally: 
                    self.h5f.flush()
        return 

if __name__ == "__main__": 
    print '\nTesting AttrDict()'
    a = AttrDict()
    a.a = 1
    a.b = 'string'
    a.c = np.arange(10)
    a.d = np.arange(36).reshape((6,6))
    a['test'] = 'test'

    a.e = AttrDict()
    a.e.a = 2
    a.e.b= 'string2'
    a.e.c= np.arange(10)
    print a

    print '\nTesting DictDB() write'
    db = AttrDictDB(filename='test.h5', mode='w')
    db.data.a = 1
    db.data.b = 'string'
    db.data.c = np.arange(10)

    db.data.e = AttrDict()
    db.data.e.a = 2
    db.data.e.b = 'string2'
    db.data.e.c = np.arange(10)
    db.data.e.d = ('this','hat')
    # db.data.e.f = io_utils.Feature3DWriter(data=[])
    print 'Write DB: ', db.data
    wkeys = db.data.keys()
    db.flush()
    db.close()
    print 'OK'

    print '\nTesting DictDB() read'
    rdb = AttrDictDB(filename='test', mode='r')
    print 'Read DB: ', rdb.data
    rkeys = rdb.data.keys()
    print 'Rkeys: ', rkeys
    print 'Wkeys: ', wkeys
    print 'Diff: ', set(rkeys).difference(set(wkeys))
    rdb.close()
    print 'OK'

    print '\nTesting DictDB() read and then write'
    rwdb = AttrDictDB(filename='test', mode='a')
    print 'Read DB: ', rwdb.data
    print 'Keys: ', rwdb.data.keys()
    add = AttrDict()
    add.desc = np.ones((64,2), dtype=np.float32)
    add.desc2 = np.eye(4)
    add.string = 'string'
    add.tup = ('this', 'that')
    rwdb.flush(data=add)
    rwdb.close()


    print '\nTesting DictDB() re-read'
    rdb = AttrDictDB(filename='test', mode='r')
    # print 'Read DB: ', rdb.data
    rkeys = rdb.data.keys()
    print 'keys: ', rkeys
    rdb.close()
    print 'OK'

