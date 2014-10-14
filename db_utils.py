#!/usr/bin/python
import time, logging, cPickle, shelve
import tables as tb
import numpy as np

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# class AttrDictDB(dict): 
#     def __init__(self, *args, **kwargs): 
#         super(AttrDict, self).__init__(*args, **kwargs)
#         self.__dict__ = self

# class AttrDictDB(object): 
#     def __init__(self, *args, **kwargs): 
#         self = shelve.open(*args, **kwargs)

#         self.__getattr__ = self.__getitem__
#         self.__setattr__ = self.__setitem__


# if __name__ == "__main__": 
#     print '\nTesting AttrDict()'
#     a = AttrDictDB('test.db', flag='c')
#     a.a = 1
#     a.b = 'string'
#     a.c = np.arange(10)
#     a.d = np.arange(36).reshape((6,6))

#     print dir(a)
#     a['test'] = 'test'
#     a.close()

# class AttrDB(shelve.DbfilenameShelf): 
#     def __init__(self, *args, **kwargs): 
#         shelve.DbfilenameShelf.__init__(self, *args, **kwargs)
#         self.__getattr__ = self.__getitem__
#         self.__setattr__ = self.__setitem__        

# def persistent_db(*args, **kwargs): 
#     import shelve
#     d = shelve.open(*args, **kwargs)
#     d.__getattr__ = d.__getitem__
#     d.__setattr__ = d.__setitem__
#     return d

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
        self.h5f = tb.openFile(output_filename, mode=mode, title='%s' % filename)

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
        for child in self.h5f.listNodes(group): 
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
                    self.h5f.removeNode(self.get_node(group._gp,k), recursive=True) 
                except tb.NoSuchNodeError:
                    pass

            # print 'In Group: ', group, k, v                
            if isinstance(v, dict):
                self.log.debug('Attempting to save dict type')
                # assert(k not in table);
                table[k] = AttrDict()
                group[k] = AttrDict();
                group[k]._gp = self.h5f.createGroup(group._gp, k)
                self.h5f.flush()
                self.log.debug('Out Group: %s' % group[k])
                self.flush(data=v, group=group[k], table=table[k])
            elif isinstance(v, np.ndarray): 
                self.log.debug('Attempting to save ndarray %s' % type(v))
                table[k] = self.h5f.createArray(group._gp, k, v)
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
                    table[k] = self.h5f.createCArray(group._gp, k, obj=v)
                except (TypeError, ValueError, AssertionError): 
                    v = 'OBJ_' + cPickle.dumps(v, -1)
                    table[k] = self.h5f.createArray(group._gp, k, v)
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

