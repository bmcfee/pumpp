import hashlib
import h5py


def save_h5(filename, **kwargs):
    '''Save data to an hdf5 file.
    Parameters
    ----------
    filename : str
        Path to the file
    kwargs
        key-value pairs of data
    See Also
    --------
    load_h5
    '''
    with h5py.File(filename, 'w') as hf:
        hf.update(kwargs)


def load_h5(filename, data=None, fields=None):
    '''Load data from an hdf5 file created by `save_h5`.
    Parameters
    ----------
    filename : str
        Path to the hdf5 file
    data : dict, optional
        A starting dict. Keys in data will be skipped.
    fields : set/tuple/list, optional
        If present, only load data from these fields.
    Returns
    -------
    data : dict
        The key-value data stored in `filename`
    See Also
    --------
    save_h5
    '''
    data = {} if data is None else data

    def collect(k, v):
        if (not fields or k in fields) and k not in data and isinstance(v, h5py.Dataset):
            data[k] = v[()]

    with h5py.File(filename, mode='r') as hf:
        hf.visititems(collect)

    return data

def get_cache_id(key):
    '''Create a unique ID to cache Pump output data.'''
    return hashlib.md5(key.encode('utf-8')).hexdigest()
