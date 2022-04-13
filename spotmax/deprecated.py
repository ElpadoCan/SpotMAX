def emit_shifting_signal(signals, shift, axis):
    if signals is not None:
        text = (
            f'Load chunk: shifting window by {shift} along axis {axis}'
        )
        signals.progress.emit(text, 'INFO')

def emit_loading_chunk_signal(
        signals, t0, t1, z0=None, z1=None
    ):
    if signals is not None:
        text = (
            'Load chunk: loading chunk with indexes '
            f'[t=({t0, t1}), z=({z0, z1})]'
        )
        signals.progress.emit(text, 'INFO')

def emit_indexing_chunk_signal(signals, t0, t1, z0=None, z1=None):
    if signals is not None:
        text = (
            'Load chunk: indexing chunk into window at position '
            f'[t=({t0,t1}), z=({z0,z1})]'
        )
        signals.progress.emit(text, 'INFO')


def shift4Dwindow(
        dset, window_arr, current_t0_window, current_z0_window,
        windowSizeT, windowSizeZ, chunkSizeT, chunkSizeZ,
        direction, signals=None
    ):
    """Function used to index an array that is like a moving window along
    a bigger dataset. Useful when dset is a h5py dataset and we need to keep
    memory usage as low as possible. The memory footprint is maximum twice
    the memory required by the window_arr.

    How it works: roll the window array by the required amount (chunk size)
    and replace the ends with a chunk indexed from dset. If the requested chunk
    is outside of the dset shape, the original window array is returned.

    Parameters
    ----------
    dset : array-like
        NumPy array or h5py dataset.
    window_arr : numpy.ndarray
        The moving window array.
    current_t0_window : int
        Starting index of the window_arr time axis in dset (absolute).
    current_z0_window : int
        Starting index of the window_arr z-axis in dset (absolute).
    windowSizeT : int
        Size of the window in the time-axis.
    windowSizeZ : int
        Size of the window in the z-axis.
    chunkSizeT : int
        Size of the chunk in the time-axis
    chunkSizeZ : int
        Size of the chunk in the z-axis.
    direction : str
        Either 'forward' or 'backward'. Direction of the moving window

    Returns
    -------
    tuple: (numpy.ndarray, int, int)
        Moved array window plus new start indexes.

    """
    SizeZ = dset.shape[-3]
    SizeT = dset.shape[-4]

    current_t1_window = current_t0_window + windowSizeT
    current_z1_window = current_z0_window + windowSizeZ
    if direction == 'forward':
        t0_chunk = current_t1_window
        z0_chunk = current_z1_window

        new_t0_window = current_t0_window+chunkSizeT
        new_z0_window = current_z0_window+chunkSizeZ

        new_t1_window = current_t1_window+chunkSizeT
        new_z1_window = current_z1_window+chunkSizeZ

        if new_t1_window<=SizeT and new_z1_window<=SizeZ:
            # Move window both in z and t
            shift = (-chunkSizeT, -chunkSizeZ)
            axis = (0, 1)
        elif new_t1_window>SizeT and new_z1_window<=SizeZ:
            # Move window only in z
            shift = -chunkSizeZ
            chunkSizeT = 0
            axis = 1
            new_t1_window = SizeT
            t0_chunk = SizeT
        elif new_t1_window<=SizeT and new_z1_window>SizeZ:
            # Move window only in t
            shift = -chunkSizeT
            chunkSizeZ = 0
            axis = 0
            new_z1_window = SizeZ
            z0_chunk = SizeZ
        else:
            new_t0_window = current_t0_window
            new_z0_window = current_z0_window
            return window_arr, new_t0_window, new_z0_window


        emit_shifting_signal(signals, shift, axis)
        window_arr = np.roll(window_arr, shift, axis=axis)

        emit_loading_chunk_signal(
            signals, t0_chunk, new_t1_window, z0_chunk, new_z1_window
        )
        chunk = dset[t0_chunk:new_t1_window, z0_chunk:new_z1_window]

        emit_indexing_chunk_signal(
            signals, -chunkSizeT, chunk.shape[0],
            z0=-chunkSizeZ, z1=chunk.shape[1]
        )
        window_arr[-chunkSizeT:chunk.shape[0], -chunkSizeZ:chunk.shape[1]] = chunk
    else:
        t1_chunk = current_t0_window
        z1_chunk = current_z0_window

        new_t0_window = current_t0_window-chunkSizeT
        new_z0_window = current_z0_window-chunkSizeZ

        new_t1_window = current_t1_window-chunkSizeT
        new_z1_window = current_z1_window-chunkSizeZ

        if new_t0_window>=0 and new_z0_window>=0:
            # Move window both in z and t
            shift = (chunkSizeT, chunkSizeZ)
            axis = (0, 1)
        elif new_t0_window<0 and new_z0_window>=0:
            # Move window only in z
            shift = chunkSizeZ
            chunkSizeT = 0
            axis = 1
            new_t0_window = 0
        elif new_t0_window>=0 and new_z0_window<0:
            # Move window only in t
            shift = chunkSizeT
            chunkSizeZ = 0
            axis = 0
            new_z0_window = 0
        else:
            # No moving required
            new_t0_window = current_t0_window
            new_z0_window = current_z0_window
            return window_arr, new_t0_window, new_z0_window

        emit_shifting_signal(signals, shift, axis)
        window_arr = np.roll(window_arr, shift, axis=axis)

        emit_loading_chunk_signal(
            signals, new_t0_window, t1_chunk, new_z0_window, z1_chunk
        )
        chunk = dset[new_t0_window:t1_chunk, new_z0_window:z1_chunk]

        emit_indexing_chunk_signal(signals, 0, chunkSizeT, z0=0, z1=chunkSizeZ)
        window_arr[:chunkSizeT, :chunkSizeZ] = chunk

    return window_arr, new_t0_window, new_z0_window

def shift3Dwindow(
        dset, window_arr, current_t0_window,
        windowSizeT, chunkSizeT, direction, signals=None
    ):
    """See shift4Dwindow for more details"""

    SizeT = dset.shape[-3]

    current_t1_window = current_t0_window + windowSizeT
    if direction == 'forward':
        t0_chunk = current_t1_window

        new_t0_window = current_t0_window+chunkSizeT
        new_t1_window = current_t1_window+chunkSizeT

        if new_t1_window<=SizeT:
            # Move window in t
            shift = -chunkSizeT
            axis = 0
        else:
            new_t0_window = current_t0_window
            return window_arr, new_t0_window

        emit_shifting_signal(signals, shift, axis)
        window_arr = np.roll(window_arr, shift, axis=axis)

        emit_loading_chunk_signal(signals, t0_chunk, new_t1_window)
        chunk = dset[t0_chunk:new_t1_window]

        emit_indexing_chunk_signal(signals, -chunkSizeT, chunk.shape[0])
        window_arr[-chunkSizeT:chunk.shape[0]] = chunk
    else:
        t1_chunk = current_t0_window
        new_t0_window = current_t0_window-chunkSizeT
        new_t1_window = current_t1_window-chunkSizeT

        if new_t0_window>=0:
            # Move window both in z and t
            shift = chunkSizeT
            axis = 0
        else:
            # No moving required
            new_t0_window = current_t0_window
            return window_arr, new_t0_window

        emit_shifting_signal(signals, shift, axis)
        window_arr = np.roll(window_arr, shift, axis=axis)

        emit_loading_chunk_signal(signals, new_t0_window, t1_chunk)
        chunk = dset[new_t0_window:t1_chunk]

        emit_indexing_chunk_signal(signals, 0, chunkSizeT)
        window_arr[:chunkSizeT] = chunk
    return window_arr, new_t0_window
