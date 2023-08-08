#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def default_params():
    data = DotDict()
    model = DotDict()
    train = DotDict()

    """
    ----------------------------------------------------------------
    DATA
    ----------------------------------------------------------------
    """
    #data.freqs = [9, 7]
    #data.number_of_beats = 4
    #data.minimal_number_of_pred = 4
    data.data_type = 0 # 0 for pulse, 1 for cosine
    data.min_length = 200
    data.max_length = 300
    data.batch_size = 5
    data.min_beats = 3
    data.max_beats = 6
    data.freq_type = 1 # 0 for max min, 1 for specific set randomly sampled
    data.jitter_type = 1
    data.min_freq = 9
    data.max_freq = 10
    data.freqs = [5,6,7,8,9,11,12,13,14,15]
    data.offset = 1 # offset the peaks or not?

    """
    ----------------------------------------------------------------
    MODEL
    ----------------------------------------------------------------
    """
    model.h_size = 200
    model.hidden_act = 'tanh'
    model.hidden_init_learn = True
    model.hidden_init_std = 1
    model.transition_init = 'orthogonal'
    model.output_act = 'sigmoid'
    model.i_size = 1
    model.t_size = 1
    model.linear_std = 1
    model.batch_size = data.batch_size
    model.num_inits = len(data.freqs)

    """
    ----------------------------------------------------------------
    TRAINING
    ----------------------------------------------------------------
    """
    train.learning_rate = 5e-5
    train.weight_decay = 0
    train.train_iters = 1000000
    train.act_weight = 0

    return DotDict({'data': data,
                    'model': model,
                    'train': train})


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        # We trust the dict to init itself better than we can.
        dict.__init__(self, *args, **kwargs)
        # Because of that, we do duplicate work, but it's worth it.
        for k, v in self.items():
            self.__setitem__(k, v)

    def __getattr__(self, k):
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            # Maintain consistent syntactical behaviour.
            raise AttributeError(
                "'DotDict' object has no attribute '" + str(k) + "'"
            )

    def __setitem__(self, k, v):
        dict.__setitem__(self, k, DotDict.__convert(v))

    __setattr__ = __setitem__

    def __delattr__(self, k):
        try:
            dict.__delitem__(self, k)
        except KeyError:
            raise AttributeError(
                "'DotDict' object has no attribute '" + str(k) + "'"
            )

    @staticmethod
    def __convert(o):
        """
        Recursively convert `dict` objects in `dict`, `list`, `set`, and
        `tuple` objects to `attrdict` objects.
        """
        if isinstance(o, dict):
            o = DotDict(o)
        elif isinstance(o, list):
            o = list(DotDict.__convert(v) for v in o)
        elif isinstance(o, set):
            o = set(DotDict.__convert(v) for v in o)
        elif isinstance(o, tuple):
            o = tuple(DotDict.__convert(v) for v in o)
        return o

    @staticmethod
    def to_dict(data):
        """
        Recursively transforms a dotted dictionary into a dict
        """
        if isinstance(data, dict):
            data_new = {}
            for k, v in data.items():
                data_new[k] = DotDict.to_dict(v)
            return data_new
        elif isinstance(data, list):
            return [DotDict.to_dict(i) for i in data]
        elif isinstance(data, set):
            return [DotDict.to_dict(i) for i in data]
        elif isinstance(data, tuple):
            return [DotDict.to_dict(i) for i in data]
        else:
            return data
