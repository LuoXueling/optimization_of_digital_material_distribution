from datetime import datetime
from copy import deepcopy


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def copy(self):
        return dotdict(super().copy())


def log(*args, **kwargs):
    l = f"[{datetime.now().strftime('%m-%d-%Y,%H:%M:%S')}]"
    output_file = kwargs.get("output_file", None)
    _kwargs = kwargs.copy()
    if output_file is not None:
        del _kwargs["output_file"]
    if len(args) == 0:
        print(*args, **_kwargs)
    else:
        print(l, *args, **_kwargs)
    if output_file is not None:
        with open(output_file, "a") as file:
            _kwargs.update(dict(file=file))
            if len(args) == 0:
                print(*args, **_kwargs)
            else:
                print(l, *args, **_kwargs)
