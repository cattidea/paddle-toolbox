import paddle

from paddle.version import major, minor, patch

if patch.isdigit():
    PADDLE_VERSION_TUPLE = (int(major), int(minor), int(patch))
else:
    PADDLE_VERSION_TUPLE = (int(major), int(minor), int(patch.split("-")[0]), patch.split("-")[-1])
