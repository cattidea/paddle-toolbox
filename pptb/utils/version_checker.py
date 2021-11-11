import importlib
import warnings
from functools import wraps

import paddle
from setuptools._vendor.packaging import version

from pptb.exceptions import PaddleVersionError

PADDLE_VERSION = version.parse(paddle.__version__)


def minimum_required_version(version_str):
    def minimum_required_version_wrapper(func):
        @wraps(func)
        def func_with_version_checking(*args, **kwargs):
            if PADDLE_VERSION < version.parse(version_str):
                raise PaddleVersionError(f"该功能最低支持 paddlepaddle {version_str}，但当前版本为 {paddle.__version__}")
            return func(*args, **kwargs)

        return func_with_version_checking

    return minimum_required_version_wrapper


def feature_redirect(version_str, mod_name, func_name=None):
    def feature_redirect_wrapper(func):
        nonlocal func_name
        func_name = func_name if func_name is None else func.__name__

        @wraps(func)
        def func_with_feature_redirect(*args, **kwargs):
            if PADDLE_VERSION >= version.parse(version_str):
                warnings.warn(
                    f"本 API 已在 paddlepaddle 中被实现，已自动重定向至 {mod_name}.{func_name}，建议直接使用该 API", DeprecationWarning
                )
                redicted_func = getattr(importlib.import_module(mod_name), func_name)
                return redicted_func(*args, **kwargs)
            return func(*args, **kwargs)

        return func_with_feature_redirect

    return feature_redirect_wrapper


def assert_version_greater_equal(version_str):
    if PADDLE_VERSION < version.parse(version_str):
        raise PaddleVersionError(f"PaddlePaddle Toolbox 最低支持 paddlepaddle {version_str}，但当前版本为 {paddle.__version__}")
