from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("typeconvert")
except PackageNotFoundError:
    # package is not installed
    pass
