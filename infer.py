"""Portable primary-method held-out inference; defaults to read-only checks."""
import sys

sys.dont_write_bytecode = True
from runtime.pipeline import infer_main

if __name__ == '__main__':
    infer_main()
