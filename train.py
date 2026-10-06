"""Portable primary-method train entry; defaults to a read-only preflight."""
import sys

sys.dont_write_bytecode = True
from runtime.pipeline import train_main

if __name__ == '__main__':
    train_main()
