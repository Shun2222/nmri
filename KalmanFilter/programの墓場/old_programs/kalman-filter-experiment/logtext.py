import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile

class LogText():
    def __init__(self):
        self.logtext = """"""
        
    def add_text(self, text):
        self.logtext += text + '\n'

    def clear(self):
        self.logtext = """"""

    def output(self, fname):
        with open(fname, mode='w', encoding='utf-8') as f:
            f.write(self.logtext)


logtext = LogText()
def add_text(text):
    logtext.add_text(text)
def clear():
    logtext.clear()
def output(fname):
    logtext.output(fname)