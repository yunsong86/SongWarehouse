#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/15 19:21
# @Author  : song
# @File    : doc_to_docx.py

import os

for filename in filenames:
    cmd = 'libreoffice  --convert-to docx  %s --outdir  %s ' % (filename, dest_dir)
    os.system(cmd)
