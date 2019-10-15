#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/15 19:21
# @Author  : PanYunSong
# @File    : docx_to_pdf.py

cmd = 'libreoffice  --convert-to pdf  %s --outdir  %s ' % (filename, dest_dir)
