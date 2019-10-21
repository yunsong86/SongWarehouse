#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/15 19:22
# @Author  : song
# @File    : pdf_to_image.py

from pdf2image import convert_from_path

convert_from_path(source_file, 300, "./", fmt="JPEG", output_file=dest_file, thread_count=1)
