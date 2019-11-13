#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


import textract


def walkFile(file):
    for root, dirs, files in os.walk(file):

        # 遍历文件
        for f in files:
            yield os.path.join(root, f)




from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.layout import *
import re


def pdf2txt(fn):
    #打开一个pdf文件
    # fn = r'C:\Users\1TSH4W2\Desktop/有三AI+视觉算法工程师成长指导手册_20190812_192257.pdf'
    fp = open(fn, 'rb')
    #创建一个PDF文档解析器对象
    parser = PDFParser(fp)
    #创建一个PDF文档对象存储文档结构
    #提供密码初始化，没有就不用传该参数
    #document = PDFDocument(parser, password)
    document = PDFDocument(parser)
    #检查文件是否允许文本提取
    if not document.is_extractable:
        raise PDFTextExtractionNotAllowed
    #创建一个PDF资源管理器对象来存储共享资源
    #caching = False不缓存
    rsrcmgr = PDFResourceManager(caching = False)
    # 创建一个PDF设备对象
    laparams = LAParams()
    # 创建一个PDF页面聚合对象
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    #创建一个PDF解析器对象
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    #处理文档当中的每个页面

    # doc.get_pages() 获取page列表
    #for i, page in enumerate(document.get_pages()):
    #PDFPage.create_pages(document) 获取page列表的另一种方式
    replace=re.compile(r'\s+')
    # 循环遍历列表，每次处理一个page的内容
    for page in PDFPage.create_pages(document):
        interpreter.process_page(page)
        # 接受该页面的LTPage对象
        layout=device.get_result()
        # 这里layout是一个LTPage对象 里面存放着 这个page解析出的各种对象
        # 一般包括LTTextBox, LTFigure, LTImage, LTTextBoxHorizontal 等等
        for x in layout:
            #如果x是水平文本对象的话
            if(isinstance(x,LTTextBoxHorizontal)):
                text=re.sub(replace,'',x.get_text())
                if len(text)!=0:
                    print(text)



data_dir =None

for name in walkFile(data_dir):
    try:
        # text = textract.process(name)
        # text = text.decode('utf-8')
        pdf2txt(name)
        # print(text)
    except Exception as e:
        print(e)
