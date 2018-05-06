# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import os
import codecs
import re
import glob
import commands

reload(sys)
sys.setdefaultencoding('utf-8')

dirname = "./json/init100"

fh_u = codecs.open('src.txt', 'w', 'utf-8')
fh_s = codecs.open('tgt.txt', 'w', 'utf-8')

filelist = glob.glob(dirname + "/*.json")
for filename in filelist:
    outputs = commands.getoutput(("python show_dial_fix.py " + filename))
    lines = outputs.split("\n")

    has_system_uttr = False
    system_uttr = ''
    has_user_uttr = False
    user_uttr = ''
    for line in lines:
        if line.startswith("S:") and has_user_uttr:
            has_system_uttr = True
            system_uttr = re.sub(r'^S:', '', line)
        if line.startswith("U:"):
            has_user_uttr = True
            user_uttr = re.sub(r'^U:', '', line)
        if has_system_uttr and has_user_uttr:
            fh_u.write(user_uttr + "\n")
            fh_s.write(system_uttr + "\n")
            has_system_uttr = False
            has_user_uttr = False
fh_u.close()
fh_s.close()
### EOF
