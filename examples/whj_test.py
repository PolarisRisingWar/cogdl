#import cogdl
#print(cogdl.__file__)
#如果直接这样的话，就会输出：/home/wanghuijuan/env17/lib/python3.8/site-packages/cogdl/__init__.py
import sys
sys.path.insert(0,'whj_code2/cogdl_fork/cogdl')
print(sys.path)
import cogdl
print(cogdl.__file__)