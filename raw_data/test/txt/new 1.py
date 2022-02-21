'MVI-1' : re.compile('[^x](M\d级*)'), 
'MVI-3' : re.compile('([查未可]*见)[^肉]?[^眼]?脉管内*癌栓[^\(（]'),
'MVI-4' : re.compile('([查未可]*见).?.?微血管内*癌栓'),
'MVI-5' : re.compile('([查未可]*见).?.?微血管内*侵犯'),
    
'TNM-1' : re.compile(r'(\w*?T\({0,1}\w\){0,1}\w*?M[x\d])'), #