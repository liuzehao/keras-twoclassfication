#-*-coding:utf-8-*-
import sys


from PIL import ImageDraw,ImageFont
from PIL import Image
import random
import math, string

class RandomChar():

  @staticmethod
  def Unicode():
    val = random.randint(0x4E00, 0x9FBF)
    return unichr(val)

  @staticmethod
  def GB2312():
    head = random.randint(0xB0, 0xCF)
    body = random.randint(0xA, 0xF)
    tail = random.randint(0, 0xF)
    val = ( head << 8 ) | (body << 4) | tail
    str = "%x" % val
    return str.decode('hex').decode('gb2312')

class ImageChar():
  def __init__(self, fontColor = (0, 0, 0),
                     size = (100, 40),
                     fontPath = 'simsun.ttc',
                     bgColor = (255, 255, 255),
                     fontSize = 10):
    self.size = size
    self.fontPath = fontPath
    self.bgColor = bgColor
    self.fontSize = fontSize
    self.fontColor = fontColor
    self.font = ImageFont.truetype(self.fontPath, self.fontSize)
    self.image = Image.new('RGB', size, bgColor)

  def drawText(self, pos, txt, fill):
    draw = ImageDraw.Draw(self.image)
    draw.text(pos, txt, font=self.font, fill=fill)
    del draw

  def randChinese(self, num):
    gap = 2
    start = 0
    flag_isd=random.randint(0, 1)
    line=0
    for i in range(0, num):
      x = start + self.fontSize * i +gap * i
      self.drawText((x,3), RandomChar().GB2312(), (0,0,0))
    if flag_isd==1:
        line=1
        size_y=random.randint(17,25)
        for i in range(1,random.randint(2,8)):
            x = start + self.fontSize * (i-1) +gap * (i-1)
            self.drawText((x,size_y), RandomChar().GB2312(), (0,0,0))
    return line

  def save(self, path):
    self.image.save(path,'jpeg')
if __name__=='__main__':
    n=100000
    i=0
    outpath="./testdata/"
    out_label_path="./label_test.txt"
    file = open(out_label_path, 'w')
    while i<n:
        try:
            ic = ImageChar(fontColor=(100,211,90))
            line=ic.randChinese(8)
            jpg_name_num= '%05d' % int(i)
            jpg_name=outpath+jpg_name_num+'.jpeg'    
        except Exception:
            continue
        print(jpg_name)
        file.write(str(line) + "\n")
        ic.save(jpg_name)
        i=i+1
    file.close()
