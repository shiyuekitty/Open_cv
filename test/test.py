import requests
import re
import os
import random
import time
from fake_useragent import UserAgent

import winreg
def get_desktop():
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                          r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders',)
    return winreg.QueryValueEx(key, "Desktop")[0]


# 存储得位置，桌面上创建一个下载文件
saveImagPath = get_desktop() +"\\" + "下载文件"
os.path.exists(saveImagPath)
if not os.path.exists(saveImagPath):
    os.mkdir(saveImagPath)

saveImagFile = saveImagPath

ua=UserAgent()


for j in range(1,237):
    t = 'http://www.mangareader.net/doraemon'+ '/'+ str(j)
    headers = {'User-Agent': ua.random}
    headers.update({'Cookie': '__cfduid=dffbec8ee6e5ebd0d1d60a22bbc10c7d01587445599; BB_plg=pm; _ga=GA1.2.2106140754.1587445603; _gid=GA1.2.456110379.1587445603; bbl=5; fjulakwal=1'})
    headers.update({'Host': 'www.mangareader.net'})
    headers.update({'Connection': 'keep-alive'})
    headers.update({'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'})
    headers.update({'Accept-Encoding': 'gzip, deflate'})
    headers.update({'Accept-Language': 'zh-CN,zh;q=0.9'})
    headers.update({'Upgrade-Insecure-Requests': '1'})
    response = requests.get(t, headers=headers)
    saveimagLog = get_desktop() + "\\" + "下载文件" + "\\Log{}.txt".format(j)
    folog = open(saveimagLog, "w")
    if response.status_code ==200:
        html = response.text
        pag = re.findall('</select> of (.*?)</div>', html)
        a = int(pag[0])
        log = ('第{}章节有{}个图片\r'.format(j,a))
        print(log)
        folog.write(log)

        for i in range(1, int(pag[0])+1):

            time.sleep(4)
            I = t + '/' + str(i)
            # 获得每一页的地址

            response = requests.get(I, headers=headers)
            if response.status_code == 200:
                html = response.text

                # 获取每个图片的地址
                urls = re.findall('<img id=".*?" width=".*?" height=".*?" src="(.*?)" alt=".*?" />', html)
                # 获取图片和名称
                for url in urls:
                    print(url)

                    name = url.split('-')[-1]

                    headImagUrl = {}
                    headImagUrl.update({'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'})
                    headImagUrl.update({'Host': 'i5.imggur.net'})
                    headImagUrl.update({'If-Modified-Since': 'Sat, 30 Oct 2010 22:44:22 GMT'})
                    headImagUrl.update({'If-None-Match': "4ccc9fc6-1b01c"})
                    headImagUrl.update({'Upgrade-Insecure-Requests': '1'})
                    headImagUrl.update({'User-Agent':ua.random})



                    response = requests.get(url, headers=headImagUrl)
                # 将图片写入文件夹
                    if response.status_code == 200:
                        information = '第 {} 章节中第 {} 个图片得地址是 {} ,文件写入成功'.format(j, i, url)
                        print(information)
                        folog.write(information + "\r")

                        saveImagFile = saveImagPath +"\\" + str(j) +"_" + name
                        with open(saveImagFile, 'wb') as f:
                            f.write(response.content)
                    else:
                        information = '第{}章节中第{}个图片没有读到，错误信息{},文件写入失败，请手动下载,图片地址是{}'.format(j, i, response.status_code, url)
                        print(information)
                        folog.write(information + "\r")
            else:
                information = '第{}章节中第{}个图片没有读到，错误信息{},文件写入失败，请手动下载'.format(j, i, response.status_code)
                print(information)
                folog.write(information + "\r")

    else:
        log ='第{}章访问出错，请自己手动访问，或者重新执行程序'.format(j)
        folog.write(log)

    folog.close()