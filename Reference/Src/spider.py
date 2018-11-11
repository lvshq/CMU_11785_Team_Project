#-*-coding:UTF-8-*-

from bs4 import BeautifulSoup
import urllib
import urllib2
import requests
import re
import sys
import thread
import time
reload(sys)
sys.setdefaultencoding('UTF-8')

global picture_store_path
global coat_male_cnt
coat_male_cnt = 1
train_path = '/home/lvshq/Caffe-DeepBinaryCode-master/data/Clothes/train_imgs/'
test_path = '/home/lvshq/Caffe-DeepBinaryCode-master/data/Clothes/test_imgs/'

def getPictures(url, src, store_path, name, cnt):
	#html = urllib2.urlopen(url)
	data = requests.get(url).text
	#print html.read()
	#soup = BeautifulSoup(html.read().decode('utf-8'),"html.parser")
	#soup = BeautifulSoup(data)

	#htmlFile = open('/home/lvshq/src/TaobaoData/test.html','r')
	#local.write(data)
	#data = htmlFile.read()
	#htmlFile.close()
	if src == 1:  # From Taobao
		pattern = re.compile(r'pic_url\":\".*?\"')
		imgList = re.findall(pattern, data)

		# Eliminate the prefix pic_url":" and postfix  
		strinfo1 = re.compile('pic_url\":\"')
		strinfo2 = re.compile('\"')

		for imgUrl in imgList:
			#_imgUrl = imgUrl.encode('utf-8')  # Convert 'unicode' object to str object to use
			_imgUrl = strinfo1.sub('http:',imgUrl)
			_imgUrl = strinfo2.sub('',_imgUrl)
			length = len(_imgUrl)
			if (_imgUrl[length-1] != 'g' and _imgUrl[length-1] != '2'):
				continue
			cntStr = '%d' %cnt
			print cntStr + " : " + _imgUrl
			#urllib.urlretrieve(_imgUrl, picture_store_path + '/coat_male_%s.jpg' % coat_male_cnt)
			try:
				urllib.urlretrieve(_imgUrl, store_path + name + '_%s.jpg' % cnt)
			except Exception, e:
				print Exception, ":", e
				for t in range(0,20):
					try:
						urllib.urlretrieve(_imgUrl, store_path + name + '_%s.jpg' % cnt)
						time.sleep(0.5)
					except:
						continue
				continue
			cnt = cnt + 1

		print cnt



	else:  # From Jingdong
		pattern = re.compile(r'<img width="220".*?jpg')
		imgList = re.findall(pattern, data)
		# Eliminate the prefix pic_url":" and postfix  
		strinfo1 = re.compile('<img width="220".*?(src|data-lazy-img)=\".*?')
		cnt = 0
		for imgUrl in imgList:
			_imgUrl = strinfo1.sub('http:',imgUrl)

			cntStr = '%d' %coat_male_cnt
			#print cntStr + " : " + _imgUrl
			
			cnt = cnt + 1
			try:
				urllib.urlretrieve(_imgUrl, store_path + '/Down_Jacket_%s.jpg' % coat_male_cnt)
			except Exception, e:
				print Exception, ":", e
				for t in range(0,20):
					try:
						urllib.urlretrieve(_imgUrl, store_path + '/Down_Jacket_%s.jpg' % coat_male_cnt)
						time.sleep(0.5)
					except:
						continue
				continue
			#urllib.urlretrieve(_imgUrl, picture_store_path + '/Tshirt_male_%s.jpg' % coat_male_cnt)
			coat_male_cnt = coat_male_cnt + 1
		print cnt



def getImgs(start, end, prefix, postfix, store_path, name):
	for i in range(start,end):
		offset = 44 * i
		offset_JD = 1 + 2 * i
		temp_str = '%d' %offset  # Convert digit to string
		temp_str_JD = '%d' %offset_JD
		url_Taobao = prefix + postfix + temp_str
		#url_JD = url_coat_female_JD + temp_str_JD
		#print url_JD
		getPictures(url_Taobao, 1, store_path, name, offset)



if __name__ == '__main__':
	url_coat_male = "https://s.taobao.com/search?q=%E5%A4%96%E5%A5%97%E7%94%B7"
	url_coat_male_JD = "https://search.jd.com/Search?keyword=%E5%A4%96%E5%A5%97%E7%94%B7&enc=utf-8&page="
	url_coat_female = "https://s.taobao.com/search?q=%E5%A4%96%E5%A5%97%E5%A5%B3"
	url_coat_female_JD = "https://search.jd.com/Search?keyword=%E5%A4%96%E5%A5%97%E5%A5%B3&enc=utf-8&page="
	url_Tshirt_male = "https://s.taobao.com/search?q=t%E6%81%A4%E7%94%B7"
	url_Tshirt_female = "https://s.taobao.com/search?q=t%E6%81%A4%E5%A5%B3"
	url_Pants = "https://s.taobao.com/search?q=%E9%95%BF%E8%A3%A4"
	url_Down_Jacket = "https://s.taobao.com/search?q=%E7%BE%BD%E7%BB%92%E6%9C%8D"
	url_Sweater = "https://s.taobao.com/search?q=%E6%AF%9B%E8%A1%A3"
	url_Vest = "https://s.taobao.com/search?q=%E8%83%8C%E5%BF%83"
	url_Suit = "https://s.taobao.com/search?q=%E6%AD%A3%E8%A3%85"
	url_Dress = "https://s.taobao.com/search?q=%E8%A3%99%E5%AD%90"

	urls = (url_coat_male,
		url_coat_female,
		url_Tshirt_male,
		url_Tshirt_female,
		url_Pants,
		url_Down_Jacket,
		url_Sweater,
		url_Vest,
		url_Suit,
		url_Dress)

	postfix = "&bcoffset=1&ntoffset=1&p4ppushleft=1%2C48&s="
	#postfix_coat_female = "&bcoffset=1&ntoffset=1&p4ppushleft=1%2C48&s="

	names = ('Coat_Male', 
		'Coat_Female', 
		'Tshirt_Male', 
		'Tshirt_Female', 
		'Pants',
		'Down_Jacket', 
		'Sweater', 
		'Vest', 
		'Suit', 
		'Dress')

	try:
		i = 0;
		while i <= 1:
			thread.start_new_thread(getImgs, (44, 80, urls[i], postfix, train_path, names[i],))
			#getImgs(95, 100, urls[i], postfix, test_path, names[i])
			time.sleep(1)
			i += 1;
			# 	thread.start_new_thread(getImgs, (0, 80, url_coat_female, postfix, train_path, names[1],))
			# 	getImgs(81, 100, url_coat_female, postfix, test_path, names[1])
			# 	print "4 thread"
			# 	thread.start_new_thread(getImgs, (0, 80, url_Tshirt_male, postfix, train_path, names[2],))
			# 	getImgs(81, 100, url_Tshirt_male, postfix, test_path, names[2])
				
			# 	thread.start_new_thread(getImgs, (0, 80, url_Tshirt_female, postfix, train_path, names[3],))
			# 	getImgs(81, 100, url_Tshirt_female, postfix, test_path, names[3])
				
			# 	thread.start_new_thread(getImgs, (0, 80, url_Pants, postfix, train_path, names[4],))
			# 	getImgs(81, 100, url_Pants, postfix, test_path, names[4])
				
			# 	thread.start_new_thread(getImgs, (0, 80, url_Down_Jacket, postfix, train_path, names[5],))
			# 	getImgs(81, 100, url_Down_Jacket, postfix, test_path, names[5])
				
			# 	thread.start_new_thread(getImgs, (0, 80, url_Sweater, postfix, train_path, names[6],))
			# 	getImgs(81, 100, url_Sweater, postfix, test_path, names[6])
				
			# 	thread.start_new_thread(getImgs, (0, 80, url_Vest, postfix, train_path, names[7],))
			# 	getImgs(81, 100, url_Vest, postfix, test_path, names[7])

			# 	thread.start_new_thread(getImgs, (0, 80, url_Suit, postfix, train_path, names[8],))
			# 	getImgs(81, 100, url_Suit, postfix, test_path, names[8])
				
			# 	thread.start_new_thread(getImgs, (0, 80, url_Dress, postfix, train_path, names[9],))
			# 	getImgs(81, 100, url_Dress, postfix, test_path, names[9])
		while i <= 9:
			thread.start_new_thread(getImgs, (0, 80, urls[i], postfix, train_path, names[i],))
			getImgs(81, 100, urls[i], postfix, test_path, names[i])
			time.sleep(1)
			i += 1;
	except Exception, e:
		print Exception, ":", e

while 1:
	pass


#查找所有标签值为img，属性class为BDE_Image的数据，返回一个集合list

#imgList = soup.find_all('img',class_='J_ItemPic img')
'''
imgList = soup.find_all(re.compile(r'jpg'))
cnt = 0
for i in imgList:
	cnt = cnt + 1
print cnt
#print imgList
'''


#img width=".*?(jpg|png)
