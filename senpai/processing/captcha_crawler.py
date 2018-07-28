from bs4 import BeautifulSoup
import urllib3
import time


url = 'https://dp.saic-gm.com/'
img_url = 'https://dp.saic-gm.com/am/oauth/captcha.jpg?uuid=%s'
local_path = '/Users/pzn666/Documents/data_enlight/data/captcha/' + \
             'tmp_data/%s.jpg'
http = urllib3.PoolManager()
for idx in range(1000):
    start_time = time.time()
    response = http.request('GET', url)
    soup = BeautifulSoup(response.data)
    captcha = soup.find('img', {'id': 'verifyCode'})
    uuid = captcha['src'][17:]
    captch_url = img_url % uuid
    img = http.request('GET', captch_url, preload_content=False)
    save_path = local_path % uuid
    with open(save_path, 'wb') as f:
        while True:
            data = img.read(1024)
            if not data:
                break
            f.write(data)
    img.release_conn()
    print('Done img %i download, spend %f' % (idx, time.time() - start_time))
