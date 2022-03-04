import cv2
import numpy as np


global radiuses
radiuses=[]
global const
global suma
suma=0


#const zawiera stosunki radiusiow
const=[0.68,0.8,1.07,0.92,1.05,1.38]




def comparsion(name):
        img=name
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 170, 200, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours)


def compare_radiuses(coin,radius):
    dec='und'
    if coin=='5zl':
        if radius>=const[0]*0.90 and radius<=const[0]*1.1:
            dec="10gr"
        if radius>=const[3]*0.90 and radius<=const[3]*1.1:
            dec="1zl"
    elif coin=='2zl':
        if radius>=const[1]*0.90 and radius<=const[1]*1.1:
            dec="10gr"
        if radius >= const[4] * 0.90 and radius <= const[4] * 1.1:
            dec = "1zl"
    elif coin=='1gr':
        if radius>=const[2]*0.90 and radius<=const[2]*1.1:
            dec="10gr"
        if radius >=const[5] * 0.90 and radius <= const[5] * 1.1:
            dec = "1zl"
    return dec
def change_image(img_path):
    img = cv2.imread(img_path, 1)
    resized_image = rescaleFrame(img)

    #resized_image = adjust_gamma(resized_image, gamma=1000)
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (21, 21), cv2.BORDER_DEFAULT)

    ret, thresh = cv2.threshold(blur, 100, 200, cv2.THRESH_BINARY)

    return thresh

def colors_values(tab):
    arr = []
    for row in tab:
        for (b, g, r) in row:
            if(b != 0 and g != 0 and r != 0):
                val = abs(int(b) - int(g)) + abs(int(b) - int(r)) + abs(int(r) - int(g))
                arr.append(val)

    arr = np.array(arr, dtype="int")
    avg = np.average(arr)
    return avg

def color_decision(mid_avg,ring_avg,r):
    global radiuses
    global suma
    prog=100
    if mid_avg<30 or ring_avg<30:
        dec="und"
    elif mid_avg<prog: #125 - pasuje tez
        if ring_avg<prog:
            dec="10/1"
        else:
            dec="2zl"
    else:
        if ring_avg>prog:
            dec="1gr"
        else:
            dec="5zl"
    if dec!="10/1":
        radiuses.append([dec,r])
    return dec

def banknote_decision(kolory):
    B = kolory[0]
    G = kolory[1]
    R = kolory[2]
    if max(kolory) == R:
        if min(kolory) == G:
            text = '20'
        else:
            text = '10'
    elif max(kolory) == B:
        text = '50'
    else:
        text = '100'
    return text


def banknotes(img_path):
    global suma
    img = cv2.imread(img_path, 1)
    resized_image = rescaleFrame(img)
    cv2.imshow('img', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    thresh=change_image(img_path)
    # EXTERNAL, TREE, LIST
    contours, hierarchies = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    czcionka = cv2.FONT_HERSHEY_SIMPLEX
    var=resized_image.shape
    obr=var[0]
    obr1=var[1]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 30 and h > 30:
            rogi = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(rogi) ==4 and w*h>15000 and w*h<=0.85*obr*obr1 :
                kolory = np.array(cv2.mean(resized_image[y + int(h / 4):y + int(3 * h / 4), x + int(4 * w / 10):x + int(3 * w / 4)])).astype(np.uint8)[0:3]
                text = banknote_decision(kolory)
                suma=suma+float(text)
                cv2.drawContours(resized_image, [rogi], -1, (0, 0, 255), 5)
                cv2.putText(resized_image, text, (x, y), czcionka, 0.7, (0, 255, 0), 2)
    return resized_image

def rescaleFrame(frame, scale = 0.4):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def coins(var):
    global suma
    global radiuses
    radiuses = []
    image=var
    img=image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    minDist = 100
    param1 = 300 #500
    param2 = 40 #200 #smaller value-> more false circles #dla 30 trzeba poprawic und
    minRadius = 5
    maxRadius = 50 #10
    skipped = []
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    maska_pom1=np.zeros(img.shape, dtype=np.uint8)
    maska_pom2=np.zeros(img.shape,dtype=np.uint8)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            x,y,r=i[0],i[1],i[2]
            mask = np.zeros(img.shape, dtype=np.uint8)
            mask = cv2.circle(mask, (x, y), int(r*0.95), (255, 255, 255), -1)
            mask1 =np.zeros(img.shape, dtype=np.uint8)
            mask1 = cv2.circle(mask1, (x, y), int(r*0.6), (255, 255, 255), -1)
            umask = mask*(mask1==0)
            masked = cv2.bitwise_and(img, mask)
            masked = cv2.bitwise_and(img, mask1)
            srodek = masked
            masked = cv2.bitwise_and(img, umask)
            pierscien = masked
            srodek1 = img[y - int(0.6 * r):y + int(0.6 * r), x - int(0.6 * r):x + int(0.6 * r)]
            maska_pom1=cv2.bitwise_and(img,mask1)
            maska_pom2=cv2.bitwise_and(img,umask)
            srodek_avg=colors_values(srodek[y-r:y+r,x-r:x+r])
            pierscien_avg=colors_values(pierscien[y-r:y+r,x-r:x+r])
            if(3.14*r*r>1800):
                dec=color_decision(srodek_avg,pierscien_avg,r)
                if dec=='10/1':
                    skipped.append([x,y,r])
                else:
                    if dec!='und':
                        if(dec=="5zl"): suma=suma+5.00
                        if(dec=="2zl"): suma=suma+2.00
                        #if(dec=="1zl"): suma=suma+1.00
                        if(dec=="1gr"): suma=suma+0.01
                        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                        cv2.putText(img, dec, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        #print("skipped:",skipped)
        #print("radiuses",radiuses)
        if skipped != []:
            for i in skipped:
                x, y, r = i[0], i[1], i[2]
                if radiuses!=[]:
                        coin=radiuses[0][0]
                        radius=radiuses[0][1]
                        dec=compare_radiuses(coin,r/radius)
                        if dec!='und':
                            if(dec=="1zl"): suma=suma+1.00
                            if(dec=="10gr"): suma=suma+0.10
                            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                            cv2.putText(img, dec, (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if skipped!=[] and radiuses==[]:
            maks = -10000
            mini = 100000
            for m in skipped:
                if m[2] > maks:
                    maks = m[2]
                if m[2] < mini:
                    mini = m[2]

            suma_pom=0
            if(maks/mini<=1.125):
                for m in skipped:
                    x,y,r=m[0],m[1],m[2]
                    srodek1 = img[y - int(0.7 * r):y + int(0.7 * r), x - int(0.7 * r):x + int(0.7 * r)]
                    mask1 = np.zeros(img.shape, dtype=np.uint8)
                    mask1 = cv2.circle(mask1, (x, y), int(r * 0.6), (255, 255, 255), -1)
                    masked = cv2.bitwise_and(img, mask1)

                    dec=comparsion(masked)
                    suma_pom=suma_pom+dec

                for m in skipped:
                    x, y, r = m[0], m[1], m[2]
                    if (suma_pom / len(skipped) > 20):
                        dec='1zl'
                        suma=suma+1.00
                        cv2.circle(img, (m[0], m[1]), m[2], (0, 255, 0), 2)
                        cv2.putText(img, dec, (m[0],m[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    else:
                        dec='10gr'
                        suma=suma+0.10
                        cv2.circle(img, (m[0], m[1]), m[2], (0, 255, 0), 2)
                        cv2.putText(img, dec, (m[0], m[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            else:
                for m in skipped:
                    if m[2] > 0.90 * maks and m[2] < 1.1 * maks:
                        dec = "1zl"
                        suma=suma+1.00
                    if m[2] > 0.9 * mini and m[2] < 1.1 * mini:
                        dec = "10gr"
                        suma=suma+0.10
                    cv2.circle(img, (m[0], m[1]), m[2], (0, 255, 0), 2)
                    cv2.putText(img, dec, (m[0],m[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.putText(img,"Total sum: "+str(suma)+" PLN",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.imshow('-', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




for i in range(1,15,1):
    suma = 0
    x=banknotes('latwe'+str(i)+'.jpg')
    coins(x)

for i in range(1,10,1):
    suma = 0
    x=banknotes('srednie'+str(i)+'.jpg')
    coins(x)

for i in range(1,10,1):
     suma = 0
     x=banknotes('trudne'+str(i)+'.jpg')
     coins(x)


