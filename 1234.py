## LOAD THU VIEN VA MODUL CAN THIET
import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt
from PIL import Image

pytesseract.pytesseract.get_languages = r'C:\Program Files\Tesseract-OCR\tessdata'
# pytesseract dùng để định dạng hình ảnh thành văn bản
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#pytesseract.pytesseract.tesseract_cmd = r'E:\PYTHON\venv\Lib\site-packages\Tesseract-OCR\tesseract.exe'
#pytesseract.pytesseract.get_languages = r'E:\PYTHON\venv\Lib\site-packages\Tesseract-OCR\tessdata'
#----------------------DOC HINH ANH - TACH HINH ANH NHAN DIEN--------------------
img = cv2.imread('8.jpg')
cv2.imshow('HINH ANH GOC', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# chuyển ảnh bình thường sang ảnh xam
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
#adaptiveThreshold dùng cho ảnh có ánh sáng không đầy đủ. phương thức này tính giá trị trung bình của các n điểm ảnh xung quanh pixle đó rồi trừ cho C chữ ko lấy ngưỡng có đỉnh( n là số lẻ, c là số nguyên bất kì)
#.THRESH_BINARY : nếu giá trị của pixel lớn hơn ngưỡng thì gán bằng maxval
#ADAPTIVE_THRESH_GAUSSIAN_C Nhân giá trị xung quanh điểm cần xét với trọng số gauss rồi tính trung bình của nó, sau đó trừ đi giá trị hằng số C

contours,h = cv2.findContours(thresh,1,2)
# findContours : trích xuất các đường nét
# contours là tập hợp các ddieemr liên tục thành 1 đường cong và ko có khoảng hơ trong đường cong
largest_rectangle = [0,0]
#tạo hình chữ nhật
for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
# cv2.approxPolDP là 1 hàm biểu thị Thuật toán sử dụng đệ quy để phân chia các điểm có trong Contour    
    if len(approx)==4:
        area = cv2.contourArea(cnt)
        if area > largest_rectangle[0]:
            largest_rectangle = [cv2.contourArea(cnt), cnt, approx]
x,y,w,h = cv2.boundingRect(largest_rectangle[1])
# tìm ranh giới đối tượng
image=img[y:y+h,x:x+w]
cv2.drawContours(img,[largest_rectangle[1]],0,(0,255,0),8)

cropped = img[y:y+h, x:x+w]
cv2.imshow('DANH DAU DOI TUONG', img)

cv2.drawContours(img,[largest_rectangle[1]],0,(255,255,255),18)

#--------------------- DOC HINH ANH CHUYEN THANH FILE TEXT-----------------------------


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.cvtColor : chuyeenr ddooir heej mau
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# cv2.THRESH_BINARY_INV :Ngưỡng nhị phân đảo ngược. Có thể hiểu là nó sẽ đảo ngược lại kết quả của THRESH_BINARY.
#cv2.THRESH_OTSU; Sử dụng thuật toán Otsu để xác định giá trị ngưỡng



cv2.imshow('CROP', thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
# loại bỏ nhiễu nhưng vẫn giữ nghuyên đc kích thước đối tượng

text = pytesseract.image_to_string(cropped, lang='eng')
# đọc ảnh thành text
print("Number is: ", text)
cv2.waitKey(0)

