import pandas as pd

read= pd.read_csv("titanic.csv") # dọc file
# print(read.to_string()) in  file csv
# print (read.head(10))   hiện thị N hàng từ đinh bảng
# print (read.tail(50))   hiện thị N hàng dưới đấy bảng
# print(read.info)         hiện thị thông tin bang

# -------------------------------------------------------
-----------------------
 làm sach dữ liệu     -
----------------------
# dropcell= read.dropna()    #loại bổ hàng cell rỗng và trả dữ liệu mới 
# read.dropna(inplace= True) # loại bỏ hàng cell rõng và cập nhât dữ liệu mới luôn (file gốc  )
# read.fillna(123, inplace=True) # thay thế gfias trị của NAN (cell rỗng)
-------------------------------------------------------------------------------------------------
# x= read['calories'].mean()                |
# read['calories'].fillna(x, inplace= True) |tính giá trị trung bình sau đó thay vào cột 'calories' chõ nào có NAn thi thây và cap nhật bằng inplace
-----------------------------------------------------------------------------------------------------
# x=read['calories'].median()                |Giá trị ở giữa sau khi sắp xếp tăng dần
# read['calories'].fillna(x , inplace= True) | 
-----------------------------------------------------------------------------------------------------
# x=read["calories"].mode()[0]                 | láy giá trị đầu tiên và tim giá trị xuất hiện nhiều nhất
# read["calories"].fillna(x, inplace=True)     | thay th thay thế và cập nhật
---------------------------------------------------------------------------------------------
# read['date']= pd.to_datetime(read['date'])  |  chuyển đổi định dạng
-------------------------------------------------------------------------------------------------
# read.dropna(subset=['date'],inplace=True)    | subset : kiểm tra cột date hàng nào chứa lõi thì loại và cập nhật
------------------------------------------------------------------------------------------------------------------
# read.loc[4, 'Survived']=45                   | thay thế  dữ liệu lỗi tại hàng 7 bàng gia tị 45
------------------------------------------------------------------------------------------------------
# for x in read.index:
#  if read.loc[x, 'Suvived'] >= 2:          | thay thế nhiều hàng ví dụ nếu cột suvived có hàng nào trên 2 hoạc bằng 2
#     read.loc[x, 'Suvived'] = 0            | thì nó sẽ thay thế bằng 0
---------------------------------------------------------------------------------------------------------
# for x in read.index:                  | loại bỏ dữ liệu lỗi  ở cột nào đó sau đó tim dữ liệu lỗi và xóa bỏ nó 
#  if read.loc[x, 'info'] >=3:
#     read.drop(x,  inplace=True)           | và cập nhật
----------------------------------------------------------------------------------------------- 
# print(read.duplicated)            | hiện thị true ở hàng nào trùng lặp
-------------------------------------------------------------------------------------------------------
# read.drop_duplicates (inplace=True)  | loại bỏ dữ liệu trùng lặp  và cập nhật


print(read.to_string())
