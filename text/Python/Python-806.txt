import calendar
from functools import reduce
c= calendar.Calendar()
fridays={}
year=input("year")
add=list.__add__
for day in reduce(add,reduce(add,reduce(add,c.yeardatescalendar(int(year))))):

    if "Fri" in day.ctime() and year in day.ctime():
        month,day=str(day).rsplit("-",1)
        fridays[month]=day

for item in sorted((month+"-"+day for month,day in list(fridays.items())),
                   key=lambda x:int(x.split("-")[1])):
    print(item)
