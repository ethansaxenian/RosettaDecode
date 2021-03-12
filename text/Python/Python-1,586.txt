import tkinter,tkinter.simpledialog

root = tkinter.Tk()
root.withdraw()

number = tkinter.simpledialog.askinteger("Integer", "Enter a Number")
string = tkinter.simpledialog.askstring("String", "Enter a String")
