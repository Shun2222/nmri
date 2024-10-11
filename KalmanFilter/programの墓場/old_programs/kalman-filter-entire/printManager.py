
last_len = None

def printline(s):
    print_clear()
    print('\r'+s, end='')
    last_len = len(s)

def clear(n=30):
    n = last_len if last_len!=None else n
    print('\r' + ' '*n + '\r', end='')
print_clear = clear

