Python 3.11.1 (tags/v3.11.1:a7a450f, Dec  6 2022, 19:58:39) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # Python 3: Fibonacci series up to n
...  def fib(n):
...      a, b = 0, 1
...      while a < n:
...          print(a, end=' ')
...          a, b = b, a+b
...      print()
