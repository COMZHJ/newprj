import random


print(random.randint(1, 5))


def add(x, y):
    return x + y


class Father(object):
    # 双下划线开始和结束，表示系统属性和方法
    # 双下划线开始，表示Private；单下划线开始，表示Protect
    def __init__(self, name):
        print('__init__')
        self.name = name

    def __del__(self):
        print('__del__')

    def show(self):
        print('name：', self.name)


if __name__ == '__main__':
    f = Father('张三')
    print(f, f.name, id(f))
    f.show()


class Son(Father):
    def __init__(self, name, age):
        super().__init__(name)
        self.age = age

    def show(self):
        print(f'name：{self.name}；age：{self.age}')


if __name__ == '__main__':
    s = Son('李四', 18)
    s.show()

