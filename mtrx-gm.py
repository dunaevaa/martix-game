import numpy as np
from copy import deepcopy
from itertools import product

MAX_MODE = 'MAX'  
MIN_MODE = 'MIN'  

class SimplexMethod:
    def __init__(self, c, a, b, mode):
        self.main_variables_count = a.shape[1]  # количество переменных
        self.restrictions_count = a.shape[0]  # количество ограничений
        self.variables_count = self.main_variables_count + self.restrictions_count  # количество дополнительных переменных
        self.mode = mode  # запоминаем режим работы

        self.c = np.concatenate([c, np.zeros((self.restrictions_count + 1))])  # коэффициенты функции
        self.f = np.zeros((self.variables_count + 1))  # значения функции F
        self.basis = [i + self.main_variables_count for i in range(self.restrictions_count)]  # индексы базисных переменных
        self.task = {'a': deepcopy(a), 'b': deepcopy(b), 'c': deepcopy(c)}  # сохраняем начальную задачу

        self.init_table(a, b)

    # инициализация таблицы
    def init_table(self, a, b):
        self.table = np.zeros((self.restrictions_count, self.variables_count + 1))  # коэффициенты таблицы

        for i in range(self.restrictions_count):
            for j in range(self.main_variables_count):
                self.table[i][j] = a[i][j]
                self.table[i][j + self.main_variables_count] = int(i == j)

            self.table[i][-1] = b[i]

    # получение строки с максимальным по модулю отрицательным значением b
    def get_negative_b_row(self):
        row = -1

        for i, a_row in enumerate(self.table):
            if a_row[-1] < 0 and (row == -1 or abs(a_row[-1]) > abs(self.table[row][-1])):
                row = i

        return row

    # получение столбца с максимальным по модулю элементом в строке
    def get_negative_b_column(self, row):
        column = -1

        for i, aij in enumerate(self.table[row][:-1]):
            if aij < 0 and (column == -1 or abs(aij) > abs(self.table[row][column])):
                column = i

        return column

    # удаление отрицательных свободных коэффициентов
    def remove_negative_b(self):
        while True:
            row = self.get_negative_b_row()  # ищем строку, в которой находится отрицательное b

            if row == -1:  # если не нашли такую строку
                return True  # то всё хорошо

            column = self.get_negative_b_column(row)  # ищем разрешающий столбец

            if column == -1:
                return False  # не удалось удалить

            self.gauss(row, column)  # выполняем исключение гаусса
            self.calculate_f()

    # выполнение шага метода гаусса
    def gauss(self, row, column):
        self.table[row] /= self.table[row][column]

        for i in range(self.restrictions_count):
            if i != row:
                self.table[i] -= self.table[row] * self.table[i][column]

        self.basis[row] = column  # делаем переменную базисной

    # расчёт значений F
    def calculate_f(self):
        for i in range(self.variables_count + 1):
            self.f[i] = -self.c[i]

            for j in range(self.restrictions_count):
                self.f[i] += self.c[self.basis[j]] * self.table[j][i]

    # расчёт симплекс-отношений для столбца column
    def get_relations(self, column):
        q = []

        for i in range(self.restrictions_count):
            if self.table[i][column] == 0 or self.table[i][column] < 0:
                q.append(np.inf)
            else:
                q_i = self.table[i][-1] / self.table[i][column]
                q.append(q_i if q_i >= 0 else np.inf)

        return q

    # получение решения
    def get_solve(self):
        x = np.zeros((self.variables_count))

        # заполняем решение
        for i in range(self.restrictions_count):
            x[self.basis[i]] = self.table[i][-1]

        return x  # возвращаем полученное решение

    # получение решения
    def get_task_solve(self):
        x = np.zeros((self.main_variables_count))

        # заполняем решение
        for i in range(self.restrictions_count):
            if self.basis[i] < self.main_variables_count:
                x[self.basis[i]] = self.table[i][-1]

        return x, self.f[-1]  # возвращаем полученное решение

    # решение
    def solve(self, debug):
        self.calculate_f()
        self.print_task()

        if debug:
            print('\n')
            self.print_table()

        if not self.remove_negative_b():
            print('Solve does not exist')
            return False

        iteration = 1

        while True:
            self.calculate_f()

            if debug:
                print('\n')
                self.print_table()

            if all(fi >= 0 if self.mode == MAX_MODE else fi <= 0 for fi in self.f[:-1]):  # если план оптимален
                break  # то завершаем работу

            column = (np.argmin if self.mode == MAX_MODE else np.argmax)(self.f[:-1])  # получаем разрешающий столбец
            q = self.get_relations(column)  # получаем симплекс-отношения для найденного столбца

            if all(qi == np.inf for qi in q):  # если не удалось найти разрешающую строку
                print('Solve does not exist')  # сообщаем, что решения нет
                return False

            row = np.argmin(q)

            self.gauss(row, column)  # выполняем исключение гаусса
            iteration += 1

        return True  # решение есть

    # вывод симплекс-таблицы
    def print_table(self):
        print('     |' + '      S0  |' + ''.join(['     x%-3d |' % (i + 1) for i in range(self.main_variables_count)]))

        for i in range(self.restrictions_count):
            print('%4s |' % ('x' + str(self.basis[i] + 1)) + ' %8.3f |' % self.table[i][-1] + ''.join([' %8.3f |' % aij for j, aij in enumerate(self.table[i][:self.main_variables_count])]))

        print('   F |' + ' %8.3f |' % self.f[-1] + ''.join([' %8.3f |' % aij for aij in self.f[:self.main_variables_count]]))

    # вывод коэффициента
    def print_coef(self, ai, i):
        if ai == 1:
            return 'x%d' % (i + 1)

        if ai == -1:
            return '-x%d' % (i + 1)

        return '%.3fx%d' % (ai, i + 1)

    # вывод задачи
    def print_task(self, full = False):
        print(' + '.join(['%.3fx%d' % (ci, i + 1) for i, ci in enumerate(self.c[:self.main_variables_count]) if ci != 0]), '-> ', self.mode)

        for row in self.table:
            if full:
                print(' + '.join([self.print_coef(ai, i) for i, ai in enumerate(row[:self.variables_count]) if ai != 0]), '=', row[-1])
            else:
                print(' + '.join([self.print_coef(ai, i) for i, ai in enumerate(row[:self.main_variables_count]) if ai != 0]), '<=', row[-1])

def print_solve(x, f, label_x, label_f, label_inv):
    print(", ".join(['%s%d = %.3f' % (label_x, i + 1, xi) for i, xi in enumerate(x)]), end=', ')
    print('%s = %.3f, %s = %.3f' % (label_f, f, label_inv, 1 / f))
    print("optimal:", ", ".join(['x%d = %.3f' % (i + 1, xi / f) for i, xi in enumerate(x)]))

def main():

    matrix = [
        [19, 6, 8, 2, 7],
        [7, 9, 2, 0, 12],
        [3, 18, 11, 9, 10],
        [19, 10, 6, 19, 4],
    ]

    nj = len(matrix)  # количество стратегий для игрока A
    ni = len(matrix[0])  # количество стратегий для игрока B

    aa = np.array([[-matrix[i][j] for i in range(nj)] for j in range(ni)])
    ba = np.array([-1 for _ in range(ni)])
    ca = np.array([1 for _ in range(nj)])
    modea = MIN_MODE

    ab = np.array([[matrix[i][j] for j in range(ni)] for i in range(nj)])
    bb = np.array([1 for _ in range(nj)])
    cb = np.array([1 for _ in range(ni)])
    modeb = MAX_MODE

    debug = True

    print("Solve for player A:")
    simplex_a = SimplexMethod(ca, aa, ba, modea)
    simplex_a.solve(debug)
    ua, wa = simplex_a.get_task_solve()
    print_solve(ua, wa, 'u', 'W', 'g')

    print("\nSolve for player B:")
    simplex_b = SimplexMethod(cb, ab, bb, modeb)
    simplex_b.solve(debug)
    vb, zb = simplex_b.get_task_solve()
    print_solve(vb, zb, 'v', 'Z', 'h')

if __name__ == '__main__':
    main()
