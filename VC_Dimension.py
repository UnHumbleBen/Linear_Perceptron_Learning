import math


def nCr(n, r):
    if (r > n):
        return 0
    f = math.factorial
    return f(n) // f(r) // f(n - r)


def string(N, m_H):
    print("m_H(" + str(N) + ") = " + str(m_H))


def experiment(q):
    N = 1
    m_H = 2
    string(N, m_H)

    while m_H == 2 ** N:
        m_H = 2 * m_H - nCr(N, q)
        N += 1
        string(N, m_H)

    print("For q = " + str(q) + ", VC dimension is " + str(N - 1))

experiment(100)