from random import randint

def coin_flip(n):
    lst = []
    eagle, tails = 0, 0
    for _ in range(n):
        val = randint(0, 1)
        if val == 0:
            eagle += 1
        else:
            tails += 1

    lst.append(eagle)
    lst.append(tails)
    return lst

def main():
    n = int(input("Введите число повторений: "))
    line = input("Введите, что нужно Орел/Решка: ")
    lst = coin_flip(n)

    if line == "Орел":
        print(lst[0] / n)
        print(lst[0])
    if line == "Решка":
        print(lst[1] / n)
        print(lst[1])

if __name__ == "__main__":
    main()