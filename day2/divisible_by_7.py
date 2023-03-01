def divisible_by(numbers, divisor):
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print("Numbers are {}\n".format(numbers))
    dividends = []
    for num in numbers:
        if num % divisor == 0:
            dividends.append(num)

    print("Numbers divisible by {} are {}".format(divisor, dividends))
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")


if __name__ == "__main__":
    numbers = [3, 4, 7, 21, 11, 49, 84]
    divisor = 7
    divisible_by(numbers, divisor)
