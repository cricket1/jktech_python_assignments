def is_prime(num):
    is_prime_no = True
    if num > 1:
        for i in range(2, int(num / 2) + 1):
            if (num % i) == 0:
                is_prime_no = False
                break
    else:
        is_prime_no = False

    return is_prime_no


def find_prime_numbers(numbers):
    print("####################################################")
    print("Numbers: {}\n".format(numbers))
    prime_nos = []
    for num in numbers:
        if is_prime(num):
            prime_nos.append(num)

    if len(prime_nos):
        print("Prime Numbers: {}".format(prime_nos))
    else:
        print("No prime numbers found")
    print("####################################################")


if __name__ == "__main__":
    numbers = [1, 3, 4, 7, 21, 11, 49, 84, 131]
    find_prime_numbers(numbers)
