def change_odd_upper_even_lower(my_string):
    final_str = ""
    for idx, char in enumerate(my_string):
        if (idx + 1) % 2:
            char = char.upper()
        else:
            char = char.lower()
        final_str = final_str + char
    return final_str


def change_odd_upper_even_lower_alt(my_string):
    indx = 0
    final_str = ""
    while indx < len(my_string):
        final_str = final_str + (my_string[indx].upper() if (indx + 1) % 2 else my_string[indx].lower())
        indx += 1
    return final_str


if __name__ == "__main__":
    my_string = "i'm Shwetha RedDy"
    final_str = change_odd_upper_even_lower(my_string)
    final_str1 = change_odd_upper_even_lower_alt(my_string)

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Odd characters to uppercase and even characters to lowercase")
    print("----------------------------------------------------------------")
    print("Original String: {}".format(my_string))
    print("Formatted String (Method 1): {}".format(final_str))
    print("Formatted String (Method 2): {}".format(final_str1))
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
