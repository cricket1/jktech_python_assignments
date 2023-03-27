def count_words(my_string):
    words = my_string.split()
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Words in a string")
    print("----------------------------------------------------------------")
    print("Original String: {}".format(my_string))
    print("----------------------------------------------------------------")
    print("No of words: {}".format(len(words)))
    print("words: {}".format(words))
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")


if __name__ == "__main__":
    my_string = "I'm    Shwetha Reddy. \n I'm not Ironman\t Yes, it is  true.\n\nNow let's go fight aliens\n"
    count_words(my_string)
