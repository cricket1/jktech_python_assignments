import os


def read_file_raw(file_path):
    if os.path.exists(file_path):
        print("*****************************************************")
        print("Raw Data from file - {}".format(file_path))
        print("*****************************************************")
        file = open(file_path, "r")
        print(file.read())
        print("*****************************************************\n")
        file.close()
    else:
        print("File not found - {}".format(file_path))


def read_file(file_path, delimiter):
    if os.path.exists(file_path):
        print("*****************************************************")
        print("Data from file - {}".format(file_path))
        print("*****************************************************")
        with open(file_path, "r") as file:
            data = file.readlines()
            for line in data:
                line = line.strip()
                word = line.split(delimiter)
                print(word)
        print("*****************************************************\n")
    else:
        print("File not found - {}".format(file_path))


if __name__ == "__main__":
    # read_file_raw("../common/student_comma.dat")

    read_file("../common/student_comma.dat", ",")
    read_file("../common/student_double_semi.dat", ";;")
    read_file("../common/student_semi.dat", ";")
