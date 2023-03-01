import os
import time


def get_file_properties(file_path):
    print("------------------------------------------------------")
    if os.path.exists(file_path):
        print("Properties of File - {}".format(file_path))
        print("------------------------------------------------------")
        print("File Name: {}".format(os.path.basename(file_path)))

        file_size = os.path.getsize(file_path)
        print("File Size: {} Bytes".format(file_size))

        modified_time_seconds = os.path.getmtime(file_path)
        modified_time =time.ctime(modified_time_seconds)

        print("Last Modified: {}".format(modified_time))

        with open(file_path, 'r') as file:
            data = file.readlines()
            lines = len(data)
            print('Total No. Of Lines:', lines)
            file.close()

        with open(file_path, 'r') as file:
            data = file.read()
            number_of_characters = len(data)
            print('Total No. Of Characters:', number_of_characters)
            file.close()
    else:
        print("File not found - {}".format(file_path))
    print("------------------------------------------------------\n")


if __name__ == "__main__":
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("FILE PROPERTIES")
    get_file_properties("../common/student_comma.dat")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
