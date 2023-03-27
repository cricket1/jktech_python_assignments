import datetime


def epoch_to_datetime():
    print("********************************")
    print("EPOCH Time to DateTime")
    print("--------------------------------")
    epoch_time = 473398200
    print("EPOCH Time: {}".format(epoch_time))
    date_conv = datetime.datetime.fromtimestamp(epoch_time)
    print("DateTime: {}".format(date_conv))
    print("********************************")


if __name__ == "__main__":
    epoch_to_datetime()
