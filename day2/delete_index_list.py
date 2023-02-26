list1 = [1, 4, 10, 30, 60, 70, 80 , 100]
delete_index_list = [3, 7, 4]
delete_index_list.sort(reverse=True)
for i in delete_index_list:
   list1.pop(i)

print(list1)
