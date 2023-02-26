list1= [1,2,1,1,4,5 ] 
print(list1)

count_of_1 = 2
current_1_count=0
index_count = 0
for num in list1:
    if num==1:
        current_1_count += 1

    if current_1_count ==2:
        break
    
    index_count += 1

print('2nd 1 index count is {}'.format(index_count))
list1.pop(index_count)
print(list1)



