## Script written to convert the iris dataset in our required format
f = open('iris_dataset','r')
r = f.readlines()
g = open('irisnew','w')
s = ""
for line in r:
    # print(line)
    arr = line.split(' ')
    arr[3] = arr[3][:3]
    print(arr)

    for i in range(0,4):
        # print(arr[i])
        s += arr[i]
        s += " "
# print(s)
g.write(s)