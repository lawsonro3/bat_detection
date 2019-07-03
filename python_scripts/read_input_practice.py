import csv

readlocation = '/Users/icunitz/Desktop/bat_detection/'

objecttype1_ = []
objecttype2_ = []
distance1_ = []
distance2_ = []
filename1_ = []
filename2_ = []
frame1_ = []
frame2_ = []

def findelements(inputlist, Duplicate=False):
    outputlist = []
    for element in inputlist:
        if element.startswith('#'):
            continue
        elif element == '':
            continue
        else:
            outputlist.append(element)
    if not Duplicate:
        if len(outputlist) == 1:
            return outputlist[0]
        else:
            print('Error: Elements in column =/= 1')
    else:
        if len(outputlist) == 2:
            return outputlist
        else:
            print('Error: Elements in column =/= 2')

def findfilelist(objecttype, distance):
    filelist = ''
    if objecttype == 'airplanes':
        if distance == 'close':
            filelist = row[4]
        else:
            filelist = row[5]
    elif objecttype == 'bats':
        if distance == 'close':
            filelist = row[6]
        else:
            filelist = row[7]
    elif objecttype == 'birds':
        if distance == 'close':
            filelist = row[8]
        else:
            filelist.append(row[9])
    elif objecttype == 'insects':
        if distance == 'close':
            filelist = row[10]
    return filelist

filename1_ = []
filename2_ = []

with open(readlocation + 'input.csv', newline='') as csvfile:
    inputreader = csv.reader(csvfile)
    for row in inputreader:
        objecttype1_.append(row[0])
        distance1_.append(row[1])
        objecttype2_.append(row[2])
        distance2_.append(row[3])

        frame1_.append(row[11])
        frame2_.append(row[12])

objecttype1 = findelements(objecttype1_)
objecttype2 = findelements(objecttype2_)
distance1 = findelements(distance1_)
distance2 = findelements(distance2_)

with open(readlocation + 'input.csv', newline='') as csvfile:
    inputreader = csv.reader(csvfile)
    for row in inputreader:
        filename1_.append(findfilelist(objecttype1, distance1))
        filename2_.append(findfilelist(objecttype2, distance2))

frame1 = findelements(frame1_)
frame2 = findelements(frame2_)

if filename1_ == filename2_:
    filenames_ = findelements(filename1_, Duplicate=True)
    filename1 = filenames_[0]
    filename2 = filenames_[1]
else:
    filename1 = findelements(filename1_)
    filename2 = findelements(filename2_)

print(objecttype1, distance1, filename1, frame1, objecttype2, distance2, filename2,  frame2)
