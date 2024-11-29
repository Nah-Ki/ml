import csv

def loadCsv(filename):
    with open(filename, "rt") as file:
        lines = csv.reader(file)
        dataset = list(lines)
    return dataset

attributes = ['Sky', 'Temp', 'Humidity', 'Wind', 'Water', 'Forecast']
print(attributes)

filename = "Weather.csv"
dataset = loadCsv(filename)
print(dataset)

target = ['Yes', 'Yes', 'No', 'Yes']
print(target)

num_attributes = len(attributes)
hypothesis = ['0'] * num_attributes
print(hypothesis)

print("The Hypothesis are")

for i in range(len(target)):
    if target[i] == 'Yes':
        for j in range(num_attributes):
            if hypothesis[j] == '0':
                hypothesis[j] = dataset[i][j]
            elif hypothesis[j] != dataset[i][j]:
                hypothesis[j] = '?'
        print(i + 1, '=', hypothesis)

print("Final Hypothesis")
print(hypothesis)
