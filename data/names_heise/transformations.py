with open("nam_dict.txt", "rb") as text:
    temp = text.read()
    temp = temp.split(b"\n")

# Strip the data
temp = [i[3:-3] for i in temp]

# Find out where the data starts
for idx, line in enumerate(temp):
    if b"begin of name list" in line:
        data_idx = idx
        data_idx += 2
    if b"list of countries" in line:
        countries_idx = idx
        countries_idx += 7

# Read the index of each line
idx_country_map = {}
for counter, idx in enumerate(range(countries_idx, data_idx-19)):
    line = temp[idx]
    # These are the lines that contain the countries
    if counter % 3 == 0:
        country = line.strip()
    # These are the lines that contain the index
    if counter % 3 == 1:
        country_idx = line.find(b"|")
        idx_country_map[country_idx] = country

# For every line we are going to need to read the name
# Next, we will search every number and its index
byte_str_numbers = [b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8", b"9"]
name_importance_map = {}
for idx in range(data_idx, data_idx+10000):
    line = temp[idx]
    print("\n", line)

    name = line[:10]
    name = name.strip()

    country_importance = []

    for number in byte_str_numbers:
        country_idx = line.find(number)
        importance = int(number)
        if country_idx != -1:
            country = idx_country_map[country_idx]
            country_importance.append([country, importance])

    name_importance_map[name] = country_importance

for i, j in name_importance_map.items():
    print(i, "\t", j)
